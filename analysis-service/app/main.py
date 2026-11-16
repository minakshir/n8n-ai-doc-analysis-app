import hashlib
import logging
import re
import secrets
import tempfile
from pathlib import Path
from typing import BinaryIO, Literal

from fastapi import Depends, FastAPI, File, Form, Header, Query, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from . import __version__
from .config import Settings, get_settings
from .detect import detect_kind
from .embeddings import OllamaEmbedder
from .errors import ServiceError
from .kb import KnowledgeBase
from .parsers import parse_file
from .profiling import build_profile, compact_summary
from .storage import Storage

log = logging.getLogger("analysis")
app = FastAPI(title="Data Agent Analysis Service", version=__version__)


@app.exception_handler(ServiceError)
async def _service_error(_: Request, exc: ServiceError) -> JSONResponse:
    return JSONResponse(exc.to_dict(), status_code=exc.status)


@app.exception_handler(RequestValidationError)
async def _validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
    fields = ", ".join(".".join(str(p) for p in e["loc"][1:]) or "body" for e in exc.errors())
    err = ServiceError("BAD_REQUEST", f"Missing or invalid field(s): {fields}.", status=422)
    return JSONResponse(err.to_dict(), status_code=err.status)


@app.exception_handler(Exception)
async def _unexpected(_: Request, exc: Exception) -> JSONResponse:
    log.exception("unhandled error", exc_info=exc)
    err = ServiceError("INTERNAL", "Unexpected server error.", status=500)
    return JSONResponse(err.to_dict(), status_code=500)


def get_storage(settings: Settings = Depends(get_settings)) -> Storage:
    return Storage(settings.data_dir)


def get_embedder(settings: Settings = Depends(get_settings)) -> OllamaEmbedder:
    return OllamaEmbedder(settings.ollama_url, settings.embed_model)


def get_kb(
    settings: Settings = Depends(get_settings),
    embedder: OllamaEmbedder = Depends(get_embedder),
) -> KnowledgeBase:
    return KnowledgeBase(settings.data_dir, embedder)


def require_token(
    x_service_token: str | None = Header(default=None),
    settings: Settings = Depends(get_settings),
) -> None:
    if not x_service_token or not secrets.compare_digest(
        x_service_token.encode(), settings.service_token.encode()
    ):
        raise ServiceError("UNAUTHORIZED", "Missing or invalid X-Service-Token header.", status=401)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "version": __version__}


@app.post("/ingest", dependencies=[Depends(require_token)])
def ingest(
    session_id: str = Form(...),
    file: UploadFile = File(...),
    settings: Settings = Depends(get_settings),
    store: Storage = Depends(get_storage),
) -> dict:
    store.session_dir(session_id)  # validates before any work
    filename = safe_filename(file.filename)
    with tempfile.TemporaryDirectory(dir=store.tmp_dir) as tmp:
        source = Path(tmp) / f"upload{Path(filename).suffix.lower()}"
        size = copy_limited(file.file, source, settings.max_upload_bytes)
        if size == 0:
            raise ServiceError("EMPTY_FILE", "The uploaded file is empty.")
        kind = detect_kind(source, filename)
        try:
            parsed = parse_file(source, kind, filename, settings.max_pdf_pages)
        except ServiceError:
            raise
        except Exception as e:
            log.exception("parse failed for %s", kind)
            raise ServiceError("PARSE_FAILED", f"Could not read this {kind} file.", hint=str(e)[:200])
        dataset_id = store.new_dataset_id()
        profile = build_profile(dataset_id, filename, kind, size, parsed)
        store.save(session_id, dataset_id, source, parsed, profile)
    return profile


@app.get("/sessions/{session_id}/profiles", dependencies=[Depends(require_token)])
def profiles(
    session_id: str,
    format: Literal["full", "compact"] = "full",
    store: Storage = Depends(get_storage),
) -> dict:
    items = store.list_profiles(session_id)
    if format == "compact":
        return {
            "session_id": session_id,
            "datasets": [{"dataset_id": p["dataset_id"], "file_name": p["file_name"]} for p in items],
            "summary": compact_summary(items),
        }
    return {"session_id": session_id, "datasets": items}


@app.get("/sessions/{session_id}/datasets/{dataset_id}/text", dependencies=[Depends(require_token)])
def dataset_text(
    session_id: str,
    dataset_id: str,
    offset: int = Query(0, ge=0),
    limit: int = Query(20000, ge=1, le=200000),
    store: Storage = Depends(get_storage),
) -> dict:
    text = store.read_text(session_id, dataset_id)
    if text is None:
        raise ServiceError("NO_TEXT", "This dataset has no extracted text.", status=404)
    chunk = text[offset:offset + limit]
    return {
        "dataset_id": dataset_id,
        "total_chars": len(text),
        "offset": offset,
        "text": chunk,
        "truncated": offset + len(chunk) < len(text),
    }


@app.delete("/sessions/{session_id}", dependencies=[Depends(require_token)])
def delete_session(session_id: str, store: Storage = Depends(get_storage)) -> dict:
    return {"deleted": store.delete_session(session_id)}


@app.post("/admin/cleanup", dependencies=[Depends(require_token)])
def cleanup(settings: Settings = Depends(get_settings), store: Storage = Depends(get_storage)) -> dict:
    return {"removed_sessions": store.cleanup(settings.retention_days)}


KB_KINDS = {"txt", "pdf"}


@app.post("/kb/documents", dependencies=[Depends(require_token)])
def kb_add(
    file: UploadFile = File(...),
    settings: Settings = Depends(get_settings),
    store: Storage = Depends(get_storage),
    kb: KnowledgeBase = Depends(get_kb),
) -> dict:
    filename = safe_filename(file.filename)
    with tempfile.TemporaryDirectory(dir=store.tmp_dir) as tmp:
        source = Path(tmp) / f"upload{Path(filename).suffix.lower()}"
        size = copy_limited(file.file, source, settings.max_upload_bytes)
        if size == 0:
            raise ServiceError("EMPTY_FILE", "The uploaded file is empty.")
        kind = detect_kind(source, filename)
        if kind not in KB_KINDS:
            raise ServiceError(
                "UNSUPPORTED_FILE",
                "The knowledge base accepts .txt and .pdf documents.",
                hint="Tables (CSV, Excel) are analysed through /ingest, not retrieved as text.",
            )
        try:
            parsed = parse_file(source, kind, filename, settings.max_pdf_pages)
        except ServiceError:
            raise
        except Exception as e:
            log.exception("parse failed for %s", kind)
            raise ServiceError("PARSE_FAILED", f"Could not read this {kind} file.", hint=str(e)[:200])
        sha256 = hashlib.sha256(source.read_bytes()).hexdigest()
    doc = kb.add(filename, kind, sha256, size, parsed.pages, parsed.text or "")
    return {**doc, "warnings": parsed.warnings}


@app.get("/kb/documents", dependencies=[Depends(require_token)])
def kb_list(kb: KnowledgeBase = Depends(get_kb)) -> dict:
    return {"embed_model": kb.embedder.model, "documents": kb.documents()}


@app.delete("/kb/documents/{doc_id}", dependencies=[Depends(require_token)])
def kb_delete(doc_id: str, kb: KnowledgeBase = Depends(get_kb)) -> dict:
    if not kb.delete(doc_id):
        raise ServiceError("NOT_FOUND", f"Document {doc_id} is not in the knowledge base.", status=404)
    return {"deleted": True}


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(5, ge=1, le=20)


@app.post("/kb/search", dependencies=[Depends(require_token)])
def kb_search(req: SearchRequest, kb: KnowledgeBase = Depends(get_kb)) -> dict:
    if not req.query.strip():
        raise ServiceError("BAD_REQUEST", "The query is empty.", status=422)
    return kb.search(req.query.strip(), req.top_k)


def safe_filename(name: str | None) -> str:
    base = Path(name or "").name
    base = re.sub(r"[\x00-\x1f\x7f]", "", base).strip()
    return base[:200] or "upload"


def copy_limited(src: BinaryIO, dest: Path, limit: int) -> int:
    total = 0
    with dest.open("wb") as out:
        while chunk := src.read(1024 * 1024):
            total += len(chunk)
            if total > limit:
                raise ServiceError(
                    "FILE_TOO_LARGE",
                    f"The file exceeds the {limit // (1024 * 1024)} MB limit.",
                    status=413,
                )
            out.write(chunk)
    return total
