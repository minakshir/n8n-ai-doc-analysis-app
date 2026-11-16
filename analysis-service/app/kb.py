"""Persistent knowledge base: documents, chunks, embeddings and a keyword index.

    {data_dir}/kb/kb.sqlite

Not tied to a session and never touched by retention cleanup. Search is hybrid:
cosine similarity over the embeddings plus FTS5 BM25 over the chunk text, merged
with reciprocal rank fusion. Only chunks embedded with the configured model are
searched; documents embedded with another model are reported, not mixed in.
"""

import re
import secrets
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .chunking import chunk_text
from .embeddings import OllamaEmbedder
from .errors import ServiceError

DOC_RE = re.compile(r"^kb_[0-9a-f]{12}$")
CANDIDATES = 50
RRF_K = 60

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value INTEGER NOT NULL);
INSERT OR IGNORE INTO meta VALUES ('version', 0);
CREATE TABLE IF NOT EXISTS documents (
    doc_id      TEXT PRIMARY KEY,
    file_name   TEXT NOT NULL,
    kind        TEXT NOT NULL,
    sha256      TEXT NOT NULL UNIQUE,
    size_bytes  INTEGER NOT NULL,
    pages       INTEGER,
    chunks      INTEGER NOT NULL,
    embed_model TEXT NOT NULL,
    created_at  TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS chunks (
    id        INTEGER PRIMARY KEY,
    doc_id    TEXT NOT NULL REFERENCES documents(doc_id),
    ord       INTEGER NOT NULL,
    page      INTEGER,
    text      TEXT NOT NULL,
    embedding BLOB NOT NULL
);
CREATE INDEX IF NOT EXISTS chunks_doc ON chunks(doc_id);
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    text, content='chunks', content_rowid='id', tokenize='porter unicode61');
CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
    INSERT INTO chunks_fts(rowid, text) VALUES (new.id, new.text);
END;
CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, text) VALUES ('delete', old.id, old.text);
END;
"""

# (db path, model) -> (version, chunk ids, normalized embedding matrix)
_matrix_cache: dict[tuple[str, str], tuple[int, np.ndarray, np.ndarray]] = {}
_cache_lock = threading.Lock()


class KnowledgeBase:
    def __init__(self, data_dir: Path, embedder: OllamaEmbedder):
        root = data_dir / "kb"
        root.mkdir(parents=True, exist_ok=True)
        self.path = root / "kb.sqlite"
        self.embedder = embedder
        with self._db() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.executescript(_SCHEMA)

    @contextmanager
    def _db(self):
        """One connection, one transaction: committed on success, rolled back on error, always closed."""
        db = sqlite3.connect(self.path, timeout=30)
        db.row_factory = sqlite3.Row
        try:
            with db:
                yield db
        finally:
            db.close()

    def add(self, file_name: str, kind: str, sha256: str, size: int, pages: int | None, text: str) -> dict:
        with self._db() as db:
            existing = db.execute("SELECT doc_id, file_name FROM documents WHERE sha256 = ?", (sha256,)).fetchone()
        if existing:
            raise _duplicate(existing)
        chunks = chunk_text(text, kind)
        if not chunks:
            raise ServiceError("NO_TEXT", "The document contains no extractable text.",
                               hint="Scanned PDFs need OCR, which is not enabled.")
        vectors = self.embedder.embed_documents([c.text for c in chunks])
        doc = {
            "doc_id": "kb_" + secrets.token_hex(6),
            "file_name": file_name,
            "kind": kind,
            "sha256": sha256,
            "size_bytes": size,
            "pages": pages,
            "chunks": len(chunks),
            "embed_model": self.embedder.model,
            "created_at": datetime.now(timezone.utc).isoformat(timespec="microseconds"),
        }
        try:
            with self._db() as db:
                db.execute("INSERT INTO documents VALUES (:doc_id, :file_name, :kind, :sha256, :size_bytes,"
                           " :pages, :chunks, :embed_model, :created_at)", doc)
                db.executemany(
                    "INSERT INTO chunks (doc_id, ord, page, text, embedding) VALUES (?, ?, ?, ?, ?)",
                    [(doc["doc_id"], i, c.page, c.text, v.tobytes()) for i, (c, v) in enumerate(zip(chunks, vectors))],
                )
                db.execute("UPDATE meta SET value = value + 1 WHERE key = 'version'")
        except sqlite3.IntegrityError:  # the same file was added concurrently
            with self._db() as db:
                existing = db.execute("SELECT doc_id, file_name FROM documents WHERE sha256 = ?", (sha256,)).fetchone()
            if existing:
                raise _duplicate(existing)
            raise
        return doc

    def documents(self) -> list[dict]:
        with self._db() as db:
            rows = db.execute("SELECT * FROM documents ORDER BY created_at, rowid").fetchall()
        return [dict(r) for r in rows]

    def delete(self, doc_id: str) -> bool:
        if not DOC_RE.match(doc_id or ""):
            raise ServiceError("BAD_DOC_ID", "Invalid doc_id.")
        with self._db() as db:
            db.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))
            deleted = db.execute("DELETE FROM documents WHERE doc_id = ?", (doc_id,)).rowcount
            if deleted:
                db.execute("UPDATE meta SET value = value + 1 WHERE key = 'version'")
        return bool(deleted)

    def search(self, query: str, top_k: int) -> dict:
        model = self.embedder.model
        warnings = []
        with self._db() as db:
            stale = db.execute("SELECT COUNT(*) FROM documents WHERE embed_model != ?", (model,)).fetchone()[0]
        if stale:
            warnings.append(f"{stale} document(s) were embedded with a different model and were not searched; "
                            "delete and re-add them.")

        ids, matrix = self._matrix(model)
        if not len(ids):
            return {"query": query, "embed_model": model, "results": [], "warnings": warnings}

        sims = matrix @ self.embedder.embed_query(query)
        n = min(CANDIDATES, len(ids))
        best = np.argpartition(-sims, n - 1)[:n]
        best = best[np.argsort(-sims[best])]
        similarity = {int(cid): float(sim) for cid, sim in zip(ids, sims)}

        fused: dict[int, float] = {}
        for rank, cid in enumerate((int(ids[i]) for i in best), start=1):
            fused[cid] = 1 / (RRF_K + rank)
        for rank, cid in enumerate(self._keyword(query, model), start=1):
            fused[cid] = fused.get(cid, 0) + 1 / (RRF_K + rank)
        top = sorted(fused, key=fused.get, reverse=True)[:top_k]

        with self._db() as db:
            marks = ",".join("?" * len(top))
            rows = {r["id"]: r for r in db.execute(
                f"SELECT c.id, c.doc_id, c.ord, c.page, c.text, d.file_name FROM chunks c"
                f" JOIN documents d USING (doc_id) WHERE c.id IN ({marks})", top)}
        results = [
            {
                "doc_id": rows[cid]["doc_id"],
                "file_name": rows[cid]["file_name"],
                "page": rows[cid]["page"],
                "chunk_index": rows[cid]["ord"],
                "text": rows[cid]["text"],
                # None only for a chunk added after the embedding matrix was loaded.
                "similarity": round(similarity[cid], 4) if cid in similarity else None,
                "score": round(fused[cid], 6),
            }
            for cid in top if cid in rows  # a document deleted mid-search drops out
        ]
        return {"query": query, "embed_model": model, "results": results, "warnings": warnings}

    def _keyword(self, query: str, model: str) -> list[int]:
        terms = re.findall(r"\w+", query.lower())
        if not terms:
            return []
        match = " OR ".join(f'"{t}"' for t in dict.fromkeys(terms))
        with self._db() as db:
            rows = db.execute(
                "SELECT c.id FROM chunks_fts JOIN chunks c ON c.id = chunks_fts.rowid"
                " JOIN documents d USING (doc_id)"
                " WHERE chunks_fts MATCH ? AND d.embed_model = ? ORDER BY chunks_fts.rank LIMIT ?",
                (match, model, CANDIDATES)).fetchall()
        return [r[0] for r in rows]

    def _matrix(self, model: str) -> tuple[np.ndarray, np.ndarray]:
        key = (str(self.path), model)
        with _cache_lock:
            with self._db() as db:
                version = db.execute("SELECT value FROM meta WHERE key = 'version'").fetchone()[0]
                cached = _matrix_cache.get(key)
                if cached and cached[0] == version:
                    return cached[1], cached[2]
                rows = db.execute(
                    "SELECT c.id, c.embedding FROM chunks c JOIN documents d USING (doc_id)"
                    " WHERE d.embed_model = ? ORDER BY c.id", (model,)).fetchall()
            ids = np.array([r[0] for r in rows], dtype=np.int64)
            matrix = (np.vstack([np.frombuffer(r[1], dtype=np.float32) for r in rows])
                      if rows else np.empty((0, 0), dtype=np.float32))
            _matrix_cache[key] = (version, ids, matrix)
            return ids, matrix


def _duplicate(row: sqlite3.Row) -> ServiceError:
    return ServiceError("DUPLICATE_DOCUMENT",
                        f"This file is already in the knowledge base as {row['doc_id']} ({row['file_name']}).",
                        status=409)
