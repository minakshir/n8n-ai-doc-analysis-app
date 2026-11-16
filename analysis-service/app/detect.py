"""Decide what kind of file was uploaded, from its extension AND its bytes."""

import zipfile
from pathlib import Path

import charset_normalizer

from .errors import ServiceError

EXTENSIONS = {
    ".csv": "csv",
    ".tsv": "csv",
    ".txt": "txt",
    ".xlsx": "xlsx",
    ".xlsm": "xlsx",
    ".pdf": "pdf",
}

# Limits for a zip-based .xlsx before openpyxl ever opens it (zip-bomb guard).
MAX_XLSX_UNCOMPRESSED = 1024 * 1024 * 1024
MAX_XLSX_RATIO = 200
MAX_XLSX_MEMBERS = 10_000


def detect_kind(path: Path, filename: str) -> str:
    ext = Path(filename).suffix.lower()
    if ext == ".xls":
        raise ServiceError(
            "UNSUPPORTED_FILE",
            "Legacy .xls files are not supported.",
            hint="Open the file in Excel or LibreOffice and save it as .xlsx.",
        )
    kind = EXTENSIONS.get(ext)
    if kind is None:
        raise ServiceError(
            "UNSUPPORTED_FILE",
            f"Unsupported file type '{ext or 'none'}'.",
            hint="Supported: .csv, .tsv, .txt, .xlsx, .xlsm, .pdf",
        )

    with path.open("rb") as f:
        head = f.read(65536)

    if kind == "pdf":
        if not head.startswith(b"%PDF-"):
            raise _mismatch(filename, "PDF")
    elif kind == "xlsx":
        if not head.startswith(b"PK\x03\x04"):
            raise _mismatch(filename, "Excel workbook")
        _check_xlsx_archive(path, filename)
    else:
        if b"\x00" in head or head.startswith((b"PK\x03\x04", b"%PDF-")):
            raise _mismatch(filename, "text file")
    return kind


def decode_text(path: Path) -> tuple[str, str]:
    raw = path.read_bytes()
    try:
        return raw.decode("utf-8-sig"), "utf-8"
    except UnicodeDecodeError:
        pass
    best = charset_normalizer.from_bytes(raw).best()
    if best is None:
        raise ServiceError(
            "UNREADABLE_TEXT",
            "Could not detect the text encoding of this file.",
            hint="Re-save the file as UTF-8.",
        )
    return str(best), best.encoding


def _check_xlsx_archive(path: Path, filename: str) -> None:
    try:
        with zipfile.ZipFile(path) as zf:
            infos = zf.infolist()
    except zipfile.BadZipFile:
        raise _mismatch(filename, "Excel workbook")

    names = {i.filename for i in infos}
    if "xl/workbook.xml" not in names:
        raise _mismatch(filename, "Excel workbook")
    if len(infos) > MAX_XLSX_MEMBERS:
        raise ServiceError("UNSAFE_FILE", "The workbook contains too many internal parts.")

    uncompressed = sum(i.file_size for i in infos)
    compressed = max(1, sum(i.compress_size for i in infos))
    if uncompressed > MAX_XLSX_UNCOMPRESSED or uncompressed / compressed > MAX_XLSX_RATIO:
        raise ServiceError(
            "UNSAFE_FILE",
            "The workbook expands to an unsafe size.",
            hint="Split the workbook or export the needed sheet as CSV.",
        )


def _mismatch(filename: str, expected: str) -> ServiceError:
    return ServiceError(
        "FILE_TYPE_MISMATCH",
        f"'{filename}' does not contain a valid {expected}.",
        hint="The file extension does not match the file contents.",
    )
