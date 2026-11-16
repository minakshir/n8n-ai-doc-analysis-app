"""Turn an uploaded file into raw string tables and/or plain text.

Every tabular source (CSV, Excel sheet, TXT table, PDF table) goes through the
same path: a DataFrame of stripped strings / None -> header detection ->
type inference (inference.py). That keeps behaviour identical across formats.
"""

import csv
import io
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, date
from pathlib import Path

import pandas as pd
import pdfplumber

from .detect import decode_text
from .errors import ServiceError
from .inference import infer_table

NA_TOKENS = {"na", "n/a", "nan", "null", "none", "nil", "-", "--", "#n/a", "missing"}
DELIMITERS = [",", ";", "\t", "|"]
_VALUE_LIKE = re.compile(
    r"^[-+(]?[$€£₹¥]?\s*[\d.,]+\s*%?\)?$"  # numbers, money, percents
    r"|^\d{1,4}[-/.]\d{1,2}[-/.]\d{1,4}"  # dates
)


@dataclass
class ParsedTable:
    name: str
    df: pd.DataFrame
    header_row: int | None  # 1-based row in the source, None when generated
    columns: list[dict]  # per-column inference details (type, notes)
    notes: list[str] = field(default_factory=list)


@dataclass
class ParsedFile:
    kind: str
    tables: list[ParsedTable] = field(default_factory=list)
    text: str | None = None
    pages: int | None = None
    encoding: str | None = None
    delimiter: str | None = None
    warnings: list[str] = field(default_factory=list)


def parse_file(path: Path, kind: str, filename: str, max_pdf_pages: int) -> ParsedFile:
    if kind == "csv":
        return _parse_csv(path, filename)
    if kind == "txt":
        return _parse_txt(path)
    if kind == "xlsx":
        return _parse_xlsx(path)
    if kind == "pdf":
        return _parse_pdf(path, max_pdf_pages)
    raise ValueError(f"unknown kind {kind}")


# ---------------------------------------------------------------- CSV / TXT


def _parse_csv(path: Path, filename: str) -> ParsedFile:
    text, encoding = decode_text(path)
    sep = "\t" if filename.lower().endswith(".tsv") else sniff_delimiter(text[:65536])[0]
    out = ParsedFile(kind="csv", encoding=encoding, delimiter=sep)
    raw, extra_rows = _read_delimited(text, sep)
    if extra_rows:
        out.warnings.append(
            f"{extra_rows} row(s) have more fields than most rows; check for unquoted delimiters."
        )
    table = build_table(raw, Path(filename).stem, decimal_hint="," if sep == ";" else ".")
    if table is None:
        raise ServiceError("EMPTY_FILE", "The file contains no data rows.")
    out.tables.append(table)
    return out


def _parse_txt(path: Path) -> ParsedFile:
    text, encoding = decode_text(path)
    out = ParsedFile(kind="txt", text=text, encoding=encoding)
    sep, consistent = sniff_delimiter(text[:65536])
    if consistent:
        raw, _ = _read_delimited(text, sep)
        table = build_table(raw, "text table", decimal_hint="," if sep == ";" else ".")
        if table is not None:
            out.delimiter = sep
            out.tables.append(table)
            out.warnings.append(
                f"The text looks like a table ({_sep_name(sep)}-separated); it was also parsed as one."
            )
    if not text.strip():
        raise ServiceError("EMPTY_FILE", "The text file is empty.")
    return out


def sniff_delimiter(sample: str) -> tuple[str, bool]:
    """Return (delimiter, looks_consistently_tabular)."""
    lines = [ln for ln in sample.splitlines() if ln.strip()][:50]
    if len(lines) > 1 and not sample.endswith(("\n", "\r")):
        lines = lines[:-1]  # last line of a sample may be cut mid-row
    best, best_score, consistent = ",", -1, False
    for d in DELIMITERS:
        counts = [len(r) for r in csv.reader(lines, delimiter=d)]
        if not counts:
            continue
        width, freq = Counter(counts).most_common(1)[0]
        if width < 2:
            continue
        score = freq * 1000 + width
        if score > best_score:
            best, best_score = d, score
            consistent = len(lines) >= 3 and freq / len(lines) >= 0.8
    return best, consistent


def _read_delimited(text: str, sep: str) -> tuple[pd.DataFrame, int]:
    widths = Counter(len(r) for r in csv.reader(io.StringIO(text), delimiter=sep) if r)
    if not widths:
        return pd.DataFrame(), 0
    common = widths.most_common(1)[0][0]
    max_width = max(widths)
    extra_rows = sum(n for w, n in widths.items() if w > common)
    raw = pd.read_csv(
        io.StringIO(text),
        sep=sep,
        header=None,
        names=range(max_width),  # ragged rows are padded instead of rejected
        dtype=str,
        keep_default_na=False,
        na_filter=False,
        skip_blank_lines=True,
        engine="c",
    )
    return raw, extra_rows


# ---------------------------------------------------------------- Excel


def _parse_xlsx(path: Path) -> ParsedFile:
    out = ParsedFile(kind="xlsx")
    try:
        sheets = pd.read_excel(path, sheet_name=None, header=None, dtype=object, engine="openpyxl")
    except Exception as e:  # corrupt or password-protected workbooks
        raise ServiceError("PARSE_FAILED", "Could not read the Excel workbook.", hint=str(e)[:200])
    for name, raw in sheets.items():
        table = build_table(raw, str(name), decimal_hint=".")
        if table is None:
            out.warnings.append(f"Sheet '{name}' is empty and was skipped.")
        else:
            out.tables.append(table)
    if not out.tables:
        raise ServiceError("EMPTY_FILE", "The workbook contains no data.")
    return out


# ---------------------------------------------------------------- PDF


def _parse_pdf(path: Path, max_pages: int) -> ParsedFile:
    out = ParsedFile(kind="pdf")
    try:
        pdf = pdfplumber.open(path)
    except Exception as e:
        if "password" in repr(e).lower() or "encrypt" in repr(e).lower():
            raise ServiceError("PDF_ENCRYPTED", "The PDF is password-protected.",
                               hint="Remove the password and upload it again.")
        raise ServiceError("PARSE_FAILED", "Could not read the PDF.", hint=str(e)[:200])

    with pdf:
        n_pages = len(pdf.pages)
        if n_pages > max_pages:
            raise ServiceError("TOO_MANY_PAGES", f"The PDF has {n_pages} pages; the limit is {max_pages}.")
        parts = []
        for i, page in enumerate(pdf.pages, start=1):
            parts.append(f"[Page {i}]\n{page.extract_text() or ''}".rstrip())
            for j, rows in enumerate(page.extract_tables(), start=1):
                rows = [r for r in rows if r and any(c not in (None, "") for c in r)]
                if len(rows) >= 2 and max(len(r) for r in rows) >= 2:
                    table = build_table(pd.DataFrame(rows), f"Page {i} table {j}", decimal_hint=".")
                    if table is not None:
                        out.tables.append(table)
            page.close()

    out.pages = n_pages
    out.text = "\n\n".join(parts)
    body_chars = len(re.sub(r"\[Page \d+\]|\s", "", out.text))
    if body_chars < 20 * n_pages and not out.tables:
        out.warnings.append(
            "Little or no text was found; this may be a scanned PDF. OCR is not enabled."
        )
    return out


# ---------------------------------------------------------------- shared table path


def build_table(raw: pd.DataFrame, name: str, decimal_hint: str) -> ParsedTable | None:
    """Normalise cells, find the header row, infer column types."""
    if raw.empty:
        return None
    cells = raw.astype(object).map(_cell_to_str)
    cells, na_counts = _blank_na_tokens(cells)
    cells = cells.dropna(how="all").dropna(axis=1, how="all")
    if cells.empty:
        return None

    source_rows = cells.index.to_list()  # positions in the original sheet/file
    source_cols = cells.columns.to_list()
    cells = cells.reset_index(drop=True)
    cells.columns = range(cells.shape[1])
    header_pos = detect_header(cells)

    if header_pos is None:
        names = [f"column_{i + 1}" for i in range(cells.shape[1])]
        data = cells
        header_row = None
    else:
        names = [cells.iat[header_pos, i] for i in range(cells.shape[1])]
        data = cells.iloc[header_pos + 1:]
        header_row = int(source_rows[header_pos]) + 1

    data = data.dropna(how="all").reset_index(drop=True)
    names = unique_names(names)
    data.columns = names
    if data.empty:
        return None

    na_by_name = {names[i]: na_counts.get(col, {}) for i, col in enumerate(source_cols)}
    df, columns = infer_table(data, decimal_hint=decimal_hint, na_tokens=na_by_name)
    notes = []
    if header_pos is not None and header_pos > 0:
        notes.append(f"Header found on row {header_row}; {header_pos} row(s) above it were skipped.")
    if header_pos is None:
        notes.append("No header row detected; columns were named column_1, column_2, ...")
    return ParsedTable(name=name, df=df, header_row=header_row, columns=columns, notes=notes)


def detect_header(cells: pd.DataFrame) -> int | None:
    """First reasonably full row that is mostly labels (not numbers/dates)."""
    window = cells.head(30)
    filled = window.notna().sum(axis=1)
    max_filled = int(filled.max())
    need = 1 if cells.shape[1] == 1 else max(2, math.ceil(0.5 * max_filled))
    for pos in range(len(window)):
        if filled.iat[pos] < need:
            continue
        values = window.iloc[pos].dropna()
        labels = sum(not _VALUE_LIKE.match(v) for v in values)
        return pos if labels / len(values) >= 0.6 else None
    return None


def unique_names(names: list) -> list[str]:
    out, seen = [], Counter()
    for i, n in enumerate(names):
        base = str(n).strip() if n is not None else ""
        base = base or f"column_{i + 1}"
        seen[base] += 1
        out.append(base if seen[base] == 1 else f"{base}_{seen[base]}")
    return out


def _cell_to_str(v) -> str | None:
    if v is None or v is pd.NaT:
        return None
    if isinstance(v, float):
        if math.isnan(v):
            return None
        if v.is_integer() and abs(v) < 1e15:
            return str(int(v))
    if isinstance(v, datetime):
        return v.isoformat(sep=" ") if (v.hour, v.minute, v.second, v.microsecond) != (0, 0, 0, 0) \
            else v.date().isoformat()
    if isinstance(v, date):
        return v.isoformat()
    s = str(v).strip()
    return s or None


def _blank_na_tokens(cells: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    counts: dict = {}
    for col in cells.columns:
        s = cells[col]
        mask = s.notna() & s.map(lambda v: isinstance(v, str) and v.lower() in NA_TOKENS)
        if mask.any():
            counts[col] = dict(Counter(s[mask]))
            cells.loc[mask, col] = None
    return cells, counts


def _sep_name(sep: str) -> str:
    return {",": "comma", ";": "semicolon", "\t": "tab", "|": "pipe"}[sep]
