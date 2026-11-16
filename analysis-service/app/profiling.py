"""Build the JSON profile for a dataset, and a compact text form for the LLM."""

import math
from datetime import datetime, timezone

import pandas as pd

from .parsers import ParsedFile, ParsedTable

DELIMITER_NAMES = {",": "comma", ";": "semicolon", "\t": "tab", "|": "pipe"}


def build_profile(dataset_id: str, file_name: str, kind: str, size: int, parsed: ParsedFile) -> dict:
    return {
        "dataset_id": dataset_id,
        "file_name": file_name,
        "kind": kind,
        "size_bytes": size,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="microseconds"),
        "encoding": parsed.encoding,
        "delimiter": DELIMITER_NAMES.get(parsed.delimiter) if parsed.delimiter else None,
        "tables": [profile_table(t, f"t{i}") for i, t in enumerate(parsed.tables, start=1)],
        "text": profile_text(parsed.text, parsed.pages) if parsed.text is not None else None,
        "warnings": parsed.warnings,
    }


def profile_table(table: ParsedTable, table_id: str) -> dict:
    df = table.df
    rows = len(df)
    missing_cells = int(df.isna().sum().sum())
    return {
        "table_id": table_id,
        "name": table.name,
        "rows": rows,
        "columns": len(df.columns),
        "header_row": table.header_row,
        "duplicate_rows": int(df.duplicated().sum()),
        "missing_cells": missing_cells,
        "missing_pct": _pct(missing_cells, rows * len(df.columns)),
        "column_profiles": [_profile_column(df[c["name"]], c, rows) for c in table.columns],
        "notes": table.notes,
    }


def profile_text(text: str, pages: int | None) -> dict:
    return {
        "chars": len(text),
        "words": len(text.split()),
        "lines": text.count("\n") + 1 if text else 0,
        "pages": pages,
        "est_tokens": math.ceil(len(text) / 4),
        "preview": text[:400],
    }


def _profile_column(s: pd.Series, info: dict, rows: int) -> dict:
    missing = int(s.isna().sum())
    values = s.dropna()
    col = {
        "name": info["name"],
        "type": info["type"],
        "missing": missing,
        "missing_pct": _pct(missing, rows),
        "unique": int(values.nunique()),
        "samples": [_display(v) for v in values.drop_duplicates().head(5)],
        "notes": info["notes"],
    }
    kind = info["type"]
    if kind in ("integer", "decimal") and len(values):
        v = values.astype("float64")
        col["stats"] = {
            "min": _num(v.min()), "max": _num(v.max()), "mean": _num(v.mean()),
            "median": _num(v.median()), "std": _num(v.std()) if len(v) > 1 else None,
            "sum": _num(v.sum()),
        }
    elif kind == "datetime" and len(values):
        col["stats"] = {
            "min": _display(values.min()), "max": _display(values.max()),
            "span_days": int((values.max() - values.min()).days),
        }
    elif kind in ("category", "boolean"):
        col["top_values"] = [
            {"value": _display(k), "count": int(n)} for k, n in values.value_counts().head(10).items()
        ]
    elif kind == "text" and len(values):
        lengths = values.astype(str).str.len()
        col["stats"] = {"avg_length": _num(lengths.mean()), "max_length": int(lengths.max())}
    return col


def compact_summary(profiles: list[dict]) -> str:
    """A few lines per table: small enough for a 3-4B local model's context."""
    lines = []
    for p in profiles:
        lines.append(f"Dataset {p['dataset_id']} \"{p['file_name']}\" ({p['kind']})")
        for t in p["tables"]:
            lines.append(
                f"  Table {t['table_id']} \"{t['name']}\": {t['rows']} rows x {t['columns']} cols, "
                f"{t['duplicate_rows']} duplicate rows, {t['missing_pct']}% cells missing"
            )
            for c in t["column_profiles"]:
                lines.append(f"    - {c['name']} [{c['type']}]{_column_hint(c)}")
        if p.get("text"):
            tx = p["text"]
            pages = f", {tx['pages']} pages" if tx.get("pages") else ""
            lines.append(f"  Text: {tx['words']} words{pages}")
    return "\n".join(lines)


def _column_hint(c: dict) -> str:
    parts = []
    stats = c.get("stats") or {}
    if c["type"] in ("integer", "decimal") and stats:
        parts.append(f"range {stats['min']} to {stats['max']}")
    elif c["type"] == "datetime" and stats:
        parts.append(f"{stats['min']} to {stats['max']}")
    elif c["type"] in ("category", "boolean") and c.get("top_values"):
        shown = [str(v["value"]) for v in c["top_values"][:6]]
        more = f" +{c['unique'] - len(shown)} more" if c["unique"] > len(shown) else ""
        parts.append("values: " + ", ".join(shown) + more)
    if c["missing"]:
        parts.append(f"{c['missing_pct']}% missing")
    return (" " + "; ".join(parts)) if parts else ""


def _display(v):
    if isinstance(v, pd.Timestamp):
        return v.date().isoformat() if v == v.normalize() else v.isoformat(sep=" ")
    if hasattr(v, "item"):
        v = v.item()
    if isinstance(v, float):
        return _num(v)
    if isinstance(v, str) and len(v) > 60:
        return v[:57] + "..."
    return v


def _num(x) -> float | int | None:
    if x is None or pd.isna(x):
        return None
    x = float(x)
    if x.is_integer() and abs(x) < 2**53:
        return int(x)
    return round(x, 6)


def _pct(part: int, whole: int) -> float:
    return round(100 * part / whole, 2) if whole else 0.0

