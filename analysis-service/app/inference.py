"""Column type inference.

Conservative by design: a column is converted only when at least 95% of its
non-empty values agree, and every lossy step is recorded in the column notes
so the final answer can say what was assumed.
"""

import os
import re

import pandas as pd

THRESHOLD = 0.95
BOOL_TRUE = {"true", "yes", "y"}
BOOL_FALSE = {"false", "no", "n"}
# Order used when every d/m/y date is ambiguous (e.g. 03/04/2025).
DAYFIRST_DEFAULT = os.environ.get("AMBIGUOUS_DATE_ORDER", "dayfirst").lower() == "dayfirst"

_STRIP = re.compile(r"[$€£₹¥%\s]")
_US = re.compile(r"^[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?$|^[-+]?\.\d+$")
_EU = re.compile(r"^[-+]?(?:\d{1,3}(?:\.\d{3})+|\d+)(?:,\d+)?$|^[-+]?,\d+$")
_PLAIN_INT = re.compile(r"^[-+]?\d+$")
_LEADING_ZERO = re.compile(r"^0\d+$")
_DATE_LIKE = re.compile(
    r"\d{1,4}[-/.]\d{1,2}[-/.]\d{1,4}"
    r"|\d{1,2}\s+[A-Za-z]{3,9}\.?,?\s+\d{2,4}"
    r"|[A-Za-z]{3,9}\.?\s+\d{1,2},?\s+\d{2,4}"
)
_DMY = r"^(\d{1,2})[-/.](\d{1,2})[-/.]\d{2,4}"
_ID_NAME = re.compile(r"(?:^|[\s_-])(?:id|code|no|num|number|zip|pin|phone|sku)$", re.I)


def infer_table(data: pd.DataFrame, decimal_hint: str, na_tokens: dict) -> tuple[pd.DataFrame, list[dict]]:
    series, columns = {}, []
    for name in data.columns:
        s, info = infer_column(data[name], name, decimal_hint)
        if na_tokens.get(name):
            found = ", ".join(f"'{k}' x{v}" for k, v in na_tokens[name].items())
            info["notes"].append(f"Placeholder values treated as missing: {found}.")
        series[name] = s
        columns.append(info)
    return pd.DataFrame(series), columns


def infer_column(s: pd.Series, name: str, decimal_hint: str = ".") -> tuple[pd.Series, dict]:
    s = s.astype(object).where(s.notna(), None)
    nonnull = s.dropna().astype(str)
    info = {"name": name, "type": "text", "notes": []}

    if nonnull.empty:
        info.update(type="empty", notes=["Column has no values."])
        return s, info

    lower = nonnull.str.lower()
    if lower.isin(BOOL_TRUE | BOOL_FALSE).all():
        info["type"] = "boolean"
        return s.map(lambda v: None if v is None else str(v).lower() in BOOL_TRUE).astype("boolean"), info

    numeric = _as_number(nonnull, decimal_hint, info["notes"])
    if numeric is not None:
        full = numeric.reindex(s.index)
        values = full.dropna()
        if full.dtype.kind == "f" or str(full.dtype) == "Float64":
            if len(values) and (values % 1 == 0).all() and values.abs().max() < 2**53:
                full = full.astype("Int64")
        is_int = str(full.dtype) == "Int64"
        info["type"] = "integer" if is_int else "decimal"
        if is_int and values.nunique() == len(values) and _ID_NAME.search(name):
            info["type"] = "identifier"
            info["notes"].append("Unique whole numbers in an ID-like column; not summed or averaged.")
        return full, info

    parsed = _as_datetime(nonnull, info["notes"])
    if parsed is not None:
        info["type"] = "datetime"
        return parsed.reindex(s.index), info

    n, unique = len(nonnull), nonnull.nunique()
    if unique <= 50 and unique < n and nonnull.str.len().mean() <= 40:
        info["type"] = "category"
    elif unique == n and (_ID_NAME.search(name) or nonnull.str.contains(r"\d").mean() >= 0.9):
        info["type"] = "identifier"
    return s.astype("str"), info


def _as_number(values: pd.Series, decimal_hint: str, notes: list[str]) -> pd.Series | None:
    t = values.str.strip()
    negative = t.str.match(r"^\(.*\)$").fillna(False).astype(bool)
    t = t.str.replace(r"^\((.*)\)$", r"\1", regex=True)
    percent = t.str.endswith("%").fillna(False).astype(bool)
    t = t.str.replace(_STRIP, "", regex=True)

    if t.str.match(_LEADING_ZERO).any() and t.str.match(_PLAIN_INT).mean() >= THRESHOLD:
        notes.append("Kept as text because some values have leading zeros (codes, not quantities).")
        return None
    if t.str.match(_PLAIN_INT).all() and t.str.lstrip("+-").str.len().max() >= 16:
        notes.append("Kept as text: 16+ digit numbers are treated as codes to avoid precision loss.")
        return None

    us = t.str.match(_US).fillna(False).astype(bool)
    eu = t.str.match(_EU).fillna(False).astype(bool)
    if max(us.mean(), eu.mean()) < THRESHOLD:
        return None

    if us.mean() >= THRESHOLD and eu.mean() >= THRESHOLD:
        decimal = decimal_hint
        if t.str.contains(r"[.,]").any():
            sep = "comma" if decimal == "," else "dot"
            notes.append(f"Separators are ambiguous (e.g. 1,234 vs 1.234); assumed {sep} is the decimal mark.")
    else:
        decimal = "." if us.mean() >= eu.mean() else ","
        if decimal == ",":
            notes.append("Read with comma as decimal mark (European format).")

    ok = us if decimal == "." else eu
    clean = t.where(ok)
    if decimal == ".":
        clean = clean.str.replace(",", "", regex=False)
    else:
        clean = clean.str.replace(".", "", regex=False).str.replace(",", ".", regex=False)
    numbers = pd.to_numeric(clean, errors="coerce", dtype_backend="numpy_nullable")
    numbers = numbers.where(~negative, -numbers)

    failed = values[numbers.isna()]
    if len(failed):
        examples = ", ".join(f"'{v}'" for v in failed.unique()[:3])
        notes.append(f"{len(failed)} value(s) are not numbers and were treated as missing, e.g. {examples}.")
    if percent.any():
        notes.append("Percent signs removed; values are in percent (15% is stored as 15).")
    if negative.any():
        notes.append("Values in parentheses read as negative (accounting format).")
    return numbers


def _as_datetime(values: pd.Series, notes: list[str]) -> pd.Series | None:
    if values.str.contains(_DATE_LIKE).mean() < THRESHOLD:
        return None

    parts = values.str.extract(_DMY).dropna().astype(int)
    first_big, second_big = (parts[0] > 12).any(), (parts[1] > 12).any()
    if first_big and second_big:
        notes.append("Kept as text: dates mix DD/MM and MM/DD order.")
        return None
    if first_big:
        dayfirst = True
    elif second_big:
        dayfirst = False
    else:
        dayfirst = DAYFIRST_DEFAULT if len(parts) else False
        if len(parts):
            order = "DD/MM/YYYY" if dayfirst else "MM/DD/YYYY"
            notes.append(f"Day/month order is ambiguous; assumed {order}.")

    try:
        parsed = pd.to_datetime(values, errors="coerce", format="mixed", dayfirst=dayfirst)
    except (ValueError, TypeError, OverflowError):
        return None
    if not pd.api.types.is_datetime64_any_dtype(parsed) or parsed.notna().mean() < THRESHOLD:
        return None
    if getattr(parsed.dt, "tz", None) is not None:
        parsed = parsed.dt.tz_convert("UTC").dt.tz_localize(None)
        notes.append("Time zones converted to UTC.")

    failed = values[parsed.isna()]
    if len(failed):
        examples = ", ".join(f"'{v}'" for v in failed.unique()[:3])
        notes.append(f"{len(failed)} value(s) are not dates and were treated as missing, e.g. {examples}.")
    return parsed
