import pandas as pd

from app.inference import infer_column
from app.parsers import detect_header, sniff_delimiter


def col(values, name="value", hint="."):
    return infer_column(pd.Series(values, dtype=object), name, hint)


def test_us_thousands_and_decimals():
    s, info = col(["1,234.5", "2,000", "15.25"])
    assert info["type"] == "decimal"
    assert s.tolist() == [1234.5, 2000.0, 15.25]


def test_ambiguous_separator_follows_hint():
    s, info = col(["1,234", "5,678"], hint=".")
    assert s.tolist() == [1234, 5678]
    s, info = col(["1,234", "5,678"], hint=",")
    assert s.tolist() == [1.234, 5.678]
    assert any("ambiguous" in n for n in info["notes"])


def test_accounting_negative_and_percent():
    s, info = col(["(500)", "250", "$1,000"])
    assert s.tolist() == [-500, 250, 1000]
    assert info["type"] == "integer"
    s, info = col(["15%", "7.5%"])
    assert s.tolist() == [15, 7.5]
    assert any("percent" in n for n in info["notes"])


def test_codes_stay_text():
    _, info = col(["01234", "02139", "10001"], name="Zip")
    assert info["type"] == "identifier"
    _, info = col(["4111111111111111", "5500000000000004"], name="card")
    assert info["type"] != "integer"


def test_one_bad_value_in_twenty_is_reported_not_hidden():
    s, info = col([str(i) for i in range(19)] + ["tbd"])
    assert info["type"] == "integer"
    assert s.isna().sum() == 1
    assert any("'tbd'" in n for n in info["notes"])


def test_too_many_bad_values_keeps_text():
    _, info = col(["1", "2", "x", "y"])
    assert info["type"] in ("text", "category")


def test_dates_month_first_when_unambiguous():
    s, info = col(["12/25/2025", "01/02/2025"])
    assert info["type"] == "datetime"
    assert s.iloc[0] == pd.Timestamp(2025, 12, 25)
    assert s.iloc[1] == pd.Timestamp(2025, 1, 2)


def test_dates_ambiguous_use_default_and_say_so():
    s, info = col(["03/04/2025", "05/06/2025"])
    assert s.iloc[0] == pd.Timestamp(2025, 4, 3)  # AMBIGUOUS_DATE_ORDER=dayfirst in conftest
    assert any("ambiguous" in n for n in info["notes"])


def test_iso_dates_are_not_flipped():
    s, _ = col(["2025-01-05", "2025-02-17"])
    assert s.iloc[0] == pd.Timestamp(2025, 1, 5)


def test_boolean_and_identifier():
    s, info = col(["Yes", "no", "YES"])
    assert info["type"] == "boolean"
    assert s.tolist() == [True, False, True]
    _, info = col(["1", "2", "3"], name="customer_id")
    assert info["type"] == "identifier"


def test_sniff_delimiter():
    assert sniff_delimiter("a;b;c\n1;2;3\n4;5;6\n") == (";", True)
    assert sniff_delimiter("a,b\n1,2\n")[0] == ","
    assert sniff_delimiter("Just a sentence, with a comma.\nAnother line here.\n")[1] is False


def test_detect_header_skips_title_rows():
    cells = pd.DataFrame([["Report", None, None], ["A", "B", "C"], ["1", "2", "3"]])
    assert detect_header(cells) == 1
    numeric_first = pd.DataFrame([["1", "2"], ["3", "4"]])
    assert detect_header(numeric_first) is None
