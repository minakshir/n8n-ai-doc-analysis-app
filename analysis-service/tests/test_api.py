import pandas as pd

SESSION = "sess-123"


def upload(client, path, session=SESSION):
    with path.open("rb") as f:
        return client.post("/ingest", data={"session_id": session}, files={"file": (path.name, f)})


def by_name(table, name):
    return next(c for c in table["column_profiles"] if c["name"] == name)


def test_health_needs_no_token(client):
    r = client.get("/health", headers={"X-Service-Token": ""})
    assert r.json()["status"] == "ok"


def test_token_required(client, files):
    with (files / "notes.txt").open("rb") as f:
        r = client.post("/ingest", data={"session_id": SESSION}, files={"file": ("notes.txt", f)},
                        headers={"X-Service-Token": "wrong"})
    assert r.status_code == 401
    assert r.json()["error"]["code"] == "UNAUTHORIZED"


def test_european_csv(client, files):
    r = upload(client, files / "sales_eu.csv")
    assert r.status_code == 200, r.text
    p = r.json()
    assert p["kind"] == "csv" and p["delimiter"] == "semicolon"
    t = p["tables"][0]
    assert (t["rows"], t["columns"], t["duplicate_rows"], t["missing_cells"]) == (5, 5, 1, 3)

    revenue = by_name(t, "Revenue")
    assert revenue["type"] == "decimal"
    assert revenue["stats"]["sum"] == 3899.25
    assert revenue["missing"] == 1

    units = by_name(t, "Units")
    assert units["type"] == "integer" and units["stats"]["sum"] == 32

    date = by_name(t, "Date")
    assert date["type"] == "datetime"
    assert (date["stats"]["min"], date["stats"]["max"]) == ("2025-04-03", "2025-04-20")

    assert by_name(t, "Status")["type"] == "boolean"
    assert by_name(t, "Region")["type"] == "category"


def test_excel_title_rows_codes_and_empty_sheet(client, files):
    p = upload(client, files / "report.xlsx").json()
    assert len(p["tables"]) == 1
    assert any("Empty" in w for w in p["warnings"])
    t = p["tables"][0]
    assert t["name"] == "Q3 Sales"
    assert t["header_row"] == 3
    assert t["rows"] == 4

    assert by_name(t, "Order ID")["type"] == "identifier"
    assert by_name(t, "Customer")["type"] == "category"
    amount = by_name(t, "Amount")
    assert amount["type"] == "decimal" and amount["stats"]["sum"] == 4600.5 and amount["missing"] == 1
    assert by_name(t, "Order Date")["stats"]["max"] == "2025-09-30"
    zip_col = by_name(t, "Zip")
    assert zip_col["type"] == "identifier" and "01234" in zip_col["samples"]


def test_typed_table_round_trips_through_parquet(client, files):
    p = upload(client, files / "report.xlsx").json()
    from app.config import get_settings
    from app.storage import Storage

    df = Storage(get_settings().data_dir).load_table(SESSION, p["dataset_id"], "t1")
    assert str(df["Amount"].dtype) in ("Float64", "float64")
    assert pd.api.types.is_datetime64_any_dtype(df["Order Date"])
    assert df["Zip"].tolist()[0] == "01234"


def test_plain_text_and_tabular_text(client, files):
    p = upload(client, files / "notes.txt").json()
    assert p["kind"] == "txt" and p["tables"] == []
    assert p["text"]["words"] > 10

    p = upload(client, files / "table.txt").json()
    assert p["tables"][0]["rows"] == 3
    assert by_name(p["tables"][0], "score")["stats"]["sum"] == 253


def test_pdf_text_and_table(client, files):
    p = upload(client, files / "report.pdf").json()
    assert p["kind"] == "pdf" and p["text"]["pages"] == 1
    assert "Annual summary" in p["text"]["preview"]
    qty = by_name(p["tables"][0], "Qty")
    assert qty["stats"]["sum"] == 14

    r = client.get(f"/sessions/{SESSION}/datasets/{p['dataset_id']}/text")
    assert "Annual summary" in r.json()["text"]


def test_rejections(client, files):
    cases = {
        "fake.pdf": (400, "FILE_TYPE_MISMATCH"),
        "old.xls": (400, "UNSUPPORTED_FILE"),
        "big.csv": (413, "FILE_TOO_LARGE"),
    }
    for name, (status, code) in cases.items():
        r = upload(client, files / name)
        assert (r.status_code, r.json()["error"]["code"]) == (status, code), name

    r = upload(client, files / "notes.txt", session="../etc")
    assert r.json()["error"]["code"] == "BAD_SESSION_ID"

    r = client.post("/ingest", data={"session_id": SESSION})
    assert r.status_code == 422 and r.json()["error"]["code"] == "BAD_REQUEST"


def test_profiles_compact_and_delete(client, files):
    upload(client, files / "sales_eu.csv")
    upload(client, files / "notes.txt")

    full = client.get(f"/sessions/{SESSION}/profiles").json()
    assert [d["file_name"] for d in full["datasets"]] == ["sales_eu.csv", "notes.txt"]

    compact = client.get(f"/sessions/{SESSION}/profiles", params={"format": "compact"}).json()
    assert "Revenue [decimal]" in compact["summary"]
    assert "values: North" in compact["summary"]

    assert client.delete(f"/sessions/{SESSION}").json() == {"deleted": True}
    assert client.get(f"/sessions/{SESSION}/profiles").json()["datasets"] == []


def test_unknown_dataset_is_404(client):
    r = client.get(f"/sessions/{SESSION}/datasets/ds_000000000000/text")
    assert r.status_code == 404
