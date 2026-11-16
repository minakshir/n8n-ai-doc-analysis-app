import os
from datetime import datetime
from pathlib import Path

import pytest

os.environ["SERVICE_TOKEN"] = "test-token-" + "x" * 32
os.environ["MAX_UPLOAD_MB"] = "1"
os.environ["AMBIGUOUS_DATE_ORDER"] = "dayfirst"


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    from app.config import get_settings

    get_settings.cache_clear()
    from fastapi.testclient import TestClient

    from app.main import app

    with TestClient(app) as c:
        c.headers["X-Service-Token"] = os.environ["SERVICE_TOKEN"]
        yield c
    get_settings.cache_clear()


@pytest.fixture
def files(tmp_path) -> Path:
    d = tmp_path / "fixtures"
    d.mkdir()

    # European CSV: BOM, semicolons, comma decimals, €, NA tokens, one duplicate row.
    (d / "sales_eu.csv").write_bytes(
        "﻿Region;Date;Revenue;Units;Status\n"
        "North;03/04/2025;1.234,50;10;yes\n"
        "South;15/04/2025;€ 980,00;5;no\n"
        "North;03/04/2025;1.234,50;10;yes\n"
        "East;N/A;450,25;n/a;yes\n"
        "West;20/04/2025;-;7;no\n".encode("utf-8")
    )

    # Workbook: title row + blank row above the header, leading-zero zips, an empty sheet.
    from openpyxl import Workbook

    wb = Workbook()
    ws = wb.active
    ws.title = "Q3 Sales"
    ws.append(["Quarterly Sales Report"])
    ws.append([])
    ws.append(["Order ID", "Customer", "Amount", "Order Date", "Zip"])
    ws.append([1001, "Acme", 1500.5, datetime(2025, 7, 1), "01234"])
    ws.append([1002, "Globex", 2300, datetime(2025, 8, 15), "02139"])
    ws.append([1003, "Acme", None, datetime(2025, 9, 30), "10001"])
    ws.append([1004, "Initech", 800, datetime(2025, 9, 30), "60601"])
    wb.create_sheet("Empty")
    wb.save(d / "report.xlsx")

    (d / "notes.txt").write_text(
        "Quarterly review.\n\nRevenue grew in the North region while the South declined.\n"
        "The team recommends a pricing review before Q4.\n",
        encoding="utf-8",
    )
    (d / "table.txt").write_text("name\tscore\nana\t90\nben\t75\ncara\t88\n", encoding="utf-8")

    from reportlab.lib.pagesizes import A4
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Table, TableStyle
    from reportlab.lib.styles import getSampleStyleSheet

    doc = SimpleDocTemplate(str(d / "report.pdf"), pagesize=A4)
    grid = Table([["Item", "Qty"], ["Apples", "10"], ["Pears", "4"]])
    grid.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 1, "black")]))
    doc.build([Paragraph("Annual summary of fruit stock levels.", getSampleStyleSheet()["Normal"]), grid])

    (d / "fake.pdf").write_text("this is not a pdf")
    (d / "old.xls").write_bytes(b"\xd0\xcf\x11\xe0" + b"\x00" * 100)
    (d / "big.csv").write_text("a,b\n" + "1,2\n" * 300_000)
    return d
