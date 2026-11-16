import hashlib
import re

import numpy as np
import pytest

from app.chunking import OVERLAP, TARGET, chunk_text
from app.embeddings import OllamaEmbedder, normalize


class FakeEmbedder(OllamaEmbedder):
    """Deterministic bag-of-words vectors, so tests need no Ollama server."""

    def __init__(self, model="fake-embed"):
        super().__init__("http://unused", model)

    def _embed(self, inputs):
        out = np.zeros((len(inputs), 256), dtype=np.float32)
        for row, text in enumerate(inputs):
            for word in re.findall(r"\w+", text.lower()):
                out[row, int(hashlib.md5(word.encode()).hexdigest(), 16) % 256] += 1
        return normalize(out)


@pytest.fixture
def kb_client(client):
    from app.main import app, get_embedder

    app.dependency_overrides[get_embedder] = lambda: FakeEmbedder()
    yield client
    app.dependency_overrides.clear()


def add(client, path):
    with path.open("rb") as f:
        return client.post("/kb/documents", files={"file": (path.name, f)})


@pytest.fixture
def policy(tmp_path):
    p = tmp_path / "leave_policy.txt"
    p.write_text(
        "Leave policy.\n\nEmployees accrue 18 days of annual leave per calendar year.\n\n"
        "Unused leave up to 30 days carries forward to the next year.\n\n"
        "Fleet maintenance.\n\nTrucks are serviced every 15,000 kilometres at an authorised workshop.\n",
        encoding="utf-8",
    )
    return p


def test_chunks_respect_target_pages_and_overlap():
    para = "Sentence number {} is here."
    body = "\n\n".join(" ".join(para.format(f"{p}.{s}") for s in range(6)) for p in range(16))
    text = f"[Page 1]\n{body}\n\n[Page 2]\nShort second page.\n\n[Page 3]\n"
    chunks = chunk_text(text, "pdf")
    assert all(len(c.text) <= TARGET for c in chunks)
    assert {c.page for c in chunks} == {1, 2}  # blank page 3 yields nothing
    assert chunks[-1].text == "Short second page."
    page1 = [c.text for c in chunks if c.page == 1]
    assert len(page1) > 1
    overlaps = [a.split("\n")[-1] for a, b in zip(page1, page1[1:]) if b.startswith(a.split("\n")[-1])]
    assert overlaps and all(len(o) <= OVERLAP for o in overlaps)


def test_chunks_split_unbroken_text():
    chunks = chunk_text("x" * (TARGET * 2 + 5), "txt")
    assert [len(c.text) for c in chunks] == [TARGET, TARGET, 5]
    assert chunks[0].page is None


def test_kb_requires_token(kb_client):
    r = kb_client.get("/kb/documents", headers={"X-Service-Token": "wrong"})
    assert r.status_code == 401


def test_add_search_delete(kb_client, policy, files):
    r = add(kb_client, policy)
    assert r.status_code == 200, r.text
    doc = r.json()
    assert doc["doc_id"].startswith("kb_") and doc["kind"] == "txt" and doc["chunks"] == 1
    assert doc["embed_model"] == "fake-embed"

    pdf = add(kb_client, files / "report.pdf").json()
    assert pdf["pages"] == 1

    r = kb_client.post("/kb/search", json={"query": "How often are fruit stock levels summarised?", "top_k": 2})
    assert r.status_code == 200, r.text
    top = r.json()["results"][0]
    assert top["file_name"] == "report.pdf" and top["page"] == 1
    assert "fruit stock" in top["text"] and top["similarity"] > 0

    listed = kb_client.get("/kb/documents").json()["documents"]
    assert [d["doc_id"] for d in listed] == [doc["doc_id"], pdf["doc_id"]]

    assert kb_client.delete(f"/kb/documents/{pdf['doc_id']}").json() == {"deleted": True}
    assert kb_client.delete(f"/kb/documents/{pdf['doc_id']}").status_code == 404
    results = kb_client.post("/kb/search", json={"query": "fruit stock"}).json()["results"]
    assert {x["file_name"] for x in results} == {"leave_policy.txt"}


def test_keyword_match_is_fused(kb_client, policy, tmp_path):
    add(kb_client, policy)
    other = tmp_path / "other.txt"
    other.write_text("Workshop opening hours are nine to five.", encoding="utf-8")
    add(kb_client, other)
    results = kb_client.post("/kb/search", json={"query": "kilometres"}).json()["results"]
    assert results[0]["file_name"] == "leave_policy.txt"


def test_duplicate_and_unsupported(kb_client, policy, files):
    first = add(kb_client, policy).json()
    r = add(kb_client, policy)
    assert r.status_code == 409 and first["doc_id"] in r.json()["error"]["message"]

    r = add(kb_client, files / "sales_eu.csv")
    assert r.status_code == 400 and r.json()["error"]["code"] == "UNSUPPORTED_FILE"


def test_search_validation_and_empty_kb(kb_client):
    assert kb_client.post("/kb/search", json={"query": "anything"}).json()["results"] == []
    assert kb_client.post("/kb/search", json={"query": "   "}).status_code == 422
    assert kb_client.post("/kb/search", json={"query": "x", "top_k": 50}).status_code == 422
    assert kb_client.delete("/kb/documents/../../etc").status_code in (400, 404)


def test_other_model_documents_are_not_searched(kb_client, policy):
    from app.main import app, get_embedder

    add(kb_client, policy)
    app.dependency_overrides[get_embedder] = lambda: FakeEmbedder("other-embed")
    body = kb_client.post("/kb/search", json={"query": "leave"}).json()
    assert body["results"] == [] and "different model" in body["warnings"][0]


def test_embedding_server_down(client, policy):
    from app.main import app, get_embedder

    app.dependency_overrides[get_embedder] = lambda: OllamaEmbedder("http://127.0.0.1:9", "m", timeout=2)
    try:
        r = add(client, policy)
    finally:
        app.dependency_overrides.clear()
    assert r.status_code == 503 and r.json()["error"]["code"] == "EMBEDDINGS_UNAVAILABLE"
    assert client.get("/kb/documents").json()["documents"] == []
