# Data Analysis Agent (n8n + local LLM + pandas)

n8n orchestrates, a Python service computes every number, and a local LLM (Ollama) plans
and explains. The LLM never calculates and never writes code that gets executed.

## Layout

| Path | Purpose |
|---|---|
| `docker-compose.yml` | `analysis` (FastAPI + pandas + knowledge base) on 127.0.0.1:8000, `ollama` on 127.0.0.1:11434 |
| `.env` | Service token and limits (created locally, mode 600, not in git). Template: `.env.example` |
| `analysis-service/` | Python service: `app/` code, `tests/` pytest suite, `certs/` extra root CAs for the build |
| `models/` | GGUF model files and their Modelfiles (the `.gguf` files are not in git) |
| `scripts/start-n8n.sh` | Starts n8n on 127.0.0.1 with a 50 MiB upload cap |
| `workflows/` | n8n workflow exports (import with `n8n import:workflow --input=<file>`) |

## Running

```bash
docker compose up -d                       # analysis service + ollama
./scripts/start-n8n.sh                     # n8n on http://127.0.0.1:5678
```

Installing the model (once). `ollama pull` fails on this network: the proxy blocks the Ollama
registry's file storage. The same model is downloaded from Qwen's official Hugging Face repo instead:

```bash
curl -L -C - -o models/qwen2.5-3b-instruct-q4_k_m.gguf \
  https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/qwen2.5-3b-instruct-q4_k_m.gguf
echo "626b4a6678b86442240e33df819e00132d3ba7dddfe1cdc4fbb18e0a9615c62d  models/qwen2.5-3b-instruct-q4_k_m.gguf" | sha256sum -c -
docker compose exec ollama ollama create qwen2.5:3b -f /models/Modelfile.qwen2.5-3b
```

The knowledge base also needs the embedding model, from Nomic's official Hugging Face repo:

```bash
curl -L -C - -o models/nomic-embed-text-v1.5.Q8_0.gguf \
  https://huggingface.co/nomic-ai/nomic-embed-text-v1.5-GGUF/resolve/main/nomic-embed-text-v1.5.Q8_0.gguf
echo "3e24342164b3d94991ba9692fdc0dd08e3fd7362e0aacc396a9a5c54a544c3b7  models/nomic-embed-text-v1.5.Q8_0.gguf" | sha256sum -c -
docker compose exec ollama ollama create nomic-embed-text:v1.5 -f /models/Modelfile.nomic-embed-text
```

Measured on this machine (CPU only, 8 cores): about 20 s to load the model after idle, and about 20–40 s
per planning call.

Run the tests:

```bash
cd analysis-service
docker build --target test -t data-agent-analysis:test . && docker run --rm data-agent-analysis:test
```

## Analysis service API (Stage 1)

Every route except `/health` needs the header `X-Service-Token: <SERVICE_TOKEN from .env>`.
Errors always look like `{"error": {"code", "message", "hint"}}`.

| Route | Does |
|---|---|
| `GET /health` | Liveness |
| `POST /ingest` (multipart `session_id`, `file`) | Validates, parses, types and stores a file; returns its profile |
| `GET /sessions/{sid}/profiles?format=full\|compact` | All profiles in a session; `compact` is a short text summary for the LLM |
| `GET /sessions/{sid}/datasets/{id}/text?offset=&limit=` | Extracted text (TXT, PDF) in chunks |
| `DELETE /sessions/{sid}` | Deletes a session's files |
| `POST /admin/cleanup` | Deletes sessions older than `RETENTION_DAYS` |

What ingestion handles:

- **CSV/TSV**: encoding detection, delimiter detection (`,` `;` tab `|`), ragged rows.
- **Excel**: every sheet, title rows above the header, empty sheets skipped. Legacy `.xls` is rejected with a hint.
- **TXT**: kept as text; also parsed as a table when it is consistently delimited.
- **PDF**: text per page plus ruled tables. Scanned PDFs get a warning (no OCR).
- **Types**: integer, decimal, boolean, datetime, category, text, identifier. A column converts only
  when 95% of values agree. Dropped values, European decimals, accounting negatives, percents,
  leading-zero codes and ambiguous dates are all recorded as column notes.
- **Safety**: token auth, size limit, content-vs-extension check, xlsx zip-bomb limits,
  `defusedxml`, UUID-style storage paths, session-ID validation.

## Knowledge base (RAG)

A persistent document collection, shared by every session and never removed by retention cleanup.
Stored in `/data/kb/kb.sqlite` on the `analysis-data` volume.

| Route | Does |
|---|---|
| `POST /kb/documents` (multipart `file`) | Extracts, chunks and embeds a TXT or PDF; returns `doc_id`, `chunks`, `pages`. The same file twice returns 409 |
| `GET /kb/documents` | Every document, with the embedding model it was indexed with |
| `DELETE /kb/documents/{doc_id}` | Removes a document and its passages |
| `POST /kb/search` (JSON `query`, `top_k` 1-20, default 5) | Best passages with `file_name`, `page`, `text`, `similarity` (cosine) and `score` (fused rank) |

How it works:

- **Chunks** are paragraphs packed to about 1,200 characters, with up to 200 characters of overlap.
  A chunk never spans two PDF pages, so every passage cites one page.
- **Search** is hybrid. Vector similarity (nomic-embed-text v1.5, with its required `search_document:` /
  `search_query:` prefixes) and BM25 keyword match (SQLite FTS5) are merged by reciprocal rank fusion,
  so exact terms (codes, names) and paraphrased questions both work.
- **Answers** are written in n8n by qwen2.5:3b, from the numbered passages only. The reply lists
  only the sources the model cited, and says when nothing in the knowledge base answers the question.
- **Tables stay out**: CSV and Excel files are rejected here. Numbers in tables are computed through `/ingest`.
- **Model changes**: each document records the model that embedded it. After changing `EMBED_MODEL`,
  documents from the old model are skipped with a warning until they are deleted and re-added.

Measured on this machine (4-core i7-1165G7, CPU only): indexing runs at about 1.9 passages per
second (a 40-page PDF: 85 passages in 46 s), so about 12 s for 10 pages and 6 minutes for 300. A search takes about
0.1 s and an answer 30–40 s. Embeddings are held in memory for search at about 3 KB per passage.

## Stage checklists

### Stage 0: environment
1. Stop the running `n8n start` (Ctrl+C), then run `./scripts/start-n8n.sh`.
   Check: `ss -ltnp | grep 5678` shows `127.0.0.1:5678`.
2. *Stage 0 - Upload Echo*: open the chat, attach files. Each is listed with type and size.
3. *Stage 0 - Local LLM Ping*: execute. *Check Reply* shows `ok: true`.

### Stage 1: ingestion
1. In n8n create a credential: **Header Auth**, name `X-Service-Token`, value = `SERVICE_TOKEN` from `.env`.
2. *Stage 1 - Ingest Test* → *Ingest File* node → select that credential → save.
3. Open the chat and attach your own CSV / XLSX / PDF / TXT files. Check the row counts, column
   types and notes against what you know about each file.

### Knowledge base
1. Install the embedding model (see *Running*), then `docker compose up -d --build analysis`.
2. Import `workflows/kb-chat.json`. In *Add Document* and *Search KB* select the `X-Service-Token` credential, then save.
3. Open the chat and attach TXT or PDF files. Each reply shows the `doc_id` and the passage count.
4. Ask questions with no attachment. Check that each answer cites the right file and page, and that a
   question the documents do not cover gets "I could not find this in the knowledge base."

## Switching to Claude later

Only the LLM HTTP nodes change (Stage 3+, and *Ollama Answer* in the knowledge base workflow). For `claude-opus-5-5` do not send `temperature`,
assistant prefill, `thinking: {type: "disabled"}` or forced `tool_choice`: each returns HTTP 400.
`workflows/stage0-claude-ping.json` is a ready-made test call.
