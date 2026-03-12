# Tiliqua

**LLM Document Mining Framework**
*OpenAI Responses API · Batch API · OCR Pipeline · RAG*

Tiliqua is a Python framework for converting PDF research corpora into structured, queryable knowledge datasets. It is built for:

- Systematic literature review (SLR)
- Bibliographic extraction and RIS generation
- Citation graph construction
- Dataset and method mining
- RAG knowledge bases over extracted research knowledge

---

## Table of Contents

1. [Installation](#installation)
2. [End-to-End Pipeline](#end-to-end-pipeline)
3. [Module Reference](#module-reference)
   - [pdf_reader](#pdf_reader)
   - [extraction](#extraction)
   - [rag](#rag)
   - [llm_runtime](#llm_runtime)
   - [batch_runtime](#batch_runtime)
   - [text_utils](#text_utils)
   - [models](#models)
4. [Multiple Clients (Speed Up Processing)](#multiple-clients-speed-up-processing)
5. [Large-Scale Batch Processing](#large-scale-batch-processing)
6. [Demo Notebook](#demo-notebook)
7. [Output Files](#output-files)
8. [Project Structure](#project-structure)

---

## Installation

```bash
pip install pymupdf numpy openai pandas
pip install -e .          # install tiliqua from source
```

Optional (for the demo notebook):

```bash
pip install jupyter ipywidgets
```

**Requires** Python ≥ 3.10 and an `OPENAI_API_KEY` environment variable.

---

## End-to-End Pipeline

```
PDF file
    │
    ▼
PyMuPDF text extraction  ──(fallback if < 1000 chars)──▶  OpenAI OCR (Responses API)
    │                                                         page-by-page, file uploaded once
    ▼
Raw text  →  saved as  .txt
    │
    ▼
LLM extraction  (3 parallel Responses API calls)
    ├── RIS metadata          →  .ris file
    ├── Contribution &        →  statements.csv
    │   referenced statements
    └── Datasets & methods    →  datasets.csv  /  methods.csv
    │
    ▼
Full knowledge JSON  →  knowledge.json
    │
    ▼
OpenAI Embeddings  (text-embedding-3-small)
    │
    ▼
In-memory vector store  (numpy cosine similarity)
    │
    ▼
RAG query  (Responses API, grounded answers)
```

For **large corpora** (hundreds of PDFs), swap the direct Responses API calls for the Batch API path — same prompts, 50 % cost reduction, 24-hour completion window.

---

## Module Reference

### pdf_reader

`tiliqua/pdf_reader.py`

Handles all PDF-to-text conversion. PyMuPDF is always tried first; if the result is fewer than 1 000 characters the system automatically switches to OpenAI OCR via the Responses API.

#### `read_pdf_pymupdf(pdf_pathfile)`

Fast extraction for digital (non-scanned) PDFs.

```python
from tiliqua.pdf_reader import read_pdf_pymupdf

text = read_pdf_pymupdf("paper.pdf")
```

#### `upload_pdf_to_openai(client, pdf_pathfile)`

Upload a PDF to the OpenAI Files API once, reuse the returned `file_id` across all page OCR calls.

```python
file_id = upload_pdf_to_openai(client, "paper.pdf")
```

#### `OCR_openai_single_page(client, file_id, page_number, model_name)`

OCR one page of a previously uploaded PDF.

```python
response, page_text = OCR_openai_single_page(client, file_id, page_number=1)
```

The OCR prompt instructs the model to transcribe exactly — no summarising, no paraphrasing, numbers and punctuation preserved, tables reproduced in Markdown, figures described.

#### `OCR_pdf_pages(client, pdf_pathfile, model_name)`

OCR every page of a PDF. Uploads the file once, iterates pages, concatenates results.

```python
full_text = OCR_pdf_pages(client, "scanned_paper.pdf")
```

#### `read_pdf(client, pdf_pathfile, maxlen=None, model_name="gpt-4o-mini", OCR_only=False)`

Auto-switching reader — the main entry point for single-file extraction.

```python
from tiliqua.pdf_reader import read_pdf

text = read_pdf(client, "paper.pdf")
# PyMuPDF used for digital PDFs; OCR triggered automatically for scanned ones
```

| Parameter | Description |
|---|---|
| `client` | `openai.OpenAI` instance (only needed if OCR is triggered) |
| `maxlen` | Truncate PyMuPDF result to this character count |
| `OCR_only` | Skip PyMuPDF entirely |

#### `convert_all_pdfs_to_text(pdf_folder, output_csv, client, model_name)`

Convert an entire directory of PDFs to a CSV table with columns `FID`, `filename`, `raw_text`. Each PDF is assigned a zero-padded FID (F00001, F00002, …).

```python
from tiliqua.pdf_reader import convert_all_pdfs_to_text

df = convert_all_pdfs_to_text("pdf_corpus/", "raw_texts.csv", client=client)
```

---

### extraction

`tiliqua/extraction.py`

Structured knowledge extraction via the OpenAI Responses API. Three extraction types are available as both direct API calls (single document) and Batch API job builders (large scale).

#### Direct API calls

Each function sends the first 15 000 characters of the document text to the model and returns a parsed Python dict.

##### `extract_ris_metadata(client, text, model_name)`

Extracts bibliographic metadata as a dict with keys: `type`, `title`, `authors`, `year`, `journal`, `doi`, `abstract`, `keywords`.

```python
from tiliqua.extraction import extract_ris_metadata

ris = extract_ris_metadata(client, raw_text)
# {'type': 'JOUR', 'title': 'Example Paper', 'authors': ['Smith, John'], ...}
```

##### `extract_statements(client, text, model_name)`

Extracts two lists:

- `original_contributions` — sentences where the authors claim their own novel work
  *(signals: "we propose", "this paper", "our method", "we introduce", …)*
- `referenced_statements` — sentences citing other works
  *(signals: "Smith (2020)", "[1]", "X et al.", "prior studies", …)*

```python
from tiliqua.extraction import extract_statements

stmts = extract_statements(client, raw_text)
for s in stmts["original_contributions"]:
    print(s["text"])
```

Each item contains `text` and `context` (brief note on section/topic); referenced statements also carry `citation`.

##### `extract_data_methods(client, text, model_name)`

Extracts:

- `datasets` — name, description, source
- `methods` — name, description, category
- `evaluation_metrics` — list of metric names

```python
from tiliqua.extraction import extract_data_methods

dm = extract_data_methods(client, raw_text)
for ds in dm["datasets"]:
    print(ds["name"], "—", ds["description"])
```

##### `extract_all(client, text, model_name)`

Convenience wrapper that runs all three extractions and returns a single dict with keys `ris`, `statements`, `data_methods`.

```python
from tiliqua.extraction import extract_all

knowledge = extract_all(client, raw_text)
```

#### `format_ris(ris, doc_id="")`

Converts an extracted RIS dict to the standard plain-text RIS format.

```python
from tiliqua.extraction import format_ris

ris_text = format_ris(knowledge["ris"], doc_id="F00001")
# TY  - JOUR
# ID  - F00001
# AU  - Smith, John
# TI  - Example Paper
# ...
# ER  -
```

#### Batch job builders

For large corpora, use these functions to build JSONL jobs targeting the `/v1/responses` endpoint.

```python
from tiliqua.extraction import (
    build_ris_batch_job,
    build_statements_batch_job,
    build_data_methods_batch_job,
)

job = build_ris_batch_job("F00001_ris", text=raw_text)
# {'custom_id': 'F00001_ris', 'method': 'POST', 'url': '/v1/responses', 'body': {...}}
```

Pass an `ExecutionConfig` or dict as `execution_config` to control the model.

---

### rag

`tiliqua/rag.py`

In-memory embedding-based retrieval-augmented generation. No external vector database required — similarity search runs entirely in numpy.

**Embedding model:** `text-embedding-3-small` (default)
**Generation model:** OpenAI Responses API

#### `build_rag_store(client, knowledge, embedding_model)`

Flattens an `extract_all()` knowledge dict into typed text chunks (RIS metadata, contributions, referenced statements, datasets, methods, evaluation metrics) and embeds each one.

```python
from tiliqua.rag import build_rag_store

store = build_rag_store(client, knowledge)
# RAG store built: 12 chunks embedded.
```

Returns a list of dicts with keys `text`, `source`, `metadata`, `embedding`.

#### `build_rag_store_from_texts(client, texts, sources, embedding_model)`

Build a store directly from a list of raw text strings — useful for embedding raw document chunks instead of structured knowledge.

```python
from tiliqua.text_utils import chunk_text_with_overlap
from tiliqua.rag import build_rag_store_from_texts

chunks = chunk_text_with_overlap(raw_text, chunk_size=800, overlap=100)
store = build_rag_store_from_texts(client, chunks)
```

#### `retrieve(store, client, query, top_k, embedding_model)`

Embed the query and return the top-k most similar store entries ranked by cosine similarity. Each result includes a `score` field.

```python
from tiliqua.rag import retrieve

results = retrieve(store, client, "What evaluation metrics were used?", top_k=5)
for r in results:
    print(f"[{r['source']}] ({r['score']:.3f}) {r['text'][:80]}")
```

#### `query_rag(store, client, question, top_k, model_name, embedding_model)`

Full RAG pipeline: retrieve top-k chunks → build grounded prompt → call Responses API → return answer with cited sources.

```python
from tiliqua.rag import query_rag

result = query_rag(store, client, "What is the main contribution of this paper?")
print(result["answer"])
for src in result["sources"]:
    print(f"  [{src['source']}] {src['text'][:80]}")
```

Returns `{"answer": str, "sources": list[dict]}`.

---

### llm_runtime

`tiliqua/llm_runtime.py`

Low-level body builders for the OpenAI API. Accepts either a plain `dict` or an `ExecutionConfig` dataclass as `execution_config`.

| Function | Endpoint | Use |
|---|---|---|
| `build_llm_body(messages, config)` | Chat Completions | Direct chat calls |
| `build_batch_job(custom_id, messages, config)` | `/v1/chat/completions` | Legacy batch jobs |
| `build_responses_body(messages, config)` | Responses API | Direct responses calls |
| `build_responses_batch_job(custom_id, messages, config)` | `/v1/responses` | Recommended batch jobs |
| `save_jsonl(lines, suffix)` | — | Write JSONL to a temp file |
| `default_execution_config(model)` | — | Default config dict |
| `make_clients(api_keys, model)` | — | Create client pool from API key list |
| `MultiClient(clients)` | — | Thread-safe round-robin client wrapper |

**Responses API batch job structure:**

```python
from tiliqua.llm_runtime import build_responses_batch_job

job = build_responses_batch_job(
    custom_id="F00001_chunk001",
    messages=[{"role": "user", "content": "..."}],
)
# {
#   "custom_id": "F00001_chunk001",
#   "method":    "POST",
#   "url":       "/v1/responses",
#   "body":      {"model": "gpt-4o-mini", "input": [...]}
# }
```

---

### batch_runtime

`tiliqua/batch_runtime.py`

Wraps the OpenAI Batch API: upload, create, poll, download.

#### `submit_batch(client, jsonl_path, endpoint, completion_window)`

Upload a JSONL file and create a batch job.

```python
from tiliqua.batch_runtime import submit_batch

batch_id = submit_batch(client, "jobs.jsonl", endpoint="/v1/responses")
```

#### `retrieve_batch_result(client, batch_id)`

Poll for status. Returns `(status, output_file_id)`.

```python
status, output_file_id = retrieve_batch_result(client, batch_id)
# status: "validating" | "in_progress" | "completed" | "failed" | "expired"
```

#### `download_batch_output(client, output_file_id, save_path=None)`

Download and parse the completed JSONL output. Returns a list of dicts.

```python
lines = download_batch_output(client, output_file_id, save_path="results.jsonl")
```

#### `submit_batches_multi_client(clients, jsonl_paths, endpoint, completion_window)`

Distribute multiple JSONL batch files across a pool of clients in round-robin order. Returns a list of `(client_index, batch_id)` tuples.

```python
from tiliqua.batch_runtime import submit_batches_multi_client

results = submit_batches_multi_client(clients, [ris_jsonl, stmt_jsonl], endpoint="/v1/responses")
# [(0, 'batch_aaa'), (1, 'batch_bbb')]
```

---

### text_utils

`tiliqua/text_utils.py`

Text processing helpers.

#### `chunk_text_with_overlap(text, chunk_size=800, overlap=100)`

Split text into overlapping word-count chunks.

```python
from tiliqua.text_utils import chunk_text_with_overlap

chunks = chunk_text_with_overlap(raw_text, chunk_size=800, overlap=100)
```

#### `segment_document(doc_id, text, chunk_size, overlap, page)`

Produce a list of segment dicts (`doc_id`, `segment_id`, `text`, `page`) — pandas-friendly.

#### `extract_text_from_batch_line(line)`

Parse assistant text from a **Chat Completions** batch output line.

#### `extract_text_from_responses_batch_line(line)`

Parse assistant text from a **Responses API** batch output line. Handles both the `output_text` convenience field and the structured `output[].content[].text` path.

```python
from tiliqua.text_utils import extract_text_from_responses_batch_line

text = extract_text_from_responses_batch_line(line)
```

#### `extract_custom_id(line)`

Return `custom_id` from any batch output line.

---

### models

`tiliqua/models.py`

Dataclass abstractions. All support `.as_dict()` for pandas / CSV workflows.

| Class | Fields |
|---|---|
| `Document` | `doc_id`, `metadata`, `raw_text`, `segments`, `knowledge` |
| `Segment` | `doc_id`, `segment_id`, `text`, `page` |
| `Task` | `task_id`, `task_type`, `doc_id`, `segment_id`, `prompt`, `extra` |
| `Statement` | `doc_id`, `segment_id`, `type`, `text`, `citation` |
| `RISRecord` | `doc_id`, `title`, `authors`, `year`, `journal`, `doi` |
| `CitationEdge` | `source_doc`, `target_reference`, `statement_id` |
| `ExecutionConfig` | `model`, `tools`, `tool_choice` |
| `TaskType` (enum) | `RIS_METADATA`, `STATEMENT_EXTRACTION`, `REFERENCE_EXTRACTION`, `RELEVANCE_CLASSIFICATION`, `METHOD_EXTRACTION` |
| `StatementType` (enum) | `ORIGINAL_CONTRIBUTION`, `REFERENCED_STATEMENT`, `DATA_METHOD` |

---

## Multiple Clients (Speed Up Processing)

When you have multiple OpenAI API keys, Tiliqua can distribute work across them in parallel, significantly reducing wall-clock time for large corpora.

### Setup

```python
from tiliqua.llm_runtime import make_clients, MultiClient

api_keys = [
    'sk-proj-1',
    'sk-proj-2',
    'sk-proj-3',
    'sk-proj-4',
    'sk-proj-5',
]
api_keys = api_keys[::-1]   # reverse order if desired

clients, gpt_model = make_clients(api_keys, model='gpt-5-mini')
```

`make_clients` returns a plain `list[openai.OpenAI]` and the model string.
`MultiClient` wraps that list with a thread-safe round-robin `.next()` method.

```python
mc = MultiClient(clients)
client = mc.next()   # picks the next client in rotation
```

### Parallel PDF conversion

Pass `clients` to `convert_all_pdfs_to_text`. It spawns one worker thread per client and distributes PDFs round-robin:

```python
from tiliqua.pdf_reader import convert_all_pdfs_to_text

df = convert_all_pdfs_to_text(
    "pdf_corpus/",
    "raw_texts.csv",
    clients=clients,          # list or MultiClient
    model_name=gpt_model,
    max_workers=len(clients), # optional, defaults to len(clients)
)
```

### Distributing batch jobs across clients

Use `submit_batches_multi_client` to assign multiple JSONL files to different clients in round-robin:

```python
import json
from tiliqua.extraction import build_ris_batch_job, build_statements_batch_job
from tiliqua.llm_runtime import save_jsonl
from tiliqua.batch_runtime import submit_batches_multi_client, retrieve_batch_result, download_batch_output

# Build JSONL files
ris_jobs, stmt_jobs = [], []
for _, row in df.iterrows():
    fid, text = row["FID"], str(row["raw_text"])
    ris_jobs.append(json.dumps(build_ris_batch_job(f"{fid}_ris", text)))
    stmt_jobs.append(json.dumps(build_statements_batch_job(f"{fid}_stmts", text)))

ris_jsonl  = save_jsonl(ris_jobs,  "_ris.jsonl")
stmt_jsonl = save_jsonl(stmt_jobs, "_stmts.jsonl")

# Submit to different clients (round-robin)
results = submit_batches_multi_client(
    clients,
    [ris_jsonl, stmt_jsonl],
    endpoint="/v1/responses",
)
# results = [(0, 'batch_ris_id'), (1, 'batch_stmt_id')]

for client_idx, batch_id in results:
    print(f"Submitted {batch_id} via client {client_idx}")
```

To poll and download, use the matching client:

```python
import time

for client_idx, batch_id in results:
    client = clients[client_idx]
    while True:
        status, output_file_id = retrieve_batch_result(client, batch_id)
        if status == "completed":
            break
        if status in ("failed", "expired", "cancelled"):
            raise RuntimeError(f"Batch {batch_id} failed: {status}")
        time.sleep(60)
    lines = download_batch_output(client, output_file_id)
    print(f"Batch {batch_id}: {len(lines)} results")
```

### Summary

| Scenario | How to use |
|---|---|
| Parallel PDF → text | `convert_all_pdfs_to_text(..., clients=clients)` |
| Batch job distribution | `submit_batches_multi_client(clients, jsonl_paths)` |
| Manual round-robin | `mc = MultiClient(clients); c = mc.next()` |

---

## Large-Scale Batch Processing

For processing hundreds of PDFs, replace direct `extract_*` calls with Batch API jobs. This reduces cost by ~50 % and removes synchronous API call bottlenecks.

```python
import json
from tiliqua.pdf_reader import convert_all_pdfs_to_text
from tiliqua.text_utils import chunk_text_with_overlap
from tiliqua.extraction import build_ris_batch_job, build_statements_batch_job
from tiliqua.llm_runtime import save_jsonl
from tiliqua.batch_runtime import submit_batch, retrieve_batch_result, download_batch_output
from tiliqua.text_utils import extract_text_from_responses_batch_line, extract_custom_id

# 1. Convert PDFs to text table
df = convert_all_pdfs_to_text("pdf_corpus/", "raw_texts.csv", client=client)

# 2. Build batch jobs
ris_jobs, stmt_jobs = [], []
for _, row in df.iterrows():
    fid = row["FID"]
    text = str(row["raw_text"])
    ris_jobs.append(json.dumps(build_ris_batch_job(f"{fid}_ris", text)))
    stmt_jobs.append(json.dumps(build_statements_batch_job(f"{fid}_stmts", text)))

ris_jsonl  = save_jsonl(ris_jobs,  "_ris.jsonl")
stmt_jsonl = save_jsonl(stmt_jobs, "_stmts.jsonl")

# 3. Submit
ris_batch_id  = submit_batch(client, ris_jsonl,  endpoint="/v1/responses")
stmt_batch_id = submit_batch(client, stmt_jsonl, endpoint="/v1/responses")

# 4. Poll (status: "validating" → "in_progress" → "completed")
import time
for batch_id, label in [(ris_batch_id, "RIS"), (stmt_batch_id, "Statements")]:
    while True:
        status, output_file_id = retrieve_batch_result(client, batch_id)
        if status == "completed":
            break
        if status in ("failed", "expired", "cancelled"):
            raise RuntimeError(f"{label} batch failed: {status}")
        time.sleep(60)

# 5. Download and parse
ris_lines  = download_batch_output(client, retrieve_batch_result(client, ris_batch_id)[1])
stmt_lines = download_batch_output(client, retrieve_batch_result(client, stmt_batch_id)[1])

for line in ris_lines:
    fid  = extract_custom_id(line).replace("_ris", "")
    text = extract_text_from_responses_batch_line(line)
    print(fid, text[:80])
```

The SLR relevance detection example (`examples/slr_relevance.py`) demonstrates the same pattern applied to relevance classification using the Chat Completions endpoint.

---

## Demo Notebook

`demo.ipynb` provides an interactive single-PDF walkthrough of the full pipeline.

**To run:**

```bash
jupyter notebook demo.ipynb
```

**Edit the configuration cell:**

```python
PDF_PATH       = "path/to/your/paper.pdf"   # ← set this
OPENAI_API_KEY = "sk-..."                   # ← or set OPENAI_API_KEY env var
MODEL          = "gpt-4o-mini"
OUTPUT_DIR     = pathlib.Path("output")
```

Then run all cells. The notebook covers:

| Part | What happens |
|---|---|
| 1 — OCR & Text Extraction | `read_pdf()` auto-selects PyMuPDF or OCR; raw text saved to `.txt` |
| 2 — Structured Extraction | `extract_all()` produces RIS, statements, datasets/methods; saved to `.ris`, `.csv`, `.json` |
| 3 — RAG | `build_rag_store()` embeds the knowledge; `query_rag()` answers questions with cited sources |

---

## Output Files

| File | Description |
|---|---|
| `<stem>_raw_text.txt` | Full OCR or PyMuPDF plain text |
| `<stem>.ris` | Standard RIS bibliographic record |
| `<stem>_statements.csv` | Original contributions + referenced statements |
| `<stem>_datasets.csv` | Datasets identified in the paper |
| `<stem>_methods.csv` | Methods and algorithms identified |
| `<stem>_knowledge.json` | Complete extraction result (all fields) |

---

## Project Structure

```
tiliqua/
├── __init__.py          # package exports (version 0.2.0)
├── pdf_reader.py        # PyMuPDF extraction + OpenAI OCR fallback
├── extraction.py        # RIS, statement, data/method extraction + batch builders
├── rag.py               # embedding store + cosine retrieval + RAG query
├── llm_runtime.py       # body builders for Chat Completions & Responses API
├── batch_runtime.py     # Batch API: submit / poll / download
├── text_utils.py        # chunking, segmentation, batch output parsers
└── models.py            # Document, Segment, Task, Statement, RISRecord dataclasses

examples/
└── slr_relevance.py     # SLR relevance detection — end-to-end batch example

demo.ipynb               # interactive single-PDF demo notebook
```

---

## Dependencies

| Package | Purpose |
|---|---|
| `openai >= 1.30.0` | Responses API, Batch API, Embeddings API |
| `pymupdf >= 1.23.0` | Fast PDF text extraction (`fitz`) |
| `numpy >= 1.24.0` | Cosine similarity for RAG retrieval |
| `pandas >= 2.0.0` | Optional — CSV table workflows |
| `jupyter`, `ipywidgets` | Optional — demo notebook |

---

## License

MIT
