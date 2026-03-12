# Tiliqua
## LLM Document Mining Framework (Batch + Tools Ready)

A modular framework for **large-scale document mining, systematic literature review (SLR), and bibliographic extraction** using OpenAI models.

This framework is designed to:

- Work efficiently with **OpenAI Batch API**
- Maintain **backward compatibility with existing code**
- Support **future tools** (web search, function calling, tool search, MCP)
- Preserve **existing prompts** without modification
- Enable scalable **knowledge extraction pipelines**

The architecture treats document processing as a **dataflow pipeline** that converts PDFs into structured knowledge.

---

# Core Design Principles

## 1. Backward Compatibility

Existing prompts and logic are preserved.

Old code:

```python
body = {"model": gpt_model, "messages": messages}
```

New architecture simply replaces this with:

```python
body = build_llm_body(messages, execution_config)
```

All prompts remain unchanged.

---

## 2. Batch-First Architecture

Most document analysis tasks follow a **map operation pattern**:

```
segments → LLM extraction → structured outputs
```

Therefore the system is optimized for:

- Batch API
- Parallel processing
- Large document collections

---

## 3. Tool-Ready Execution

Modern models (GPT-5.4+) support tools such as:

- Web search
- File search
- Tool search
- Function calling
- Remote MCP servers

The framework introduces an **ExecutionConfig abstraction** that enables tools without changing prompts or pipeline logic.

---

# High-Level Pipeline

The system transforms documents into structured knowledge.

```
PDF
 ↓
Raw Text
 ↓
Segments
 ↓
LLM Tasks
 ↓
Batch API
 ↓
Results
 ↓
Structured Knowledge
```

More explicitly:

```
PDF Folder
   ↓
Raw Text Table
   ↓
Segments Table
   ↓
Task Table
   ↓
Batch API
   ↓
Result Table
   ↓
Knowledge Tables
```

Each step converts **one table into another**.

---

# Core Abstractions

## Document

Represents a source document.

```
Document
├─ doc_id
├─ metadata
├─ raw_text
├─ segments
└─ knowledge
```

Example:

```python
doc = {
  "FID": "F00012",
  "metadata": {...},
  "segments": [...],
  "knowledge": {...}
}
```

---

## Segment

All LLM operations occur at the **segment level**.

```python
segment = {
  "doc_id": FID,
  "segment_id": i,
  "text": "...",
  "page": ...
}
```

Segments are generated using chunking functions such as:

```
chunk_text_with_overlap()
```

Segments enable:

- Batch processing
- Parallel processing
- Stable memory updates

---

## Task

Every LLM operation is represented as a **task**.

```
Task
├─ task_id
├─ task_type
├─ doc_id
├─ segment_id
└─ prompt
```

Example task types:

- RIS_METADATA
- STATEMENT_EXTRACTION
- REFERENCE_EXTRACTION
- RELEVANCE_CLASSIFICATION
- METHOD_EXTRACTION

Tasks are later converted into **Batch API jobs**.

---

# Knowledge Objects

LLM extraction produces **structured objects**.

## Statements

```python
statement = {
  "doc_id":
  "segment_id":
  "type":
  "text":
  "citation":
}
```

Types:

- ORIGINAL_CONTRIBUTION
- REFERENCED_STATEMENT
- DATA_METHOD

---

## RIS Metadata

```python
RISRecord = {
 "doc_id":
 "title":
 "authors":
 "year":
 "journal":
 "doi":
}
```

---

## Citation Edges

Later stages can build **citation graphs**.

```python
CitationEdge = {
  "source_doc":
  "target_reference":
  "statement_id":
}
```

Example:

```
Paper A
   ↓ cites
Paper B
```

---

# ExecutionConfig (Tool Compatibility Layer)

`ExecutionConfig` controls how the model is executed.

Example:

```python
ExecutionConfig = {
   "model": "gpt-5-mini",
   "tools": None,
   "tool_choice": None
}
```

Future configuration example:

```python
ExecutionConfig = {
   "model": "gpt-5.4",
   "tools": [
       {"type": "web_search"}
   ],
   "tool_choice": "auto"
}
```

This allows **tool usage without changing prompts or task definitions**.

---

# Core Runtime Module

## `llm_runtime.py`

```python
import json

def default_execution_config(model="gpt-5-mini"):
    """
    Default execution config.
    Compatible with old code.
    """
    return {
        "model": model,
        "tools": None,
        "tool_choice": None
    }


def build_llm_body(messages, execution_config=None):
    """
    Build the body for OpenAI API calls.
    Supports future tool usage.
    """

    if execution_config is None:
        execution_config = default_execution_config()

    body = {
        "model": execution_config.get("model", "gpt-5-mini"),
        "messages": messages
    }

    tools = execution_config.get("tools")
    if tools:
        body["tools"] = tools

    tool_choice = execution_config.get("tool_choice")
    if tool_choice:
        body["tool_choice"] = tool_choice

    return body


def build_batch_job(custom_id, messages, execution_config=None):
    """
    Create a standard OpenAI Batch API job.
    """

    body = build_llm_body(messages, execution_config)

    job = {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": body
    }

    return job


def save_jsonl(lines, suffix):
    """
    Save JSONL batch file.
    """

    import tempfile

    tmp = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=suffix,
        mode="w",
        encoding="utf-8"
    )

    tmp.write("\n".join(lines))
    tmp.close()

    print(f"Batch file saved: {tmp.name}")
    return tmp.name
```

---

# Batch Runtime

## `batch_runtime.py`

```python
import os
import json

def submit_batch(client, jsonl_path, endpoint="/v1/chat/completions", completion_window="24h"):
    """
    Upload batch job to OpenAI.
    """

    with open(jsonl_path, "rb") as f:
        file_obj = client.files.create(file=f, purpose="batch")

    batch = client.batches.create(
        input_file_id=file_obj.id,
        endpoint=endpoint,
        completion_window=completion_window
    )

    print(f"Batch submitted: {batch.id}")
    return batch.id


def retrieve_batch_result(client, batch_id):
    """
    Get batch status.
    """

    batch_obj = client.batches.retrieve(batch_id)

    status = getattr(batch_obj, "status", "unknown")
    output_file_id = getattr(batch_obj, "output_file_id", None)

    return status, output_file_id


def download_batch_output(client, output_file_id, save_path=None):
    """
    Download batch output JSONL.
    """

    file_content = client.files.content(output_file_id).text

    if save_path:
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(file_content)

    lines = [
        json.loads(line)
        for line in file_content.strip().splitlines()
        if line.strip()
    ]

    return lines
```

---

# Example Batch Task Generator

Example: **SLR relevance detection**

```python
from llm_runtime import build_batch_job, save_jsonl, default_execution_config

def chunk_content_to_batch(
       df,
       chunk_number,
       affix,
       gpt_model="gpt-5-mini",
       execution_config=None,
       encryption="slryesrelevant00800slr",
       topic_col="topic",
       memory_col="summary_memory"
):

    if execution_config is None:
        execution_config = default_execution_config(gpt_model)

    lines = []

    for _, row in df.iterrows():

        chunk_col = f"chunk_{chunk_number}"

        chunk_content = str(row[chunk_col])
        summary_memory = str(row.get(memory_col, "BLANK"))
        slr_topic = str(row.get(topic_col, "general topic"))
        fid = row["FID"]

        prompt = (
            f"Here is one chunk of an academic/professional publication:\n"
            f"Text: {chunk_content}\n\n"
            f"Here is the memorised summary for all previous chunks for the same publication:\n"
            f"Memory: {summary_memory}\n\n"
            "========================\n"
            f"I am conducting a systematic literature review on the following topic:\n"
            f"SLR Topic: {slr_topic}\n"
            "========================\n"
            f"Based on the provided chunk of article, do you think the paper is relevant? "
            f"If yes, reply with this exact encryption: {encryption}; "
            f"otherwise, give me an updated summary."
        )

        messages = [
            {"role": "system", "content": "You are a helpful and rigorous research assistant."},
            {"role": "user", "content": prompt},
        ]

        job = build_batch_job(
            custom_id=f"{fid}_{affix}_chunk{chunk_number:03d}",
            messages=messages,
            execution_config=execution_config
        )

        lines.append(json.dumps(job))

    return save_jsonl(lines, f"_{affix}_chunk{chunk_number:03d}.jsonl")
```

---

# Enabling Tools (Future)

### Enable web search

```python
execution_config = {
   "model": "gpt-5.4",
   "tools": [
       {"type": "web_search"}
   ],
   "tool_choice": "auto"
}
```

### Enable function calling

```python
execution_config = {
   "model": "gpt-5.4",
   "tools": [
       {
           "type": "function",
           "name": "resolve_doi",
           "description": "Resolve DOI metadata"
       }
   ]
}
```

### Enable tool search

```python
execution_config = {
   "model": "gpt-5.4",
   "tools": [
       {"type": "tool_search"}
   ]
}
```

---

# System Advantages

This architecture provides:

- Minimal changes to existing code
- Full Batch API compatibility
- Tool-ready execution
- Prompt stability
- Scalable document processing
- Easy future extension

Future capabilities may include:

- web_search
- file_search
- tool_search
- function calling
- remote MCP servers

All enabled by modifying **ExecutionConfig only**.

---

# Future Extensions

This framework can evolve into a **Scientific Knowledge Graph Builder**.

Potential pipeline:

```
PDF
 ↓
RIS Metadata
 ↓
Statement Extraction
 ↓
Citation Extraction
 ↓
Dataset / Method Extraction
 ↓
Citation Graph
```

Example graph:

```
Paper → Citation → Dataset → Method → Result
```

This enables **automated literature analysis and research intelligence systems**.
