"""
Knowledge extraction for Tiliqua.

Provides:
- Direct Responses-API calls for single-document extraction
- Batch job builders for large-scale processing

Extracts:
  * RIS bibliographic metadata
  * Original contribution statements
  * Referenced statements
  * Datasets & methodology
"""

from __future__ import annotations
import json
from typing import Optional


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_RIS_PROMPT = """\
Extract bibliographic metadata from the academic paper text below.
Return a single JSON object with ONLY these keys (use null for missing values):

{{
  "type":     "publication type: JOUR, CONF, BOOK, RPRT, or THES",
  "title":    "full title of the paper",
  "authors":  ["Last, First", ...],
  "year":     1234,
  "journal":  "journal or conference name",
  "doi":      "10.xxxx/yyyy or null",
  "abstract": "abstract text or null",
  "keywords": ["keyword1", "keyword2", ...]
}}

Return ONLY valid JSON — no markdown fences, no explanation.

PAPER TEXT:
{text}
"""

_STATEMENTS_PROMPT = """\
Extract two kinds of statements from the academic paper text below.

1. original_contributions — sentences where the authors claim their OWN novel work.
   Signals: "we propose", "we present", "this paper", "our approach", "our method",
            "we develop", "we introduce", "we demonstrate", "we show that", "our model",
            "our algorithm", "in this work", "we contribute".

2. referenced_statements — sentences where OTHER works are cited.
   Signals: author-year patterns like "Smith (2020)", numbered refs like "[1]" or "(1)",
            phrases like "previous work", "prior studies", "X et al.".

Return a single JSON object:
{{
  "original_contributions": [
    {{"text": "...", "context": "brief note on the section or topic"}}
  ],
  "referenced_statements": [
    {{"text": "...", "citation": "cited author/ref if visible", "context": "..."}}
  ]
}}

Return ONLY valid JSON — no markdown fences, no explanation.

PAPER TEXT:
{text}
"""

_DATA_METHOD_PROMPT = """\
Extract datasets and methods from the academic paper text below.

Return a single JSON object:
{{
  "datasets": [
    {{
      "name": "dataset name",
      "description": "brief description",
      "source": "origin or URL if mentioned, else null"
    }}
  ],
  "methods": [
    {{
      "name": "method or algorithm name",
      "description": "brief description",
      "category": "e.g. deep learning, statistical, NLP, remote sensing"
    }}
  ],
  "evaluation_metrics": ["metric1", "metric2"]
}}

Return ONLY valid JSON — no markdown fences, no explanation.

PAPER TEXT:
{text}
"""


# ---------------------------------------------------------------------------
# JSON helper
# ---------------------------------------------------------------------------

def _safe_parse_json(text: str) -> Optional[dict]:
    """
    Parse a JSON string returned by an LLM, stripping markdown fences if present.
    Returns None on failure.
    """
    if text is None:
        return None
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        # drop opening fence (and optional language tag) + closing fence
        inner = "\n".join(
            line for line in lines[1:]
            if not line.strip().startswith("```")
        )
        stripped = inner.strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        return None


# ---------------------------------------------------------------------------
# Direct Responses-API calls (single document)
# ---------------------------------------------------------------------------

def extract_ris_metadata(
    client,
    text: str,
    model_name: str = "gpt-4o-mini",
) -> dict:
    """
    Extract RIS-style bibliographic metadata from raw paper text.

    Parameters
    ----------
    client : openai.OpenAI
    text : str
        Raw text of the paper (will be truncated to 15 000 chars).
    model_name : str

    Returns
    -------
    dict
        Parsed metadata dict, or {"error": ..., "raw": ...} on failure.
    """
    prompt = _RIS_PROMPT.format(text=text[:15000])
    response = client.responses.create(
        model=model_name,
        input=[{"role": "user", "content": prompt}],
        stream=False,
    )
    parsed = _safe_parse_json(response.output_text)
    if parsed is None:
        return {"error": "JSON parse failed", "raw": response.output_text}
    return parsed


def extract_statements(
    client,
    text: str,
    model_name: str = "gpt-4o-mini",
) -> dict:
    """
    Extract original contributions and referenced statements from paper text.

    Parameters
    ----------
    client : openai.OpenAI
    text : str
        Raw text (will be truncated to 15 000 chars).
    model_name : str

    Returns
    -------
    dict
        Keys: "original_contributions", "referenced_statements".
        Falls back to {"error": ..., "raw": ...} on parse failure.
    """
    prompt = _STATEMENTS_PROMPT.format(text=text[:15000])
    response = client.responses.create(
        model=model_name,
        input=[{"role": "user", "content": prompt}],
        stream=False,
    )
    parsed = _safe_parse_json(response.output_text)
    if parsed is None:
        return {"error": "JSON parse failed", "raw": response.output_text}
    return parsed


def extract_data_methods(
    client,
    text: str,
    model_name: str = "gpt-4o-mini",
) -> dict:
    """
    Extract datasets, methods, and evaluation metrics from paper text.

    Parameters
    ----------
    client : openai.OpenAI
    text : str
        Raw text (will be truncated to 15 000 chars).
    model_name : str

    Returns
    -------
    dict
        Keys: "datasets", "methods", "evaluation_metrics".
        Falls back to {"error": ..., "raw": ...} on parse failure.
    """
    prompt = _DATA_METHOD_PROMPT.format(text=text[:15000])
    response = client.responses.create(
        model=model_name,
        input=[{"role": "user", "content": prompt}],
        stream=False,
    )
    parsed = _safe_parse_json(response.output_text)
    if parsed is None:
        return {"error": "JSON parse failed", "raw": response.output_text}
    return parsed


def extract_all(
    client,
    text: str,
    model_name: str = "gpt-4o-mini",
) -> dict:
    """
    Run all three extractions and return a combined result dict.

    Returns
    -------
    dict
        Keys: "ris", "statements", "data_methods".
    """
    return {
        "ris": extract_ris_metadata(client, text, model_name),
        "statements": extract_statements(client, text, model_name),
        "data_methods": extract_data_methods(client, text, model_name),
    }


# ---------------------------------------------------------------------------
# RIS text formatter
# ---------------------------------------------------------------------------

def format_ris(ris: dict, doc_id: str = "") -> str:
    """
    Convert an extracted RIS metadata dict to standard RIS text format.

    Parameters
    ----------
    ris : dict
        Dict returned by extract_ris_metadata().
    doc_id : str
        Optional identifier added as an ID field.

    Returns
    -------
    str
        RIS-formatted string.
    """
    lines = []
    ty = ris.get("type") or "JOUR"
    lines.append(f"TY  - {ty}")
    if doc_id:
        lines.append(f"ID  - {doc_id}")
    for author in ris.get("authors") or []:
        lines.append(f"AU  - {author}")
    if ris.get("title"):
        lines.append(f"TI  - {ris['title']}")
    if ris.get("year"):
        lines.append(f"PY  - {ris['year']}")
    if ris.get("journal"):
        lines.append(f"JO  - {ris['journal']}")
    if ris.get("doi"):
        lines.append(f"DO  - {ris['doi']}")
    if ris.get("abstract"):
        lines.append(f"AB  - {ris['abstract']}")
    for kw in ris.get("keywords") or []:
        lines.append(f"KW  - {kw}")
    lines.append("ER  -")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Batch job builders (large-scale processing via Batch API)
# ---------------------------------------------------------------------------

def build_ris_batch_job(
    custom_id: str,
    text: str,
    execution_config=None,
) -> dict:
    """
    Build a Batch API job dict for RIS extraction.

    Parameters
    ----------
    custom_id : str
    text : str
    execution_config : dict or ExecutionConfig, optional

    Returns
    -------
    dict
        Ready to be JSON-serialised into a JSONL batch file.
    """
    from tiliqua.llm_runtime import build_responses_batch_job
    prompt = _RIS_PROMPT.format(text=text[:15000])
    messages = [{"role": "user", "content": prompt}]
    return build_responses_batch_job(custom_id, messages, execution_config)


def build_statements_batch_job(
    custom_id: str,
    text: str,
    execution_config=None,
) -> dict:
    """Build a Batch API job dict for statement extraction."""
    from tiliqua.llm_runtime import build_responses_batch_job
    prompt = _STATEMENTS_PROMPT.format(text=text[:15000])
    messages = [{"role": "user", "content": prompt}]
    return build_responses_batch_job(custom_id, messages, execution_config)


def build_data_methods_batch_job(
    custom_id: str,
    text: str,
    execution_config=None,
) -> dict:
    """Build a Batch API job dict for data/method extraction."""
    from tiliqua.llm_runtime import build_responses_batch_job
    prompt = _DATA_METHOD_PROMPT.format(text=text[:15000])
    messages = [{"role": "user", "content": prompt}]
    return build_responses_batch_job(custom_id, messages, execution_config)
