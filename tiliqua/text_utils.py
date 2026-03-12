"""
Text utilities for Tiliqua.

Provides chunking, segmentation, and response-extraction helpers.
"""

from __future__ import annotations
from typing import Optional


def chunk_text_with_overlap(
    text: str,
    chunk_size: int = 800,
    overlap: int = 100,
) -> list[str]:
    """
    Split text into overlapping chunks by word count.

    Parameters
    ----------
    text : str
        Input text (plain, not tokenised).
    chunk_size : int
        Target number of words per chunk.
    overlap : int
        Number of words shared between consecutive chunks.

    Returns
    -------
    list[str]
        List of text chunks.
    """
    if not text or not text.strip():
        return []

    words = text.split()
    if not words:
        return []

    stride = max(1, chunk_size - overlap)
    chunks = []
    start = 0

    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start += stride

    return chunks


def segment_document(
    doc_id: str,
    text: str,
    chunk_size: int = 800,
    overlap: int = 100,
    page: Optional[int] = None,
) -> list[dict]:
    """
    Segment a document's raw text into a list of segment dicts.

    Returns plain dicts (not Segment dataclasses) for maximum compatibility
    with pandas-based workflows.

    Parameters
    ----------
    doc_id : str
        The document's FID or equivalent identifier.
    text : str
        Full document text.
    chunk_size : int
    overlap : int
    page : int, optional
        Page number to attach to all segments (use when processing page-by-page).

    Returns
    -------
    list[dict]
        Each dict has keys: doc_id, segment_id, text, page.
    """
    chunks = chunk_text_with_overlap(text, chunk_size=chunk_size, overlap=overlap)
    return [
        {"doc_id": doc_id, "segment_id": i, "text": chunk, "page": page}
        for i, chunk in enumerate(chunks)
    ]


def extract_text_from_batch_line(line: dict) -> Optional[str]:
    """
    Extract the assistant's text reply from a parsed Batch API output line.

    Parameters
    ----------
    line : dict
        A single parsed dict from download_batch_output().

    Returns
    -------
    str or None
        The assistant message content, or None if the request failed.
    """
    try:
        choices = line["response"]["body"]["choices"]
        return choices[0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        return None


def extract_custom_id(line: dict) -> Optional[str]:
    """Return the custom_id from a batch output line."""
    return line.get("custom_id")


def extract_text_from_responses_batch_line(line: dict) -> Optional[str]:
    """
    Extract the assistant's text from a Responses API batch output line.

    The Responses API batch output format differs from Chat Completions:
    - response.body.output_text  (convenience field, most reliable)
    - response.body.output[].content[].text  (fallback)

    Parameters
    ----------
    line : dict
        A single parsed dict from download_batch_output() where the batch
        used the /v1/responses endpoint.

    Returns
    -------
    str or None
    """
    try:
        body = line["response"]["body"]
        # Convenience field present in most Responses API replies
        if "output_text" in body:
            return body["output_text"]
        # Walk the structured output array
        for item in body.get("output", []):
            if item.get("type") == "message":
                for content_block in item.get("content", []):
                    if content_block.get("type") == "output_text":
                        return content_block.get("text")
        return None
    except (KeyError, IndexError, TypeError):
        return None
