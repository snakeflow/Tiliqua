"""
PDF reading utilities for Tiliqua.

Primary extraction uses PyMuPDF; falls back to OpenAI OCR (Responses API)
when the extracted text is too short or unreadable.
"""

from __future__ import annotations
import os
from typing import Optional


# ---------------------------------------------------------------------------
# OCR prompt (kept here so callers can inspect / override it)
# ---------------------------------------------------------------------------

def _make_ocr_prompt(page_number: Optional[int]) -> str:
    page_label = str(page_number) if page_number is not None else "?"
    return f"""You are a STRICT OCR transcription engine.

Your task is to transcribe the document EXACTLY as written.

CRITICAL RULES:
1. DO NOT summarise.
2. DO NOT paraphrase.
3. DO NOT rewrite sentences.
4. Preserve ALL numbers exactly.
5. Preserve punctuation exactly.
6. Preserve section headings.
7. Preserve line order as much as possible.

PAGE FORMAT:

=== PAGE {page_label} ===

[TEXT]
Transcribe the text exactly.

[TABLE X]
If a table appears, reproduce it using Markdown table format.

[Figure X. Title]
Description: short description of the figure, map, or diagram.

If the figure title is not visible:
[Figure]
Description: ...

Return only the transcription.
"""


# ---------------------------------------------------------------------------
# PyMuPDF extraction
# ---------------------------------------------------------------------------

def read_pdf_pymupdf(pdf_pathfile: str) -> str:
    """
    Extract text from a PDF using PyMuPDF (fast, digital PDFs).

    Parameters
    ----------
    pdf_pathfile : str
        Path to the PDF file.

    Returns
    -------
    str
        Concatenated text from all pages.
    """
    import fitz  # PyMuPDF

    doc = fitz.open(pdf_pathfile)
    text = ""
    for page in doc:
        page_text = page.get_text()
        if page_text:
            text += page_text
    return text


# ---------------------------------------------------------------------------
# OpenAI OCR fallback (Responses API)
# ---------------------------------------------------------------------------

def upload_pdf_to_openai(client, pdf_pathfile: str) -> str:
    """
    Upload a PDF to OpenAI Files API for use with the Responses API.

    Parameters
    ----------
    client : openai.OpenAI
    pdf_pathfile : str

    Returns
    -------
    str
        The file_id assigned by OpenAI.
    """
    with open(pdf_pathfile, "rb") as f:
        file_obj = client.files.create(file=f, purpose="user_data")
    return file_obj.id


def OCR_openai_single_page(
    client,
    file_id: str,
    page_number: Optional[int] = None,
    model_name: str = "gpt-4o-mini",
) -> tuple:
    """
    OCR a single page of a previously uploaded PDF via the Responses API.

    Parameters
    ----------
    client : openai.OpenAI
    file_id : str
        File ID returned by upload_pdf_to_openai().
    page_number : int, optional
        Page number label used in the OCR prompt.
    model_name : str

    Returns
    -------
    tuple[response, str]
        The raw Responses API response object and the extracted text.
    """
    prompt = _make_ocr_prompt(page_number)

    message = [
        {
            "role": "user",
            "content": [
                {"type": "input_file", "file_id": file_id},
                {"type": "input_text", "text": prompt},
            ],
        }
    ]

    response = client.responses.create(
        model=model_name,
        input=message,
        stream=False,
    )

    return response, response.output_text


def OCR_pdf_pages(
    client,
    pdf_pathfile: str,
    model_name: str = "gpt-4o-mini",
) -> str:
    """
    OCR every page of a PDF using the OpenAI Responses API.

    Uploads the file once, then processes page by page.

    Parameters
    ----------
    client : openai.OpenAI
    pdf_pathfile : str
    model_name : str

    Returns
    -------
    str
        Full OCR text, pages separated by blank lines.
    """
    import fitz  # PyMuPDF (used only for page count)

    reader = fitz.open(pdf_pathfile)
    file_id = upload_pdf_to_openai(client, pdf_pathfile)

    pages_text = []
    for i, _page in enumerate(reader):
        try:
            _response, page_text = OCR_openai_single_page(
                client,
                file_id,
                page_number=i + 1,
                model_name=model_name,
            )
            pages_text.append(page_text)
        except Exception:
            pages_text.append(f"\n=== PAGE {i + 1} ===\n[OCR FAILED]\n")

    return "\n\n".join(pages_text)


# ---------------------------------------------------------------------------
# Auto-switching reader
# ---------------------------------------------------------------------------

def read_pdf(
    client,
    pdf_pathfile: str,
    maxlen: Optional[int] = None,
    model_name: str = "gpt-4o-mini",
    OCR_only: bool = False,
) -> str:
    """
    Read a PDF, auto-selecting PyMuPDF or OpenAI OCR fallback.

    PyMuPDF is tried first unless OCR_only=True.  If the result is fewer
    than 1 000 characters the system switches to OpenAI OCR automatically.

    Parameters
    ----------
    client : openai.OpenAI
        Needed for the OCR path; may be None if OCR is never triggered.
    pdf_pathfile : str
    maxlen : int, optional
        Truncate PyMuPDF output to this character count.
    model_name : str
        Model used for OCR.
    OCR_only : bool
        Skip PyMuPDF entirely and go straight to OCR.

    Returns
    -------
    str
    """
    try:
        if not OCR_only:
            text = read_pdf_pymupdf(pdf_pathfile)
            if maxlen and len(text) > maxlen:
                text = text[:maxlen]
            if len(text) > 1000:
                return text
        print("⚠️  PyMuPDF text too short, switching to OpenAI OCR")
        return OCR_pdf_pages(client, pdf_pathfile, model_name)
    except Exception as exc:
        print("⚠️  Error reading PDF, fallback to OCR:", exc)
        return OCR_pdf_pages(client, pdf_pathfile, model_name)


# ---------------------------------------------------------------------------
# Bulk conversion
# ---------------------------------------------------------------------------

def convert_all_pdfs_to_text(
    pdf_folder: str,
    output_csv: str,
    client=None,
    clients=None,
    model_name: str = "gpt-4o-mini",
    max_workers: Optional[int] = None,
):
    """
    Convert every PDF in a directory to text and save as a CSV table.

    Output columns: FID, filename, raw_text.

    Pass ``clients`` (a list of OpenAI clients or a ``MultiClient``) to process
    PDFs in parallel, distributing requests across API keys. Each worker picks
    the next available client in round-robin order.

    Parameters
    ----------
    pdf_folder : str
    output_csv : str
        Path for the output CSV.
    client : openai.OpenAI, optional
        Single client. Used when ``clients`` is not provided.
    clients : list[openai.OpenAI] or MultiClient, optional
        Pool of clients for parallel processing. When provided, PDFs are
        processed concurrently using up to ``max_workers`` threads.
    model_name : str
    max_workers : int, optional
        Number of parallel workers. Defaults to ``len(clients)`` when
        ``clients`` is given, otherwise 1 (sequential).

    Returns
    -------
    pandas.DataFrame
    """
    import pandas as pd
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from tiliqua.llm_runtime import MultiClient

    pdf_files = sorted(
        f for f in os.listdir(pdf_folder) if f.lower().endswith(".pdf")
    )

    if clients is not None:
        mc = clients if isinstance(clients, MultiClient) else MultiClient(clients)
        n_workers = max_workers or len(mc)
    else:
        mc = None
        n_workers = 1

    def _process(args):
        i, fname = args
        fpath = os.path.join(pdf_folder, fname)
        _client = mc.next() if mc is not None else client
        print(f"  Processing {fname} …")
        text = read_pdf(_client, fpath, model_name=model_name)
        return i, {"FID": f"F{i:05d}", "filename": fname, "raw_text": text}

    rows = [None] * len(pdf_files)
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        futures = {ex.submit(_process, (i, fname)): i for i, fname in enumerate(pdf_files)}
        for future in as_completed(futures):
            i, row = future.result()
            rows[i] = row

    df = pd.DataFrame([r for r in rows if r is not None])
    df.to_csv(output_csv, index=False, encoding="utf-8")
    print(f"Saved: {output_csv}  ({len(df)} documents)")
    return df
