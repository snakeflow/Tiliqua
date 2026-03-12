"""
Tiliqua - LLM Document Mining Framework (Responses API + Batch + OCR Pipeline)
"""

from tiliqua.llm_runtime import (
    default_execution_config,
    build_llm_body,
    build_batch_job,
    build_responses_body,
    build_responses_batch_job,
    save_jsonl,
    make_clients,
    MultiClient,
)

from tiliqua.batch_runtime import (
    submit_batch,
    retrieve_batch_result,
    download_batch_output,
    submit_batches_multi_client,
)

from tiliqua.text_utils import (
    chunk_text_with_overlap,
    segment_document,
    extract_text_from_batch_line,
    extract_text_from_responses_batch_line,
    extract_custom_id,
)

from tiliqua.pdf_reader import (
    read_pdf_pymupdf,
    upload_pdf_to_openai,
    OCR_openai_single_page,
    OCR_pdf_pages,
    read_pdf,
    convert_all_pdfs_to_text,
)

from tiliqua.extraction import (
    extract_ris_metadata,
    extract_statements,
    extract_data_methods,
    extract_all,
    format_ris,
    build_ris_batch_job,
    build_statements_batch_job,
    build_data_methods_batch_job,
)

from tiliqua.rag import (
    build_rag_store,
    build_rag_store_from_texts,
    retrieve,
    query_rag,
)

from tiliqua.models import (
    Document,
    Segment,
    Task,
    ExecutionConfig,
    Statement,
    RISRecord,
    CitationEdge,
    TaskType,
    StatementType,
)

__version__ = "0.2.0"

__all__ = [
    # llm_runtime
    "default_execution_config",
    "build_llm_body",
    "build_batch_job",
    "build_responses_body",
    "build_responses_batch_job",
    "save_jsonl",
    "make_clients",
    "MultiClient",
    # batch_runtime
    "submit_batch",
    "retrieve_batch_result",
    "download_batch_output",
    "submit_batches_multi_client",
    # text_utils
    "chunk_text_with_overlap",
    "segment_document",
    "extract_text_from_batch_line",
    "extract_text_from_responses_batch_line",
    "extract_custom_id",
    # pdf_reader
    "read_pdf_pymupdf",
    "upload_pdf_to_openai",
    "OCR_openai_single_page",
    "OCR_pdf_pages",
    "read_pdf",
    "convert_all_pdfs_to_text",
    # extraction
    "extract_ris_metadata",
    "extract_statements",
    "extract_data_methods",
    "extract_all",
    "format_ris",
    "build_ris_batch_job",
    "build_statements_batch_job",
    "build_data_methods_batch_job",
    # rag
    "build_rag_store",
    "build_rag_store_from_texts",
    "retrieve",
    "query_rag",
    # models
    "Document",
    "Segment",
    "Task",
    "ExecutionConfig",
    "Statement",
    "RISRecord",
    "CitationEdge",
    "TaskType",
    "StatementType",
]
