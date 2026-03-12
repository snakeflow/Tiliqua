"""
RAG (Retrieval-Augmented Generation) knowledge store for Tiliqua.

Builds an in-memory vector store from extracted knowledge objects using
OpenAI embeddings, then answers free-text questions by retrieving the
most relevant chunks and passing them to the Responses API.

No external vector database required — similarity search uses numpy.
"""

from __future__ import annotations
import json
from typing import Optional


# ---------------------------------------------------------------------------
# Embedding helpers
# ---------------------------------------------------------------------------

def _embed_texts(client, texts: list[str], model: str = "text-embedding-3-small") -> list:
    """
    Embed a list of strings via the OpenAI Embeddings API.

    Returns a list of float vectors (one per input string).
    """
    import numpy as np

    if not texts:
        return []

    response = client.embeddings.create(model=model, input=texts)
    return [np.array(item.embedding, dtype="float32") for item in response.data]


def _cosine_similarity(a, b):
    """Cosine similarity between two numpy vectors."""
    import numpy as np

    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


# ---------------------------------------------------------------------------
# Store construction
# ---------------------------------------------------------------------------

def _knowledge_to_chunks(knowledge: dict) -> list[dict]:
    """
    Flatten an extracted knowledge dict (from extraction.extract_all) into
    a list of {text, source, metadata} dicts ready for embedding.
    """
    chunks = []

    # RIS metadata → single chunk
    ris = knowledge.get("ris", {})
    if ris and "error" not in ris:
        parts = []
        if ris.get("title"):
            parts.append(f"Title: {ris['title']}")
        if ris.get("authors"):
            parts.append(f"Authors: {', '.join(ris['authors'])}")
        if ris.get("year"):
            parts.append(f"Year: {ris['year']}")
        if ris.get("journal"):
            parts.append(f"Journal: {ris['journal']}")
        if ris.get("abstract"):
            parts.append(f"Abstract: {ris['abstract']}")
        if ris.get("keywords"):
            parts.append(f"Keywords: {', '.join(ris['keywords'])}")
        if parts:
            chunks.append({
                "text": "\n".join(parts),
                "source": "ris_metadata",
                "metadata": {"type": "bibliographic"},
            })

    # Original contributions
    stmts = knowledge.get("statements", {})
    for item in stmts.get("original_contributions", []):
        text = item.get("text", "").strip()
        if text:
            chunks.append({
                "text": text,
                "source": "original_contribution",
                "metadata": {"context": item.get("context", "")},
            })

    # Referenced statements
    for item in stmts.get("referenced_statements", []):
        text = item.get("text", "").strip()
        if text:
            chunks.append({
                "text": text,
                "source": "referenced_statement",
                "metadata": {
                    "citation": item.get("citation", ""),
                    "context": item.get("context", ""),
                },
            })

    # Datasets
    dm = knowledge.get("data_methods", {})
    for ds in dm.get("datasets", []):
        name = ds.get("name", "")
        desc = ds.get("description", "")
        if name or desc:
            chunks.append({
                "text": f"Dataset: {name}. {desc}".strip(". "),
                "source": "dataset",
                "metadata": {"name": name, "source": ds.get("source", "")},
            })

    # Methods
    for m in dm.get("methods", []):
        name = m.get("name", "")
        desc = m.get("description", "")
        if name or desc:
            chunks.append({
                "text": f"Method: {name} ({m.get('category', '')}). {desc}".strip(". "),
                "source": "method",
                "metadata": {"name": name, "category": m.get("category", "")},
            })

    # Evaluation metrics → single chunk
    metrics = dm.get("evaluation_metrics", [])
    if metrics:
        chunks.append({
            "text": f"Evaluation metrics used: {', '.join(metrics)}",
            "source": "evaluation_metrics",
            "metadata": {},
        })

    return chunks


def build_rag_store(
    client,
    knowledge: dict,
    embedding_model: str = "text-embedding-3-small",
) -> list[dict]:
    """
    Build an in-memory RAG store from an extracted knowledge dict.

    Parameters
    ----------
    client : openai.OpenAI
    knowledge : dict
        Output of extraction.extract_all().
    embedding_model : str

    Returns
    -------
    list[dict]
        Each entry has keys: text, source, metadata, embedding (numpy array).
    """
    chunks = _knowledge_to_chunks(knowledge)
    if not chunks:
        print("⚠️  No knowledge chunks found. Store is empty.")
        return []

    texts = [c["text"] for c in chunks]
    embeddings = _embed_texts(client, texts, model=embedding_model)

    store = []
    for chunk, emb in zip(chunks, embeddings):
        entry = dict(chunk)
        entry["embedding"] = emb
        store.append(entry)

    print(f"RAG store built: {len(store)} chunks embedded.")
    return store


def build_rag_store_from_texts(
    client,
    texts: list[str],
    sources: Optional[list[str]] = None,
    embedding_model: str = "text-embedding-3-small",
) -> list[dict]:
    """
    Build a RAG store directly from a list of text strings.

    Useful when you want to embed raw text chunks rather than structured knowledge.

    Parameters
    ----------
    client : openai.OpenAI
    texts : list[str]
    sources : list[str], optional
        Labels for each text (e.g. "chunk_1", "page_2").
    embedding_model : str

    Returns
    -------
    list[dict]
    """
    if sources is None:
        sources = [f"chunk_{i}" for i in range(len(texts))]

    embeddings = _embed_texts(client, texts, model=embedding_model)

    store = []
    for text, src, emb in zip(texts, sources, embeddings):
        store.append({
            "text": text,
            "source": src,
            "metadata": {},
            "embedding": emb,
        })

    print(f"RAG store built: {len(store)} chunks embedded.")
    return store


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

def retrieve(
    store: list[dict],
    client,
    query: str,
    top_k: int = 5,
    embedding_model: str = "text-embedding-3-small",
) -> list[dict]:
    """
    Retrieve the top-k most relevant store entries for a query.

    Parameters
    ----------
    store : list[dict]
        Output of build_rag_store() or build_rag_store_from_texts().
    client : openai.OpenAI
    query : str
    top_k : int
    embedding_model : str

    Returns
    -------
    list[dict]
        Top-k entries sorted by descending cosine similarity.
        Each entry also contains a "score" key.
    """
    if not store:
        return []

    query_emb = _embed_texts(client, [query], model=embedding_model)[0]

    scored = []
    for entry in store:
        score = _cosine_similarity(query_emb, entry["embedding"])
        item = {k: v for k, v in entry.items() if k != "embedding"}
        item["score"] = score
        scored.append(item)

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]


# ---------------------------------------------------------------------------
# RAG query
# ---------------------------------------------------------------------------

_RAG_SYSTEM_PROMPT = """\
You are a research assistant with access to extracted knowledge from an academic paper.
Answer the user's question using ONLY the provided context.
If the answer is not in the context, say so clearly.
Cite the source type (e.g. "original contribution", "method", "dataset") when relevant.
"""


def query_rag(
    store: list[dict],
    client,
    question: str,
    top_k: int = 5,
    model_name: str = "gpt-4o-mini",
    embedding_model: str = "text-embedding-3-small",
) -> dict:
    """
    Answer a question using the RAG store.

    Retrieves top-k relevant chunks, then calls the Responses API to
    generate an answer grounded in the retrieved context.

    Parameters
    ----------
    store : list[dict]
        Built by build_rag_store().
    client : openai.OpenAI
    question : str
    top_k : int
    model_name : str
    embedding_model : str

    Returns
    -------
    dict
        Keys: "answer" (str), "sources" (list of retrieved chunk dicts).
    """
    retrieved = retrieve(store, client, question, top_k=top_k, embedding_model=embedding_model)

    if not retrieved:
        return {"answer": "No knowledge in store.", "sources": []}

    context_blocks = []
    for i, item in enumerate(retrieved, 1):
        context_blocks.append(
            f"[{i}] (source: {item['source']}, score: {item['score']:.3f})\n{item['text']}"
        )
    context = "\n\n".join(context_blocks)

    user_prompt = f"""Context from the paper:

{context}

---
Question: {question}
"""

    response = client.responses.create(
        model=model_name,
        input=[
            {"role": "system", "content": _RAG_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        stream=False,
    )

    return {
        "answer": response.output_text,
        "sources": retrieved,
    }
