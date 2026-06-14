"""
rag_layer.py  –  Vector Database (Pinecone) + Embedding layer for the Blog Writing Agent.

Responsibilities:
  - Embed text using sentence-transformers (local, no extra API key needed).
  - Upsert research evidence + section drafts into Pinecone.
  - Retrieve semantically relevant context during the writing phase.
  - Attach source citations to retrieved chunks.

Environment variables:
  PINECONE_API_KEY       required for Pinecone
  PINECONE_INDEX         name of the Pinecone index (default: blog-rag)
  PINECONE_ENVIRONMENT   region (default: us-east-1)
  EMBED_MODEL            sentence-transformers model name (default: all-MiniLM-L6-v2)
"""

from __future__ import annotations

import hashlib
import os
import uuid
from typing import List, Optional

# ---------------------------------------------------------------------------
# Lazy imports – keep startup fast when RAG is disabled
# ---------------------------------------------------------------------------

_embedder = None
_pinecone_index = None

PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX", "blog-rag")
PINECONE_ENV = os.getenv("PINECONE_ENVIRONMENT", "us-east-1")
EMBED_MODEL = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")
VECTOR_SIZE = 384  # all-MiniLM-L6-v2 output dim; auto-updated on first embed


def _get_embedder():
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer  # type: ignore
        _embedder = SentenceTransformer(EMBED_MODEL)
    return _embedder


def _get_client():
    global _pinecone_index
    if _pinecone_index is not None:
        return _pinecone_index

    from pinecone import Pinecone, ServerlessSpec  # type: ignore

    api_key = os.getenv("PINECONE_API_KEY", "")
    if not api_key:
        raise RuntimeError("PINECONE_API_KEY not set")

    pc = Pinecone(api_key=api_key)

    existing = [i.name for i in pc.list_indexes()]
    if PINECONE_INDEX_NAME not in existing:
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=VECTOR_SIZE,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region=PINECONE_ENV),
        )

    _pinecone_index = pc.Index(PINECONE_INDEX_NAME)
    return _pinecone_index


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def embed(texts: List[str]) -> List[List[float]]:
    """Return L2-normalised embeddings for a list of texts."""
    model = _get_embedder()
    vecs = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return [v.tolist() for v in vecs]


def _stable_id(text: str) -> str:
    """Deterministic UUID from content hash – avoids duplicates on re-runs."""
    digest = hashlib.sha256(text.encode()).hexdigest()
    return str(uuid.UUID(digest[:32]))


# ---------------------------------------------------------------------------
# Upsert
# ---------------------------------------------------------------------------

def upsert_evidence(evidence_items: list, topic: str = "") -> int:
    """
    Embed and store EvidenceItem dicts (title + snippet + url + published_at).
    Returns number of items upserted.
    """
    if not evidence_items:
        return 0

    texts = []
    payloads = []
    for e in evidence_items:
        if hasattr(e, "model_dump"):
            e = e.model_dump()
        chunk = f"{e.get('title', '')}. {e.get('snippet', '')}".strip()
        if not chunk:
            continue
        texts.append(chunk)
        payloads.append(
            {
                "type": "evidence",
                "topic": topic,
                "title": e.get("title", ""),
                "url": e.get("url", ""),
                "published_at": e.get("published_at") or "",
                "source": e.get("source") or "",
                "snippet": e.get("snippet") or "",
                "chunk": chunk,
            }
        )

    if not texts:
        return 0

    vectors = embed(texts)
    index = _get_client()

    index.upsert(vectors=[
        {"id": _stable_id(t), "values": v, "metadata": p}
        for t, v, p in zip(texts, vectors, payloads)
    ])
    return len(texts)


def upsert_section(section_md: str, task_id: int, task_title: str, topic: str = "") -> int:
    """
    Chunk a written section and store in Pinecone for future retrieval.
    Returns number of chunks stored.
    """
    chunks = _chunk_text(section_md, max_tokens=300)
    if not chunks:
        return 0

    vectors = embed(chunks)
    index = _get_client()

    index.upsert(vectors=[
        {"id": _stable_id(f"section:{task_id}:{i}:{c}"), "values": v, "metadata": {
            "type": "section",
            "topic": topic,
            "task_id": task_id,
            "task_title": task_title,
            "chunk": c,
        }}
        for i, (c, v) in enumerate(zip(chunks, vectors))
    ])
    return len(chunks)


# ---------------------------------------------------------------------------
# Retrieve
# ---------------------------------------------------------------------------

def retrieve(
    query: str,
    top_k: int = 6,
    filter_type: Optional[str] = None,  # "evidence" | "section" | None (all)
    topic: Optional[str] = None,
) -> List[dict]:
    """
    Semantic search in Pinecone.
    Returns list of payload dicts, sorted by score desc.
    Adds 'score' field to each payload.
    """
    index = _get_client()
    query_vec = embed([query])[0]

    filter_dict = {}
    if filter_type:
        filter_dict["type"] = {"$eq": filter_type}
    if topic:
        filter_dict["topic"] = {"$eq": topic}

    results = index.query(
        vector=query_vec,
        top_k=top_k,
        filter=filter_dict or None,
        include_metadata=True,
    )

    hits = []
    for r in results.matches:
        payload = dict(r.metadata or {})
        payload["score"] = round(r.score, 4)
        hits.append(payload)
    return hits


def retrieve_for_section(task_title: str, bullets: List[str], topic: str, top_k: int = 8) -> str:
    """
    Build a formatted RAG context block for a worker section.
    Combines evidence + previously written sections.
    """
    query = f"{topic}: {task_title}. " + " ".join(bullets[:3])
    hits = retrieve(query, top_k=top_k, topic=topic)

    evidence_hits = [h for h in hits if h.get("type") == "evidence"]
    section_hits = [h for h in hits if h.get("type") == "section"]

    lines = []
    if evidence_hits:
        lines.append("### Retrieved Evidence (RAG)")
        for h in evidence_hits:
            url = h.get("url", "")
            title = h.get("title", "")
            snippet = h.get("snippet") or h.get("chunk", "")[:200]
            pub = h.get("published_at", "")
            score = h.get("score", "")
            lines.append(f"- **{title}** ({pub}) [score={score}]\n  {snippet}\n  URL: {url}")

    if section_hits:
        lines.append("\n### Related Previously Written Content (RAG)")
        for h in section_hits:
            sec_title = h.get("task_title", "")
            chunk = h.get("chunk", "")[:300]
            lines.append(f"- *{sec_title}*: {chunk}")

    return "\n".join(lines)


def format_citations(hits: List[dict]) -> str:
    """
    Generate a markdown citations block from evidence hits.
    """
    seen_urls = set()
    lines = ["\n\n---\n**Sources**\n"]
    for i, h in enumerate(hits, 1):
        url = h.get("url", "")
        if not url or url in seen_urls:
            continue
        seen_urls.add(url)
        title = h.get("title") or url
        pub = h.get("published_at", "")
        pub_str = f" ({pub})" if pub else ""
        lines.append(f"{i}. [{title}]({url}){pub_str}")
    return "\n".join(lines) if len(lines) > 1 else ""


# ---------------------------------------------------------------------------
# Text chunking
# ---------------------------------------------------------------------------

def _chunk_text(text: str, max_tokens: int = 300) -> List[str]:
    """
    Simple paragraph-aware chunker.
    Splits on double newlines; merges short paragraphs; caps at ~max_tokens words.
    """
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: List[str] = []
    current: List[str] = []
    current_len = 0

    for para in paragraphs:
        words = len(para.split())
        if current_len + words > max_tokens and current:
            chunks.append("\n\n".join(current))
            current = []
            current_len = 0
        current.append(para)
        current_len += words

    if current:
        chunks.append("\n\n".join(current))

    return chunks


# ---------------------------------------------------------------------------
# Collection management
# ---------------------------------------------------------------------------

def clear_collection(topic: Optional[str] = None):
    """Clear all (or topic-specific) vectors from the index."""
    index = _get_client()
    if topic:
        index.delete(filter={"topic": {"$eq": topic}})
    else:
        index.delete(delete_all=True)


def collection_stats() -> dict:
    """Return basic index statistics."""
    try:
        index = _get_client()
        stats = index.describe_index_stats()
        return {
            "vectors_count": stats.total_vector_count,
            "namespaces": list(stats.namespaces.keys()),
        }
    except Exception as e:
        return {"error": str(e)}