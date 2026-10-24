import time
from typing import Optional, List, Tuple, Dict, Any
from chunk_vector import load_vector_database
from langchain_core.documents import Document


def retrieve_context(
    question: str,
    k: int = 5,
    source_file: Optional[str] = None,
    file_type: Optional[str] = None
) -> Tuple[str, List[Document], Dict[str, Any]]:
    """Retrieve top-k relevant document chunks for a question with optional metadata filtering.

    Args:
        question: Query string to search.
        k: Number of relevant chunks to return (default 5).
        source_file: Optional exact filename filter (e.g. 'nlp_technical_reference_guide.md').
        file_type: Optional file type filter (e.g. 'pdf', 'html', 'md').

    Returns:
        Tuple of:
          - formatted context string with citation headers
          - list of retrieved Document objects
          - retrieval statistics dictionary (latency, counts, filter info)
    """
    vector_db = load_vector_database()

    start_time = time.perf_counter()

    # Retrieve an oversampled candidate pool to guarantee k results after metadata filtering
    fetch_multiplier = 4 if (source_file or file_type) else 2
    candidate_k = max(k * fetch_multiplier, 15)

    raw_candidates = vector_db.similarity_search_with_score(question, k=candidate_k)

    documents: List[Document] = []
    seen_chunk_ids = set()
    similarity_scores = []

    for doc, score in raw_candidates:
        content = doc.page_content.strip()
        metadata = doc.metadata or {}

        if not content:
            continue

        # Metadata filters
        if source_file and metadata.get("source_file") != source_file:
            continue

        if file_type and metadata.get("file_type", "").lower() != file_type.lower():
            continue

        # Deduplication by deterministic chunk_id or content
        chunk_id = metadata.get("chunk_id", content)
        if chunk_id in seen_chunk_ids:
            continue

        seen_chunk_ids.add(chunk_id)
        # Store FAISS L2/cosine distance score in metadata
        doc.metadata["score"] = float(score)
        similarity_scores.append(float(score))
        documents.append(doc)

        if len(documents) == k:
            break

    retrieval_time_ms = (time.perf_counter() - start_time) * 1000

    # Build grounded context string with structured source citations
    context_parts = []
    for i, doc in enumerate(documents, start=1):
        source = doc.metadata.get("source_file", "Unknown")
        page = doc.metadata.get("page")
        chunk_idx = doc.metadata.get("chunk_index", 0)

        location_str = f"page {int(page) + 1}" if page is not None and int(page) > 0 else f"section/chunk {chunk_idx + 1}"
        citation_header = f"[Source {i}: {source}, {location_str}]"
        context_parts.append(f"{citation_header}\n{doc.page_content.strip()}")

    context = "\n\n".join(context_parts)

    stats = {
        "retrieval_latency_ms": round(retrieval_time_ms, 2),
        "candidate_count": len(raw_candidates),
        "retrieved_chunk_count": len(documents),
        "similarity_scores": similarity_scores,
        "filter_applied": {
            "source_file": source_file,
            "file_type": file_type
        }
    }

    return context, documents, stats
