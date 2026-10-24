import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from dotenv import load_dotenv

from load import load_material

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
VECTOR_DIR = BASE_DIR / "vectorstore"
MANIFEST_PATH = VECTOR_DIR / "index_manifest.json"

os.environ.setdefault("HF_HOME", str(BASE_DIR.parent / "models"))

# Default chunking parameters (can be overridden via environment variables)
DEFAULT_CHUNK_SIZE = 800
DEFAULT_CHUNK_OVERLAP = 120
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", str(DEFAULT_CHUNK_SIZE)))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", str(DEFAULT_CHUNK_OVERLAP)))

# Embedding Model & Dimensionality Configuration
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
EMBEDDING_DIMENSION = 384

embeddings = None


def get_embeddings():
    """Singleton getter for HuggingFace embeddings."""
    global embeddings
    if embeddings is None:
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME
        )
    return embeddings


def compute_chunk_id(doc):
    """Compute a deterministic hash ID for a chunk based on source, page, and text content.
    Ensures idempotent ingestion without duplicate vectors.
    """
    source = doc.metadata.get("source_file", "unknown")
    page = doc.metadata.get("page", 0)
    chunk_index = doc.metadata.get("chunk_index", 0)
    raw_content = doc.page_content.strip()

    hasher = hashlib.sha256()
    hasher.update(f"{source}:{page}:{chunk_index}:{raw_content}".encode("utf-8"))
    return hasher.hexdigest()[:16]


def chunk_documents(documents, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    """Split loaded documents into chunks with deterministic IDs and metadata."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    chunks = splitter.split_documents(documents)

    # Track chunk index per source file and assign deterministic IDs
    source_counts = {}
    for chunk in chunks:
        source = chunk.metadata.get("source_file", "unknown")
        idx = source_counts.get(source, 0)
        chunk.metadata["chunk_index"] = idx
        source_counts[source] = idx + 1

        chunk_id = compute_chunk_id(chunk)
        chunk.metadata["chunk_id"] = chunk_id

    return chunks


def save_manifest(vector_count, documents_summary):
    """Record model name, dimensionality, chunk settings, and statistics in an index manifest."""
    manifest = {
        "embedding_model": EMBEDDING_MODEL_NAME,
        "embedding_dimension": EMBEDDING_DIMENSION,
        "distance_metric": "cosine",
        "vector_count": vector_count,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "default_chunk_size": DEFAULT_CHUNK_SIZE,
        "default_chunk_overlap": DEFAULT_CHUNK_OVERLAP,
        "store_type": "FAISS (IndexFlatL2 / Normalized Cosine)",
        "documents": documents_summary,
        "updated_at": datetime.now(timezone.utc).isoformat()
    }
    VECTOR_DIR.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def build_vector_database(force_rebuild=True):
    """Build the FAISS vector database from scratch and write the index manifest."""
    documents = load_material()
    if not documents:
        raise ValueError("No documents found to ingest.")

    chunks = chunk_documents(documents)
    if not chunks:
        raise ValueError("No chunks were generated.")

    print(f"[chunk_vector] Loaded documents/pages: {len(documents)}")
    print(f"[chunk_vector] Created chunks: {len(chunks)}")

    # Deduplicate chunks based on deterministic chunk_id (idempotency guarantee)
    unique_chunks = []
    seen_ids = set()
    for chunk in chunks:
        cid = chunk.metadata.get("chunk_id")
        if cid not in seen_ids:
            seen_ids.add(cid)
            unique_chunks.append(chunk)

    print(f"[chunk_vector] Unique chunks after deduplication: {len(unique_chunks)}")

    emb = get_embeddings()
    vector_db = FAISS.from_documents(unique_chunks, emb)

    VECTOR_DIR.mkdir(parents=True, exist_ok=True)
    vector_db.save_local(str(VECTOR_DIR))

    # Summarize documents for the manifest
    summary = {}
    for chunk in unique_chunks:
        src = chunk.metadata.get("source_file", "unknown")
        summary[src] = summary.get(src, 0) + 1

    manifest = save_manifest(vector_db.index.ntotal, summary)

    print(f"[chunk_vector] Vectors indexed: {vector_db.index.ntotal}")
    print(f"[chunk_vector] Dimensionality: {EMBEDDING_DIMENSION}")
    print(f"[chunk_vector] Manifest saved at: {MANIFEST_PATH}")

    return vector_db, manifest


def load_vector_database():
    """Load the existing vector database, or build it if missing."""
    index_file = VECTOR_DIR / "index.faiss"
    metadata_file = VECTOR_DIR / "index.pkl"

    if not index_file.is_file() or not metadata_file.is_file():
        print("[chunk_vector] Vector database not found. Building it now...")
        vector_db, _ = build_vector_database()
        return vector_db

    return FAISS.load_local(
        str(VECTOR_DIR),
        get_embeddings(),
        allow_dangerous_deserialization=True
    )


def get_manifest():
    """Retrieve the vector store metadata manifest."""
    if MANIFEST_PATH.is_file():
        try:
            with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {
        "embedding_model": EMBEDDING_MODEL_NAME,
        "embedding_dimension": EMBEDDING_DIMENSION,
        "vector_count": "unknown",
        "store_type": "FAISS"
    }


if __name__ == "__main__":
    print("[chunk_vector] Running standalone vector database build...")
    db, manifest = build_vector_database()
    print("[chunk_vector] Build complete!")
