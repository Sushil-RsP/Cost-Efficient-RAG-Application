import logging
import os
import sys
import time
from typing import Optional, List
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from main import answer_question
from chunk_vector import build_vector_database, get_manifest

load_dotenv()

# Structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("rag_api")

app = FastAPI(
    title="Cost-Efficient RAG QA API",
    description="A production-grade, low-cost RAG QA Service backed by FAISS and Gemini.",
    version="1.0.0"
)

# Enable CORS for frontend clients / dashboards
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    question: str = Field(..., description="The user's natural language question", example="What is scaled dot-product attention?")
    k: int = Field(5, description="Number of top chunks to retrieve", ge=1, le=20)
    source_file: Optional[str] = Field(None, description="Optional filename metadata filter", example="nlp_technical_reference_guide.md")
    file_type: Optional[str] = Field(None, description="Optional file type metadata filter", example="md")


class SourceItem(BaseModel):
    citation: str
    source_file: str
    page: Optional[int]
    chunk_id: str
    file_type: str


class RetrievalStats(BaseModel):
    retrieval_latency_ms: float
    candidate_count: int
    retrieved_chunk_count: int
    similarity_scores: List[float]
    filter_applied: dict


class GenerationStats(BaseModel):
    generation_latency_ms: float
    input_tokens: int
    output_tokens: int
    total_tokens: int


class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: List[SourceItem]
    retrieval_stats: RetrievalStats
    generation_stats: GenerationStats
    total_latency_ms: float


class IngestResponse(BaseModel):
    status: str
    message: str
    vector_count: int
    manifest: dict


@app.get("/health", tags=["System"])
def health_check():
    """Health check endpoint exposing store metadata, vector count, and model configuration."""
    manifest = get_manifest()
    return {
        "status": "healthy",
        "service": "Cost-Efficient RAG QA Service",
        "vector_store": manifest.get("store_type", "FAISS"),
        "vector_count": manifest.get("vector_count", 0),
        "embedding_model": manifest.get("embedding_model"),
        "embedding_dimension": manifest.get("embedding_dimension"),
        "environment": os.getenv("ENVIRONMENT", "production")
    }


@app.get("/manifest", tags=["System"])
def get_index_manifest():
    """Retrieve full vectorstore manifest including ingested document summary."""
    return get_manifest()


@app.post("/query", response_model=QueryResponse, tags=["RAG QA"])
def query_rag(request: QueryRequest):
    """Query the RAG service to retrieve context chunks and generate a grounded, cited answer."""
    if not request.question.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Question string cannot be empty."
        )

    try:
        response = answer_question(
            question=request.question,
            k=request.k,
            source_file=request.source_file,
            file_type=request.file_type
        )
        return response
    except Exception as e:
        logger.error(f"Error processing query '{request.question}': {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal RAG processing error: {str(e)}"
        )


@app.post("/ingest", response_model=IngestResponse, tags=["Data Ingestion"])
def trigger_ingest(force_rebuild: bool = True):
    """Trigger idempotent corpus re-ingestion and index update."""
    try:
        logger.info("Triggering corpus ingestion...")
        db, manifest = build_vector_database(force_rebuild=force_rebuild)
        return {
            "status": "success",
            "message": f"Successfully ingested {manifest.get('vector_count', 0)} chunks idempotently.",
            "vector_count": manifest.get("vector_count", 0),
            "manifest": manifest
        }
    except Exception as e:
        logger.error(f"Ingestion failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ingestion failed: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)
