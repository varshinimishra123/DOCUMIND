import os
import shutil
import logging
import mimetypes

# Fix Windows registry mime-type mapping issues for static files
mimetypes.add_type("application/javascript", ".js")
mimetypes.add_type("text/css", ".css")
from typing import List, Optional
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Import modules from prior phases
from ingestion import ingest_and_chunk_pdf
from storage import DualIndexer
from retrieval import HybridRetriever
from generation import generate_grounded_answer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("DocuMind.API")

# Create directories for persistence
UPLOAD_DIR = "./uploads"
INDEX_DIR = "./docmind_index"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(INDEX_DIR, exist_ok=True)

app = FastAPI(
    title="DocuMind QA Microservice",
    description="Production-grade Hybrid RAG API with dual-index storage and grounded generation.",
    version="1.0.0"
)

# Configure CORS Middleware to allow requests from frontend frameworks
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust in production to allow specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files to serve the frontend client
app.mount("/static", StaticFiles(directory="./static"), name="static")

# Global cache for retriever
_retriever: Optional[HybridRetriever] = None


def get_retriever() -> HybridRetriever:
    """
    Dependency injection helper to lazily load and cache the HybridRetriever.
    Validates that indices have been constructed and exist on disk.
    """
    global _retriever
    # Ensure index exists before rehydration
    faiss_bin = os.path.join(INDEX_DIR, "index.faiss")
    bm25_file = os.path.join(INDEX_DIR, "bm25.pkl")
    
    if not os.path.exists(faiss_bin) or not os.path.exists(bm25_file):
        raise HTTPException(
            status_code=400,
            detail="Retrieval index not found. Please upload and index a PDF file first via /api/upload."
        )

    if _retriever is None:
        logger.info("Rehydrating HybridRetriever from index directory...")
        try:
            _retriever = HybridRetriever(INDEX_DIR)
        except Exception as e:
            logger.error(f"Failed to load HybridRetriever: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to initialize search engine: {e}")

    return _retriever


# Pydantic Schemas for Request / Response
class QueryRequest(BaseModel):
    query: str
    top_k: int = 4


class CitationObject(BaseModel):
    chunk_id: str
    source_doc: str
    page_number: int
    text: str


class QueryResponse(BaseModel):
    answer: str
    citations: List[CitationObject]


@app.get("/")
def read_root():
    return FileResponse("./static/index.html")


@app.post("/api/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """
    Accepts a PDF multipart upload, runs smart parsing, constructs dense/sparse indices,
    and serializes the state to disk.
    """
    global _retriever
    filename = file.filename
    if not filename or not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    file_path = os.path.join(UPLOAD_DIR, filename)
    logger.info(f"Received file upload request: {filename}. Saving to {file_path}")

    # Write file stream to local uploads folder
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        logger.error(f"Failed to write file '{filename}' to disk: {e}")
        raise HTTPException(status_code=500, detail=f"File save error: {e}")

    # Parse and chunk (Phase 1)
    try:
        logger.info(f"Parsing and chunking PDF '{filename}'...")
        chunks = ingest_and_chunk_pdf(file_path)
    except Exception as e:
        logger.error(f"Ingestion failed for '{filename}': {e}")
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=500, detail=f"PDF parsing or chunking error: {e}")

    if not chunks:
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=400, detail="The uploaded PDF contains no extractable text.")

    # Index and persist (Phase 2)
    try:
        logger.info(f"Building dual index for {len(chunks)} chunks...")
        indexer = DualIndexer()
        indexer.build_indices(chunks)
        indexer.save_to_disk(INDEX_DIR)
        
        # Invalidate the cached retriever since the indices changed
        _retriever = None
        logger.info("Indices successfully updated and saved to disk.")
    except Exception as e:
        logger.error(f"Indexing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Index construction error: {e}")

    return {
        "status": "success",
        "message": f"Successfully parsed and indexed PDF '{filename}'.",
        "chunks_count": len(chunks)
    }


@app.post("/api/query", response_model=QueryResponse)
async def query_rag(request: QueryRequest, retriever: HybridRetriever = Depends(get_retriever)):
    """
    Accepts search queries, executes hybrid dense/sparse search with Cross-Encoder reranking,
    and returns a citation-grounded answer from the LLM.
    """
    logger.info(f"Processing query: '{request.query}' with top_k={request.top_k}")
    
    # 1. Stage 1/2/3 Retrieve candidates & rerank (Phase 3)
    try:
        retrieved_chunks = retriever.retrieve(request.query, top_k=request.top_k)
    except Exception as e:
        logger.error(f"Retrieval pipeline failed: {e}")
        raise HTTPException(status_code=500, detail=f"Retrieval engine error: {e}")

    # 2. Stage 4 Grounded Generation (Phase 4)
    try:
        answer = generate_grounded_answer(request.query, retrieved_chunks, provider="openai")
    except Exception as e:
        logger.error(f"Grounded generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Generation layer error: {e}")

    # 3. Format structured citation outputs
    citations = []
    for chunk in retrieved_chunks:
        # Gracefully handle both flat or nested metadata schemas
        metadata = chunk.get("metadata", {})
        chunk_id = metadata.get("chunk_id") or chunk.get("chunk_id", "unknown_id")
        source_doc = metadata.get("source_doc") or chunk.get("source_doc", "unknown_source")
        page_number = metadata.get("page_number") or chunk.get("page_number", 0)
        
        chunk_text = chunk.get("text") or "No text content available."
        
        citations.append(
            CitationObject(
                chunk_id=chunk_id,
                source_doc=source_doc,
                page_number=page_number,
                text=chunk_text
            )
        )

    return QueryResponse(
        answer=answer,
        citations=citations
    )


if __name__ == "__main__":
    import uvicorn
    # Local runtime runner
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
