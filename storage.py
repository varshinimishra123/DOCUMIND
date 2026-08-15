import os
import re
import pickle
import logging
from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from rank_bm25 import BM25Okapi

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("DocuMind.Storage")


class DualIndexer:
    """
    DualIndexer constructs and manages parallel retrieval indices for RAG:
    1. A Dense (FAISS) vector index using HuggingFace 'all-MiniLM-L6-v2' embeddings.
    2. A Sparse (BM25) keyword search index using the rank_bm25 library.
    
    Includes capability to serialize/deserialize indices to/from disk with full metadata tracking.
    """

    def __init__(self) -> None:
        """
        Initializes the DualIndexer, configuring the HuggingFace embeddings model.
        """
        logger.info("Initializing DualIndexer embedding model ('all-MiniLM-L6-v2')...")
        try:
            self.embeddings = HuggingFaceEmbeddings(
                model_name="all-MiniLM-L6-v2",
                encode_kwargs={"normalize_embeddings": True}
            )
        except Exception as e:
            logger.error(f"Failed to initialize HuggingFace embeddings: {e}")
            raise RuntimeError(f"Embedding initialization error: {e}") from e

        self.faiss_index: Optional[FAISS] = None
        self.bm25: Optional[BM25Okapi] = None
        self.chunks_map: List[Dict[str, Any]] = []

    @staticmethod
    def tokenize(text: str) -> List[str]:
        """
        Cleans and tokenizes text for keyword-based indexing (BM25).
        Removes non-alphanumeric characters, converts to lowercase, and splits into terms.
        """
        if not text:
            return []
        cleaned = re.sub(r"[^\w\s]", "", text.lower())
        return [term for term in cleaned.split() if term]

    def build_indices(self, chunks: List[Dict[str, Any]]) -> None:
        """
        Constructs parallel FAISS (dense) and BM25 (sparse) indices.
        
        Args:
            chunks (List[Dict[str, Any]]): List of chunk dictionaries from Phase 1.
                Each chunk must contain: 'chunk_id', 'source_doc', 'page_number', 'text'.
        """
        if not chunks:
            logger.warning("Empty chunks list provided. Skipping index build.")
            self.chunks_map = []
            self.faiss_index = None
            self.bm25 = None
            return

        logger.info(f"Building dual indices for {len(chunks)} chunks...")
        
        # 1. Validate chunk structures & build LangChain Document objects
        documents: List[Document] = []
        tokenized_corpus: List[List[str]] = []
        
        for idx, chunk in enumerate(chunks):
            # Ensure all required keys exist
            required_keys = {"chunk_id", "source_doc", "page_number", "text"}
            missing_keys = required_keys - chunk.keys()
            if missing_keys:
                raise ValueError(
                    f"Chunk at index {idx} is missing required metadata keys: {missing_keys}. "
                    f"Provided keys: {list(chunk.keys())}"
                )

            # Build Document for FAISS
            doc_metadata = {
                "chunk_id": chunk["chunk_id"],
                "source_doc": chunk["source_doc"],
                "page_number": chunk["page_number"]
            }
            documents.append(Document(page_content=chunk["text"], metadata=doc_metadata))

            # Tokenize for BM25
            tokenized_terms = self.tokenize(chunk["text"])
            tokenized_corpus.append(tokenized_terms)

        # 2. Build FAISS index
        try:
            logger.info("Generating embeddings and building FAISS vector index...")
            self.faiss_index = FAISS.from_documents(documents, self.embeddings)
        except Exception as e:
            logger.error(f"Failed to build FAISS index: {e}")
            raise RuntimeError(f"FAISS index construction failed: {e}") from e

        # 3. Build BM25 index
        try:
            logger.info("Building BM25 keyword index...")
            self.bm25 = BM25Okapi(tokenized_corpus)
        except Exception as e:
            logger.error(f"Failed to build BM25 index: {e}")
            raise RuntimeError(f"BM25 index construction failed: {e}") from e

        # 4. Map index offsets back to original custom payloads
        # We store a shallow copy of the list of dictionaries
        self.chunks_map = list(chunks)
        logger.info("Indices built successfully.")

    def save_to_disk(self, directory_path: str) -> None:
        """
        Saves the complete state of both indices and metadata mappings to the specified directory.
        
        Args:
            directory_path (str): Path of the destination directory.
        """
        if self.faiss_index is None or self.bm25 is None or not self.chunks_map:
            raise ValueError("Indices are not built. Call build_indices() before saving.")

        try:
            os.makedirs(directory_path, exist_ok=True)
            logger.info(f"Saving indices to directory: {directory_path}")

            # Save FAISS Index (saves index.faiss and index.pkl)
            self.faiss_index.save_local(directory_path)

            # Save BM25 state and the chunks metadata mapping
            bm25_file_path = os.path.join(directory_path, "bm25.pkl")
            bm25_payload = {
                "bm25": self.bm25,
                "chunks_map": self.chunks_map
            }
            with open(bm25_file_path, "wb") as f:
                pickle.dump(bm25_payload, f)

            logger.info("Indices persisted to disk successfully.")
        except Exception as e:
            logger.error(f"Failed to save indices to disk: {e}")
            raise RuntimeError(f"Disk persistence error: {e}") from e

    def load_from_disk(self, directory_path: str) -> None:
        """
        Loads and rehydrates FAISS, BM25, and metadata states from the specified directory.
        
        Args:
            directory_path (str): Path to the index storage directory.
        """
        logger.info(f"Loading indices from directory: {directory_path}")
        
        # 1. Rehydrate FAISS Index (requires allow_dangerous_deserialization for pickle loading)
        try:
            self.faiss_index = FAISS.load_local(
                directory_path,
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            logger.info("FAISS vector store rehydrated successfully.")
        except Exception as e:
            logger.error(f"Failed to load FAISS index: {e}")
            raise RuntimeError(f"FAISS index rehydration failed: {e}") from e

        # 2. Rehydrate BM25 state and chunks map
        bm25_file_path = os.path.join(directory_path, "bm25.pkl")
        if not os.path.exists(bm25_file_path):
            raise FileNotFoundError(f"BM25 storage file not found at: {bm25_file_path}")

        try:
            with open(bm25_file_path, "rb") as f:
                bm25_payload = pickle.load(f)
            
            self.bm25 = bm25_payload["bm25"]
            self.chunks_map = bm25_payload["chunks_map"]
            logger.info(f"BM25 keyword store and {len(self.chunks_map)} metadata payloads rehydrated successfully.")
        except Exception as e:
            logger.error(f"Failed to load BM25 index: {e}")
            raise RuntimeError(f"BM25 index rehydration failed: {e}") from e


if __name__ == "__main__":
    print("-" * 60)
    print("DocuMind Dual-Index Storage (Phase 2) Build & Save Example")
    print("-" * 60)

    # Sample mock chunks from Phase 1
    mock_chunks = [
        {
            "chunk_id": "chunk_0001",
            "source_doc": "sample_ml.pdf",
            "page_number": 1,
            "text": "Machine learning uses neural networks to recognize complex visual patterns in digital images."
        },
        {
            "chunk_id": "chunk_0002",
            "source_doc": "sample_ml.pdf",
            "page_number": 2,
            "text": "Marine biology investigates oceanic life forms like coral reefs and deep sea thermal vents."
        }
    ]

    target_dir = "test_dual_index"

    try:
        # Initialize
        indexer = DualIndexer()
        
        # Build
        indexer.build_indices(mock_chunks)
        
        # Save
        indexer.save_to_disk(target_dir)
        print(f"\nSuccessfully built and saved index files in '{target_dir}/'.")
        print(f"Files created: {os.listdir(target_dir)}")
        
        # Verify clean up (normally done in tests)
        import shutil
        if os.path.exists(target_dir):
            shutil.rmtree(target_dir)
            print(f"Cleaned up demo folder: {target_dir}")
            
    except Exception as err:
        print(f"\nInitialization/Build failed: {err}")
