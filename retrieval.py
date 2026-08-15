import time
import logging
from typing import List, Dict, Any, Tuple
from storage import DualIndexer
from sentence_transformers import CrossEncoder

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("DocuMind.Retrieval")


class HybridRetriever:
    """
    HybridRetriever implements a production-grade two-stage hybrid search pipeline:
    - Stage 1 (Candidate Retrieval): Executes concurrent/parallel-concept query passes against
      a dense vector index (FAISS) and a sparse keyword index (BM25) to retrieve candidate chunks.
    - Stage 2 (Reciprocal Rank Fusion): Merges dense and sparse ranks using the Reciprocal Rank Fusion (RRF)
      algorithm (with smoothing constant k=60) and selects the top 10 candidates.
    - Stage 3 (Cross-Encoder Reranking): Reranks the top 10 candidates using a Cross-Encoder model
      ('cross-encoder/ms-marco-MiniLM-L-6-v2') to compute precise query-chunk relevance.
    """

    def __init__(self, index_dir: str) -> None:
        """
        Initializes the HybridRetriever. Loads indices from disk and instantiates the Cross-Encoder.
        
        Args:
            index_dir (str): Directory containing the FAISS and BM25 index files.
        """
        logger.info("Initializing HybridRetriever...")
        
        # 1. Load the dual indices
        try:
            self.indexer = DualIndexer()
            self.indexer.load_from_disk(index_dir)
        except Exception as e:
            logger.error(f"Failed to load indices from {index_dir}: {e}")
            raise RuntimeError(f"Index rehydration failed in HybridRetriever: {e}") from e

        # 2. Load the Cross-Encoder model for stage 2 reranking
        model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
        logger.info(f"Loading Cross-Encoder model: '{model_name}'...")
        try:
            self.cross_encoder = CrossEncoder(model_name)
        except Exception as e:
            logger.error(f"Failed to load Cross-Encoder model: {e}")
            raise RuntimeError(f"Cross-Encoder initialization failed: {e}") from e
            
        logger.info("HybridRetriever initialized successfully.")

    def retrieve(self, query: str, top_k: int = 4) -> List[Dict[str, Any]]:
        """
        Runs the full two-stage retrieval and reranking pipeline.
        
        Args:
            query (str): The raw user search query.
            top_k (int): Exact number of top relevant chunks to return. Default is 4.
            
        Returns:
            List[Dict[str, Any]]: Exactly the top_k most relevant chunks. Each dict contains:
                - 'chunk_id': unique sequence string
                - 'source_doc': filename of the PDF
                - 'page_number': 1-indexed page number
                - 'text': raw text string of the chunk
                - 'rerank_score': cross-encoder relevancy score
        """
        if not query or not query.strip():
            logger.warning("Empty search query provided. Returning empty list.")
            return []

        logger.info(f"Starting retrieval pipeline for query: '{query}'")

        # ==========================================
        # STAGE 1: CANDIDATE RETRIEVAL
        # ==========================================
        stage1_start = time.perf_counter()
        
        faiss_candidates: List[Tuple[Any, float]] = []
        bm25_candidates: List[Tuple[Dict[str, Any], int]] = []
        
        try:
            # A. Dense Vector retrieval (Retrieve top 20 candidates)
            # FAISS similarity_search_with_score returns List[Tuple[Document, distance]]
            if self.indexer.faiss_index is not None:
                faiss_candidates = self.indexer.faiss_index.similarity_search_with_score(query, k=20)
            else:
                logger.warning("FAISS index is not initialized. Dense search skipped.")
        except Exception as e:
            logger.error(f"Error during FAISS dense candidate retrieval: {e}")
            # We log and proceed so the system is fault-tolerant to dense failure

        try:
            # B. Sparse Keyword retrieval (Retrieve top 20 candidates)
            if self.indexer.bm25 is not None:
                tokenized_query = self.indexer.tokenize(query)
                scores = self.indexer.bm25.get_scores(tokenized_query)
                
                # Pair with indexes and sort descending
                scored_docs = list(enumerate(scores))
                scored_docs.sort(key=lambda x: x[1], reverse=True)
                
                # Keep top 20 documents that have relevance (score > 0)
                rank = 1
                for doc_idx, score in scored_docs[:20]:
                    if score > 0.0:
                        chunk = self.indexer.chunks_map[doc_idx]
                        bm25_candidates.append((chunk, rank))
                        rank += 1
            else:
                logger.warning("BM25 index is not initialized. Sparse search skipped.")
        except Exception as e:
            logger.error(f"Error during BM25 sparse candidate retrieval: {e}")
            # We log and proceed so the system is fault-tolerant to sparse failure

        stage1_time = time.perf_counter() - stage1_start
        logger.info(
            f"Stage 1 completed in {stage1_time:.4f}s. "
            f"Candidates retrieved: Dense={len(faiss_candidates)}, Sparse={len(bm25_candidates)}"
        )

        # ==========================================
        # STAGE 2: RECIPROCAL RANK FUSION (RRF)
        # ==========================================
        stage2_start = time.perf_counter()
        
        # Build index lookup maps and compile unified candidate pool
        faiss_ranks: Dict[str, int] = {}
        candidate_pool: Dict[str, Dict[str, Any]] = {}
        
        for rank_idx, (doc, _) in enumerate(faiss_candidates):
            cid = doc.metadata["chunk_id"]
            faiss_ranks[cid] = rank_idx + 1
            candidate_pool[cid] = {
                "chunk_id": cid,
                "source_doc": doc.metadata["source_doc"],
                "page_number": doc.metadata["page_number"],
                "text": doc.page_content
            }
            
        bm25_ranks: Dict[str, int] = {}
        for chunk, rank in bm25_candidates:
            cid = chunk["chunk_id"]
            bm25_ranks[cid] = rank
            if cid not in candidate_pool:
                candidate_pool[cid] = chunk.copy()

        # Compute RRF score with smoothing constant k=60
        k_smooth = 60
        rrf_scores: Dict[str, float] = {}
        
        for cid in candidate_pool:
            f_rank = faiss_ranks.get(cid)
            b_rank = bm25_ranks.get(cid)
            
            rrf_val = 0.0
            if f_rank is not None:
                rrf_val += 1.0 / (k_smooth + f_rank)
            if b_rank is not None:
                rrf_val += 1.0 / (k_smooth + b_rank)
            rrf_scores[cid] = rrf_val

        # Sort candidate IDs based on RRF scores descending
        sorted_candidates = sorted(candidate_pool.keys(), key=lambda cid: rrf_scores[cid], reverse=True)
        
        # Select the top 10 candidates for Cross-Encoder reranking
        top_rrf_cids = sorted_candidates[:10]
        top_rrf_chunks = [candidate_pool[cid] for cid in top_rrf_cids]
        
        stage2_time = time.perf_counter() - stage2_start
        logger.info(
            f"Stage 2 completed in {stage2_time:.4f}s. "
            f"Fused {len(candidate_pool)} unique candidates down to {len(top_rrf_chunks)} RRF outputs."
        )

        if not top_rrf_chunks:
            logger.info("No retrieval candidates found matching the query. Returning empty results.")
            return []

        # ==========================================
        # STAGE 3: CROSS-ENCODER RERANKING
        # ==========================================
        stage3_start = time.perf_counter()
        
        try:
            # Pair query with each candidate text
            pairs = [[query, chunk["text"]] for chunk in top_rrf_chunks]
            
            # Predict relevance scores (higher means more relevant)
            rerank_scores = self.cross_encoder.predict(pairs)
            
            # Attach score and sort candidates descending
            for chunk, score in zip(top_rrf_chunks, rerank_scores):
                chunk["rerank_score"] = float(score)
            
            top_rrf_chunks.sort(key=lambda x: x["rerank_score"], reverse=True)
        except Exception as e:
            logger.error(f"Error during Cross-Encoder reranking: {e}. Returning un-reranked RRF results.")
            # Fail gracefully: default score to 0.0 and keep RRF sorting
            for chunk in top_rrf_chunks:
                chunk["rerank_score"] = 0.0

        stage3_time = time.perf_counter() - stage3_start
        logger.info(f"Stage 3 completed in {stage3_time:.4f}s. Cross-encoder reranked successfully.")

        # Log total pipeline time metric
        total_pipeline_time = stage1_time + stage2_time + stage3_time
        logger.info(f"Total hybrid retrieval and reranking executed in {total_pipeline_time:.4f}s.")

        # Return exactly top_k chunks
        return top_rrf_chunks[:top_k]


if __name__ == "__main__":
    print("-" * 60)
    print("DocuMind Hybrid Retrieval & Reranking (Phase 3) Usage Example")
    print("-" * 60)

    # Note: Requires indices already created in Phase 2
    # For illustration, if indices exist, run:
    # retriever = HybridRetriever("test_dual_index")
    # results = retriever.retrieve("What is neural networks?")
    print("HybridRetriever class defined. To test query retrieval, run the verification script.")
