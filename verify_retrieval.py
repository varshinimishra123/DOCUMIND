import os
import shutil
import logging
from verify_ingestion import create_sample_pdf, TEST_PDF_PATH
from ingestion import ingest_and_chunk_pdf
from storage import DualIndexer
from retrieval import HybridRetriever

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DocuMind.RetrievalVerification")

INDEX_DIR = "test_retrieval_index"


def verify_retrieval_pipeline():
    logger.info("Starting Phase 3 (Hybrid Retrieval & Reranking) Verification...")

    # 1. Generate the sample PDF
    create_sample_pdf(TEST_PDF_PATH)

    try:
        # 2. Ingest and semantic chunk the PDF (Phase 1)
        logger.info("Ingesting PDF into semantic chunks...")
        chunks = ingest_and_chunk_pdf(TEST_PDF_PATH)
        assert len(chunks) == 6, f"Expected 6 chunks, got {len(chunks)}"

        # 3. Build indices and save to disk (Phase 2)
        logger.info("Building and persisting dual indices...")
        indexer = DualIndexer()
        indexer.build_indices(chunks)
        indexer.save_to_disk(INDEX_DIR)

        # 4. Instantiate HybridRetriever (Phase 3)
        logger.info("Initializing HybridRetriever...")
        retriever = HybridRetriever(INDEX_DIR)

        # 5. Run retrieval tests
        # Test Case A: Query targeting machine learning
        query_a = "artificial intelligence neural networks and deep learning systems"
        top_k_a = 2
        logger.info(f"Retrieving for Query A: '{query_a}' (top_k={top_k_a})")
        results_a = retriever.retrieve(query_a, top_k=top_k_a)

        assert isinstance(results_a, list), "Results should be a list"
        assert len(results_a) == top_k_a, f"Expected {top_k_a} results, got {len(results_a)}"
        
        # Verify rank ordering and metadata content
        for idx, result in enumerate(results_a):
            assert "chunk_id" in result, "Missing 'chunk_id'"
            assert "source_doc" in result, "Missing 'source_doc'"
            assert "page_number" in result, "Missing 'page_number'"
            assert "text" in result, "Missing 'text'"
            assert "rerank_score" in result, "Missing 'rerank_score'"
            
            logger.info(f"A[{idx}]: chunk_id={result['chunk_id']}, page={result['page_number']}, score={result['rerank_score']:.4f}")
            logger.info(f"Snippet: {result['text'][:100]}...\n")

            # Check that Cross-Encoder scored the correct top match
            if idx == 0:
                # The most relevant machine learning chunk should be from Page 1
                assert result["page_number"] == 1, f"Top match should be from Page 1 (AI/ML), got Page {result['page_number']}"
            
            # Check descending score order
            if idx > 0:
                assert results_a[idx - 1]["rerank_score"] >= result["rerank_score"], "Scores are not in descending order"

        # Test Case B: Query targeting baking chocolate cake
        query_b = "baking powder baking soda carbon dioxide pockets cake rises"
        top_k_b = 3
        logger.info(f"Retrieving for Query B: '{query_b}' (top_k={top_k_b})")
        results_b = retriever.retrieve(query_b, top_k=top_k_b)

        assert len(results_b) == top_k_b, f"Expected {top_k_b} results, got {len(results_b)}"
        for idx, result in enumerate(results_b):
            logger.info(f"B[{idx}]: chunk_id={result['chunk_id']}, page={result['page_number']}, score={result['rerank_score']:.4f}")
            logger.info(f"Snippet: {result['text'][:100]}...\n")

            # Check that Cross-Encoder scored the correct top match
            if idx == 0:
                # The baking chemical expansion chunk is chunk_0006 on Page 3
                assert result["page_number"] == 3, f"Top match should be from Page 3 (Baking), got Page {result['page_number']}"

            # Check descending score order
            if idx > 0:
                assert results_b[idx - 1]["rerank_score"] >= result["rerank_score"], "Scores are not in descending order"

        print("=" * 60)
        print("ALL HYBRID RETRIEVAL & RERANKING TESTS PASSED SUCCESSFULLY!")
        print("=" * 60)

    finally:
        # 6. Cleanup
        if os.path.exists(TEST_PDF_PATH):
            os.remove(TEST_PDF_PATH)
            logger.info(f"Cleaned up test file: {TEST_PDF_PATH}")
        if os.path.exists(INDEX_DIR):
            shutil.rmtree(INDEX_DIR)
            logger.info(f"Cleaned up index directory: {INDEX_DIR}")


if __name__ == "__main__":
    verify_retrieval_pipeline()
