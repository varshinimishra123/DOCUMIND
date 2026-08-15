import os
import shutil
import logging
from verify_ingestion import create_sample_pdf, TEST_PDF_PATH
from ingestion import ingest_and_chunk_pdf
from storage import DualIndexer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DocuMind.StorageVerification")

INDEX_DIR = "test_storage_index"


def search_bm25(indexer: DualIndexer, query: str, k: int = 1):
    """
    Helper function to query the BM25 index and retrieve mapped chunks.
    """
    tokenized_query = indexer.tokenize(query)
    scores = indexer.bm25.get_scores(tokenized_query)
    
    # Associate scores with document indices
    scored_docs = list(enumerate(scores))
    # Sort by score descending
    scored_docs.sort(key=lambda x: x[1], reverse=True)
    
    # Retrieve mapped chunks for top k results
    results = []
    for doc_idx, score in scored_docs[:k]:
        if score > 0.0:  # Only return relevant results
            chunk = indexer.chunks_map[doc_idx].copy()
            chunk["score"] = score
            results.append(chunk)
    return results


def verify_dual_index():
    logger.info("Starting Phase 2 (Dual-Index Storage) Verification...")

    # 1. Generate the sample PDF
    create_sample_pdf(TEST_PDF_PATH)

    try:
        # 2. Ingest and semantic chunk the PDF (Phase 1)
        logger.info("Ingesting PDF into semantic chunks...")
        chunks = ingest_and_chunk_pdf(TEST_PDF_PATH)
        assert len(chunks) > 0, "No chunks generated from test PDF"
        logger.info(f"Ingested {len(chunks)} chunks successfully.")

        # 3. Instantiate DualIndexer and Build indices
        logger.info("Building dense and sparse indices...")
        indexer = DualIndexer()
        indexer.build_indices(chunks)

        # Assert indexes are set up in memory
        assert indexer.faiss_index is not None, "FAISS index should not be None"
        assert indexer.bm25 is not None, "BM25 index should not be None"
        assert len(indexer.chunks_map) == len(chunks), "Metadata mapping length mismatch"

        # 4. Save indices to disk
        logger.info(f"Persisting indices to local directory '{INDEX_DIR}'...")
        indexer.save_to_disk(INDEX_DIR)

        # Assert files exist
        assert os.path.exists(INDEX_DIR), "Index directory was not created"
        faiss_bin = os.path.join(INDEX_DIR, "index.faiss")
        faiss_pkl = os.path.join(INDEX_DIR, "index.pkl")
        bm25_pkl = os.path.join(INDEX_DIR, "bm25.pkl")
        assert os.path.exists(faiss_bin), f"Missing FAISS binary file: {faiss_bin}"
        assert os.path.exists(faiss_pkl), f"Missing FAISS metadata pickle: {faiss_pkl}"
        assert os.path.exists(bm25_pkl), f"Missing BM25 index file: {bm25_pkl}"
        logger.info("All files persisted to disk successfully.")

        # 5. Rehydrate a new indexer instance from disk
        logger.info("Instantiating fresh DualIndexer and loading from disk...")
        rehydrated_indexer = DualIndexer()
        rehydrated_indexer.load_from_disk(INDEX_DIR)

        # Assert rehydrated states
        assert rehydrated_indexer.faiss_index is not None, "Rehydrated FAISS index is None"
        assert rehydrated_indexer.bm25 is not None, "Rehydrated BM25 index is None"
        assert len(rehydrated_indexer.chunks_map) == len(chunks), "Rehydrated metadata mapping length mismatch"

        # 6. Test Retrieval
        # Test Case A: Dense (FAISS) similarity search
        dense_query = "neural networks and deep learning systems"
        logger.info(f"Testing FAISS Dense Retrieval with query: '{dense_query}'")
        dense_results = rehydrated_indexer.faiss_index.similarity_search(dense_query, k=1)
        
        assert len(dense_results) == 1, "FAISS did not return results"
        dense_doc = dense_results[0]
        logger.info(f"FAISS Match metadata: {dense_doc.metadata}")
        logger.info(f"FAISS Match content: {dense_doc.page_content[:120]}...")
        
        # Verify metadata mapping
        assert dense_doc.metadata["chunk_id"] == "chunk_0002", "FAISS failed to return the correct semantic chunk"
        assert dense_doc.metadata["source_doc"] == TEST_PDF_PATH, "Source document mapping incorrect"
        assert dense_doc.metadata["page_number"] == 1, "Page number mapping incorrect"

        # Test Case B: Sparse (BM25) keyword search
        sparse_query = "chocolate baking oven spring"
        logger.info(f"Testing BM25 Sparse Retrieval with query: '{sparse_query}'")
        sparse_results = search_bm25(rehydrated_indexer, sparse_query, k=1)
        
        assert len(sparse_results) == 1, "BM25 did not return results"
        sparse_match = sparse_results[0]
        logger.info(f"BM25 Match payload: chunk_id={sparse_match['chunk_id']}, page_number={sparse_match['page_number']}, score={sparse_match['score']:.4f}")
        logger.info(f"BM25 Match content: {sparse_match['text'][:120]}...")

        # Verify metadata mapping
        assert sparse_match["chunk_id"] == "chunk_0006", "BM25 failed to return the correct keyword chunk"
        assert sparse_match["source_doc"] == TEST_PDF_PATH, "BM25 source document incorrect"
        assert sparse_match["page_number"] == 3, "BM25 page number incorrect"

        print("=" * 60)
        print("ALL DUAL-INDEX STORAGE TESTS PASSED SUCCESSFULLY!")
        print("=" * 60)

    finally:
        # 7. Cleanup
        if os.path.exists(TEST_PDF_PATH):
            os.remove(TEST_PDF_PATH)
            logger.info(f"Cleaned up test file: {TEST_PDF_PATH}")
        if os.path.exists(INDEX_DIR):
            shutil.rmtree(INDEX_DIR)
            logger.info(f"Cleaned up index directory: {INDEX_DIR}")


if __name__ == "__main__":
    verify_dual_index()
