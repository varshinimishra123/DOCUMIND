import os
import shutil
import logging
from fastapi.testclient import TestClient

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DocuMind.DeploymentVerification")

# Configure mock mode for offline testing
os.environ["DOCMIND_MOCK_LLM"] = "true"

from verify_ingestion import create_sample_pdf, TEST_PDF_PATH
from main import app, UPLOAD_DIR, INDEX_DIR


def verify_production_deployment():
    logger.info("Starting Phase 6 (Production Deployment) Verification...")

    # 1. Generate the sample PDF
    create_sample_pdf(TEST_PDF_PATH)

    # 2. Instantiate TestClient
    client = TestClient(app)

    try:
        # 3. Test POST /api/upload
        logger.info("Testing POST /api/upload endpoint with PDF multipart form-data...")
        with open(TEST_PDF_PATH, "rb") as f:
            upload_response = client.post(
                "/api/upload",
                files={"file": (TEST_PDF_PATH, f, "application/pdf")}
            )

        # Assertions
        assert upload_response.status_code == 200, f"Upload failed: {upload_response.text}"
        upload_data = upload_response.json()
        assert upload_data["status"] == "success", "Response status should be 'success'"
        assert "chunks_count" in upload_data, "Response missing 'chunks_count'"
        assert upload_data["chunks_count"] == 6, f"Expected 6 chunks to be indexed, got {upload_data['chunks_count']}"
        logger.info("POST /api/upload test passed successfully. Indices constructed.")

        # 4. Test POST /api/query
        logger.info("Testing POST /api/query endpoint with JSON query request...")
        query_payload = {
            "query": "Explain neural networks and machine learning systems",
            "top_k": 2
        }
        query_response = client.post(
            "/api/query",
            json=query_payload,
            headers={"Origin": "http://localhost:3000"}
        )

        # Assertions
        assert query_response.status_code == 200, f"Query failed: {query_response.text}"
        query_data = query_response.json()
        assert "answer" in query_data, "Response missing 'answer'"
        assert "citations" in query_data, "Response missing 'citations'"
        assert len(query_data["citations"]) == 2, f"Expected exactly 2 citations, got {len(query_data['citations'])}"

        # Validate citation schema
        for idx, citation in enumerate(query_data["citations"]):
            assert "chunk_id" in citation, f"Citation {idx} missing 'chunk_id'"
            assert "source_doc" in citation, f"Citation {idx} missing 'source_doc'"
            assert "page_number" in citation, f"Citation {idx} missing 'page_number'"
            logger.info(f"Citation [{idx}]: chunk_id={citation['chunk_id']}, source_doc={citation['source_doc']}, page={citation['page_number']}")

        logger.info(f"Grounded response: '{query_data['answer']}'")
        logger.info("POST /api/query test passed successfully.")

        # 5. Verify CORS headers are present in response
        logger.info("Verifying CORS Headers...")
        cors_headers = ["access-control-allow-origin", "access-control-allow-credentials"]
        for header in cors_headers:
            assert header in query_response.headers, f"Missing expected CORS header: {header}"
            
        logger.info(f"CORS Headers Verified: Access-Control-Allow-Origin = '{query_response.headers.get('access-control-allow-origin')}'")

        print("=" * 60)
        print("ALL FASTAPI PRODUCTION DEPLOYMENT TESTS PASSED SUCCESSFULLY!")
        print("=" * 60)

    finally:
        # 6. Cleanup files and indexes
        if os.path.exists(TEST_PDF_PATH):
            os.remove(TEST_PDF_PATH)
            logger.info(f"Cleaned up test file: {TEST_PDF_PATH}")
        if os.path.exists(UPLOAD_DIR):
            shutil.rmtree(UPLOAD_DIR)
            logger.info(f"Cleaned up uploads folder directory: {UPLOAD_DIR}")
        if os.path.exists(INDEX_DIR):
            shutil.rmtree(INDEX_DIR)
            logger.info(f"Cleaned up index folder directory: {INDEX_DIR}")


if __name__ == "__main__":
    verify_production_deployment()
