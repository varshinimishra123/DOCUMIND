import os
import logging
from fastapi.testclient import TestClient

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DocuMind.FrontendVerification")

# Configure mock environment
os.environ["DOCMIND_MOCK_LLM"] = "true"

from main import app


def verify_static_frontend():
    logger.info("Starting Phase 7 (UI Frontend Client) Verification...")

    # Instantiate TestClient
    client = TestClient(app)

    try:
        # 1. Test GET / to retrieve index.html
        logger.info("Verifying GET / returns index.html...")
        response = client.get("/")
        assert response.status_code == 200, f"Failed to get root index: {response.text}"
        assert "text/html" in response.headers.get("content-type", ""), "Content-Type must be text/html"
        assert "<title>DocuMind - Hybrid RAG Document QA Engine</title>" in response.text, "Title missing or incorrect in HTML response"
        logger.info("Root path verification passed. Served index.html.")

        # 2. Test GET /static/style.css
        logger.info("Verifying GET /static/style.css...")
        css_response = client.get("/static/style.css")
        assert css_response.status_code == 200, f"Failed to retrieve style.css: {css_response.text}"
        assert "text/css" in css_response.headers.get("content-type", ""), "Content-Type must be text/css"
        assert "DocuMind Premium Design System Stylesheet" in css_response.text, "CSS header comment missing"
        logger.info("style.css verification passed.")

        # 3. Test GET /static/app.js
        logger.info("Verifying GET /static/app.js...")
        js_response = client.get("/static/app.js")
        assert js_response.status_code == 200, f"Failed to retrieve app.js: {js_response.text}"
        assert "javascript" in js_response.headers.get("content-type", "").lower(), f"Content-Type must contain 'javascript', got: {js_response.headers.get('content-type')}"
        assert "DocuMind UI Frontend Script" in js_response.text, "JS header comment missing"
        logger.info("app.js verification passed.")

        print("=" * 60)
        print("ALL FRONTEND STATIC ASSET SERVING TESTS PASSED SUCCESSFULLY!")
        print("=" * 60)

    except AssertionError as e:
        logger.error(f"Assertion failed: {e}")
        raise e
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise e


if __name__ == "__main__":
    verify_static_frontend()
