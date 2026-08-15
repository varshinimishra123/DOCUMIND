import os
import logging
from generation import generate_grounded_answer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DocuMind.GenerationVerification")


def verify_grounded_generation():
    logger.info("Starting Phase 4 (Grounded Generation Layer) Verification...")

    # Enable mock mode for offline test verification
    os.environ["DOCMIND_MOCK_LLM"] = "true"

    # Define test contexts
    # Flat format (Phase 3 output)
    flat_context = [
        {
            "chunk_id": "chunk_0001",
            "source_doc": "test_doc.pdf",
            "page_number": 1,
            "text": "Artificial Intelligence refers to the simulation of human intelligence."
        },
        {
            "chunk_id": "chunk_0002",
            "source_doc": "test_doc.pdf",
            "page_number": 1,
            "text": "Machine Learning is a subset of AI that focuses on learning from data."
        }
    ]

    # Nested format (Phase 4 requirement spec)
    nested_context = [
        {
            "text": "Baking is a precise science.",
            "metadata": {
                "chunk_id": "chunk_0005",
                "source_doc": "test_doc.pdf",
                "page_number": 3
            }
        },
        {
            "text": "The oven spring process expanding carbon dioxide defines fluffiness.",
            "metadata": {
                "chunk_id": "chunk_0006",
                "source_doc": "test_doc.pdf",
                "page_number": 3
            }
        }
    ]

    # 1. Test Flat metadata ingestion
    logger.info("Testing flat metadata schema parsing...")
    query_a = "Explain machine learning neural networks"
    answer_a = generate_grounded_answer(query_a, flat_context, provider="openai")
    
    assert "chunk_0001" in answer_a, "Flat parsing failed to generate citation for chunk_0001"
    assert "chunk_0002" in answer_a, "Flat parsing failed to generate citation for chunk_0002"
    logger.info(f"Query A Answer (Flat): '{answer_a}'")

    # 2. Test Nested metadata ingestion
    logger.info("Testing nested metadata schema parsing...")
    query_b = "What is the science of baking cake?"
    answer_b = generate_grounded_answer(query_b, nested_context, provider="openai")
    
    assert "chunk_0005" in answer_b, "Nested parsing failed to generate citation for chunk_0005"
    assert "chunk_0006" in answer_b, "Nested parsing failed to generate citation for chunk_0006"
    logger.info(f"Query B Answer (Nested): '{answer_b}'")

    # 3. Test empty context handling (anti-hallucination guard)
    logger.info("Testing empty context handling...")
    refusal_msg = "I am sorry, but the provided context does not contain enough information to answer this question."
    answer_c = generate_grounded_answer("How deep is the ocean?", [], provider="openai")
    
    assert answer_c == refusal_msg, f"Expected standard refusal message, got: '{answer_c}'"
    logger.info("Empty context successfully triggered anti-hallucination refusal.")

    # 4. Test invalid provider validation
    logger.info("Testing provider parameter verification...")
    try:
        generate_grounded_answer("test query", flat_context, provider="unsupported_provider")
        assert False, "Should have raised ValueError for unsupported provider"
    except ValueError as val_err:
        logger.info(f"Successfully caught unsupported provider exception: {val_err}")

    # 5. Test missing API key validation (when mock is disabled)
    logger.info("Testing API key presence validation...")
    del os.environ["DOCMIND_MOCK_LLM"]
    
    # Save original API key if present
    original_key = os.environ.get("OPENAI_API_KEY")
    if "OPENAI_API_KEY" in os.environ:
        del os.environ["OPENAI_API_KEY"]

    try:
        generate_grounded_answer("test query", flat_context, provider="openai")
        assert False, "Should have raised ValueError for missing API key when mock is disabled"
    except ValueError as val_err:
        logger.info(f"Successfully caught missing API key exception: {val_err}")

    # Restore environment state
    if original_key is not None:
        os.environ["OPENAI_API_KEY"] = original_key

    print("=" * 60)
    print("ALL GROUNDED GENERATION LAYER TESTS PASSED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    verify_grounded_generation()
