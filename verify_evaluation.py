import os
import shutil
import logging
import pandas as pd
from evaluation import run_pipeline_evaluation, print_comparison_table

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DocuMind.EvaluationVerification")


def verify_evaluation_harness():
    logger.info("Starting Phase 5 (Quantitative Evaluation Harness) Verification...")

    # Enable mock mode for testing offline
    os.environ["DOCMIND_MOCK_LLM"] = "true"

    # Define test dataset
    test_eval_set = [
        {
            "question": "What is AI?",
            "answer": "AI simulates human intelligence [chunk_0001].",
            "contexts": ["AI is the simulation of human intelligence by machines."],
            "ground_truth": "AI is the simulation of human intelligence."
        },
        {
            "question": "What is machine learning?",
            "answer": "Machine learning is a subset of AI that learns from data [chunk_0002].",
            "contexts": ["Machine Learning is a subfield of AI focused on building learning systems."],
            "ground_truth": "Machine learning is a subfield of AI focused on learning from data."
        }
    ]

    # Baseline scores for Naive RAG
    naive_scores = {
        "faithfulness": 0.6500,
        "answer_relevancy": 0.6800,
        "context_precision": 0.5500,
        "context_recall": 0.6000
    }

    try:
        # 1. Run DocuMind pipeline evaluation
        logger.info("Executing run_pipeline_evaluation...")
        docu_scores = run_pipeline_evaluation(test_eval_set)

        # 2. Assert dictionary scores
        assert isinstance(docu_scores, dict), "Evaluation output should be a dictionary"
        required_metrics = {"faithfulness", "answer_relevancy", "context_precision", "context_recall"}
        assert required_metrics.issubset(docu_scores.keys()), f"Missing metrics. Expected: {required_metrics}, Got: {list(docu_scores.keys())}"
        
        for metric, score in docu_scores.items():
            assert isinstance(score, float), f"Score for {metric} must be a float, got: {type(score)}"
            assert 0.0 <= score <= 1.0, f"Score for {metric} must be between 0.0 and 1.0, got: {score}"

        # 3. Assert CSV export and verify content
        logger.info("Verifying generated CSV file...")
        assert os.path.exists("evals/experiments"), "Directory 'evals/experiments' should exist"
        
        csv_files = [f for f in os.listdir("evals/experiments") if f.startswith("eval_") and f.endswith(".csv")]
        assert len(csv_files) >= 1, "No timestamped CSV report was created"

        # Read the latest CSV file
        latest_csv = sorted(csv_files)[-1]
        csv_path = os.path.join("evals/experiments", latest_csv)
        df_csv = pd.read_csv(csv_path)

        # Validate columns
        expected_cols = {"question", "answer", "contexts", "ground_truth", "faithfulness", "answer_relevancy", "context_precision", "context_recall"}
        assert expected_cols.issubset(df_csv.columns), f"CSV missing expected headers. Expected: {expected_cols}, Got: {list(df_csv.columns)}"
        assert len(df_csv) == len(test_eval_set), f"CSV record count mismatch. Expected: {len(test_eval_set)}, Got: {len(df_csv)}"

        # 4. Print comparison table
        logger.info("Testing CLI comparison table printing...")
        print_comparison_table(docu_scores, naive_scores)

        print("=" * 60)
        print("ALL EVALUATION HARNESS TESTS PASSED SUCCESSFULLY!")
        print("=" * 60)

    finally:
        # 5. Clean up CSV reports and experiments directory
        if os.path.exists("evals"):
            shutil.rmtree("evals")
            logger.info("Cleaned up evals folder directory.")


if __name__ == "__main__":
    verify_evaluation_harness()
