import os
import sys
import datetime
import logging
import pandas as pd
from typing import List, Dict, Any

# Inject runtime pydantic compatibility mappings for older LangChain/Ragas code
# to prevent Pydantic v2 validation errors on deprecated class validators.
import pydantic.v1
sys.modules["langchain_core.pydantic_v1"] = pydantic.v1
sys.modules["langchain.pydantic_v1"] = pydantic.v1

# Inject ChatVertexAI mock class to langchain_community.chat_models before importing ragas
# to bypass VertexAI dependency checking in Ragas codebase.
from unittest.mock import MagicMock
import langchain_community.chat_models
langchain_community.chat_models.ChatVertexAI = MagicMock

# Import evaluation components
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("DocuMind.Evaluation")


class MockRagasResult:
    """
    Mock container that mimics the dict interface and DataFrame exporter
    of a real Ragas Result object for offline test execution.
    """
    def __init__(self, scores: Dict[str, float], df: pd.DataFrame) -> None:
        self.scores = scores
        self._df = df

    def to_pandas(self) -> pd.DataFrame:
        return self._df

    def __getitem__(self, key: str) -> float:
        return self.scores[key]

    def keys(self):
        return self.scores.keys()

    def values(self):
        return self.scores.values()

    def items(self):
        return self.scores.items()


def run_pipeline_evaluation(eval_dataset: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Runs quantitative RAG evaluation using the Ragas framework. Computes:
    - Faithfulness
    - Answer Relevancy
    - Context Precision
    - Context Recall

    Results are exported to a local timestamped CSV file under 'evals/experiments/'.

    Args:
        eval_dataset (List[Dict[str, Any]]): List of dictionary records, each containing:
            'question' (str), 'answer' (str), 'contexts' (list[str]), 'ground_truth' (str).

    Returns:
        Dict[str, float]: Dict of aggregate score results for the four target metrics.
    """
    if not eval_dataset:
        raise ValueError("Evaluation dataset cannot be empty.")

    logger.info(f"Starting quantitative evaluation harness on {len(eval_dataset)} test items...")

    # 1. Validate dataset records
    required_keys = {"question", "answer", "contexts", "ground_truth"}
    for idx, record in enumerate(eval_dataset):
        missing_keys = required_keys - record.keys()
        if missing_keys:
            raise ValueError(
                f"Record at index {idx} is missing required evaluation fields: {missing_keys}. "
                f"Provided: {list(record.keys())}"
            )
        if not isinstance(record["contexts"], list):
            raise TypeError(f"Record at index {idx}: 'contexts' field must be a list of strings.")

    # 2. Check if we are running in Offline Mock Mode
    if os.environ.get("DOCMIND_MOCK_LLM") == "true":
        logger.info("[Mock Mode] Generating mock evaluation scores and DataFrame.")
        mock_scores = {
            "faithfulness": 0.9450,
            "answer_relevancy": 0.9120,
            "context_precision": 0.8870,
            "context_recall": 0.9050
        }
        
        # Build mock DataFrame containing inputs plus mock score columns
        records = []
        for idx, record in enumerate(eval_dataset):
            records.append({
                "question": record["question"],
                "answer": record["answer"],
                "contexts": record["contexts"],
                "ground_truth": record["ground_truth"],
                "faithfulness": 0.95 - (idx * 0.02),
                "answer_relevancy": 0.92 - (idx * 0.01),
                "context_precision": 0.88 + (idx * 0.01),
                "context_recall": 0.90 - (idx * 0.01)
            })
        df = pd.DataFrame(records)
        results = MockRagasResult(mock_scores, df)
    
    else:
        # 3. Live Evaluation using ChatOpenAI
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY is not set. Ragas live evaluation requires ChatOpenAI. "
                "Configure OPENAI_API_KEY or set DOCMIND_MOCK_LLM=true to run offline tests."
            )

        # Convert to HuggingFace Dataset
        logger.info("Converting evaluation list to HuggingFace Dataset format...")
        dict_data = {
            "question": [rec["question"] for rec in eval_dataset],
            "answer": [rec["answer"] for rec in eval_dataset],
            "contexts": [rec["contexts"] for rec in eval_dataset],
            "ground_truth": [rec["ground_truth"] for rec in eval_dataset]
        }
        dataset = Dataset.from_dict(dict_data)

        # Initialize the LLM judge and Embeddings wrapper
        logger.info("Initializing OpenAI judge model (gpt-4o-mini, temp=0.0)...")
        judge_llm = LangchainLLMWrapper(
            ChatOpenAI(model="gpt-4o-mini", temperature=0.0, api_key=api_key)
        )
        
        logger.info("Initializing embeddings evaluator...")
        eval_embeddings = LangchainEmbeddingsWrapper(
            HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        )

        metrics = [faithfulness, answer_relevancy, context_precision, context_recall]

        # Execute evaluation
        try:
            logger.info("Invoking Ragas evaluation engine...")
            results = evaluate(
                dataset=dataset,
                metrics=metrics,
                llm=judge_llm,
                embeddings=eval_embeddings
            )
        except Exception as e:
            logger.error(f"Ragas evaluation engine failed: {e}")
            raise RuntimeError(f"Ragas evaluation error: {e}") from e

    # 4. Save and export DataFrame reporting
    df_results = results.to_pandas()
    os.makedirs("evals/experiments", exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = f"evals/experiments/eval_{timestamp}.csv"
    
    try:
        df_results.to_csv(csv_path, index=False)
        logger.info(f"Evaluation report successfully saved to: {csv_path}")
    except Exception as e:
        logger.warning(f"Failed to save CSV report to disk: {e}")

    # Return aggregate metrics
    return dict(results)


def print_comparison_table(docu_results: Dict[str, float], naive_results: Dict[str, float]) -> None:
    """
    Prints a clean, formatted CLI comparison table comparing 'Naive RAG Baseline'
    against the optimized 'DocuMind Pipeline'.
    """
    metrics_map = {
        "faithfulness": "Faithfulness (No Hallucination)",
        "answer_relevancy": "Answer Relevancy",
        "context_precision": "Context Retrieval Precision",
        "context_recall": "Context Retrieval Recall"
    }

    # Print table header
    print("\n" + "=" * 80)
    print(f"| {'RAG Metric':<30} | {'Naive Baseline':<18} | {'DocuMind Pipeline':<20} |")
    print("-" * 80)

    # Print each row
    for metric_key, display_name in metrics_map.items():
        docu_val = docu_results.get(metric_key, 0.0)
        naive_val = naive_results.get(metric_key, 0.0)
        print(f"| {display_name:<30} | {naive_val:<18.4f} | {docu_val:<20.4f} |")

    print("=" * 80 + "\n")


if __name__ == "__main__":
    print("-" * 60)
    print("DocuMind Quantitative Evaluation Harness (Phase 5)")
    print("-" * 60)

    # Simulate evaluation
    os.environ["DOCMIND_MOCK_LLM"] = "true"

    mock_eval_set = [
        {
            "question": "What is machine learning?",
            "answer": "Machine learning is a subset of AI that builds learning systems [chunk_0002].",
            "contexts": [
                "Machine Learning (ML) is a subset of AI that focuses on building systems that learn based on data."
            ],
            "ground_truth": "Machine learning is a subset of AI focusing on building systems that learn from data."
        },
        {
            "question": "Why preheat the oven when baking a chocolate cake?",
            "answer": "Preheating the oven causes carbon dioxide to expand rapidly, creating air pockets [chunk_0006].",
            "contexts": [
                "Preheating the oven causes the carbon dioxide gas produced by baking powder to expand rapidly, creating air pockets."
            ],
            "ground_truth": "Preheating creates heat that expands carbon dioxide gas, giving rise to battery pockets."
        }
    ]

    # Baseline scores representing a naive, unoptimized chunking retrieval system
    mock_naive_scores = {
        "faithfulness": 0.6800,
        "answer_relevancy": 0.7020,
        "context_precision": 0.5840,
        "context_recall": 0.6200
    }

    try:
        # Run DocuMind pipeline evaluation
        docu_scores = run_pipeline_evaluation(mock_eval_set)
        
        # Display the comparison table
        print_comparison_table(docu_scores, mock_naive_scores)
    except Exception as err:
        print(f"Evaluation failed: {err}")
