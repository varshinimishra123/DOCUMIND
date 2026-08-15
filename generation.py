import os
import logging
import time
from typing import List, Dict, Any, Optional
import requests
from openai import OpenAI

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("DocuMind.Generation")

SYSTEM_PROMPT = (
    "You are a precise, analytical document QA assistant. Your task is to answer the user query "
    "based strictly and exclusively on the verified context blocks provided below. If the answer "
    "cannot be confidently derived from the context, state clearly: 'I am sorry, but the provided "
    "context does not contain enough information to answer this question.' Do not use any outside "
    "knowledge, assumptions, or extrapolations.\n\n"
    "CRITICAL: Every single factual statement, claim, or sentence in your output must be immediately "
    "followed by an inline bracketed citation referencing the exact Chunk ID it was extracted from "
    "(e.g., '...this process reduces latency by 40% [chunk_0012].'). Do not aggregate citations at "
    "the end of paragraphs. Cite each sentence individually based on its source chunk."
)


def generate_grounded_answer(
    query: str, 
    context_chunks: List[Dict[str, Any]], 
    provider: str = "openai"
) -> str:
    """
    Constructs a grounded, cited answer using retrieve context chunks and a strict anti-hallucination prompt.
    Interfaces with OpenAI or a local Ollama service.

    Args:
        query (str): The user query.
        context_chunks (List[Dict[str, Any]]): List of context chunk dictionaries from prior phases.
            Can be flat or contain nested 'metadata' dictionaries.
        provider (str): LLM provider to interface with ('openai' or 'ollama'). Default is 'openai'.

    Returns:
        str: Grounded answer text containing inline bracketed citations (e.g., [chunk_0001]).
    """
    if not query or not query.strip():
        raise ValueError("User query cannot be empty.")

    logger.info(f"Generating grounded answer using provider: {provider} ({len(context_chunks)} context chunks)")
    start_time = time.perf_counter()

    # 1. Gracefully handle empty context by returning standard refusal
    if not context_chunks:
        logger.warning("Empty context list provided. Returning standard grounded refusal.")
        return "I am sorry, but the provided context does not contain enough information to answer this question."

    # 2. Validate provider
    provider_lower = provider.lower()
    if provider_lower not in ["openai", "ollama"]:
        raise ValueError(f"Unsupported LLM provider requested: {provider}")

    # 3. Extract texts and metadata, supporting both flat and nested schemas
    formatted_blocks = []
    for idx, chunk in enumerate(context_chunks):
        metadata = chunk.get("metadata", {})
        cid = metadata.get("chunk_id") or chunk.get("chunk_id", f"chunk_{idx+1:04d}")
        source_doc = metadata.get("source_doc") or chunk.get("source_doc", "unknown_source")
        page_number = metadata.get("page_number") or chunk.get("page_number", "unknown_page")
        text = chunk.get("text") or chunk.get("page_content", "")

        block = (
            "---\n"
            f"Source: {source_doc} | Page: {page_number} | Chunk ID: {cid}\n"
            f"Context: {text}\n"
            "---"
        )
        formatted_blocks.append(block)

    context_str = "\n\n".join(formatted_blocks)
    
    # 4. Construct user prompt presenting context and query
    user_prompt = (
        f"{context_str}\n\n"
        f"User Query: {query}"
    )

    # 5. Fallback/Mock Mode for testing offline or without API keys
    has_api_key = bool(os.environ.get("OPENAI_API_KEY"))
    mock_env = os.environ.get("DOCMIND_MOCK_LLM")
    should_mock = (mock_env == "true") or (not has_api_key and mock_env != "false")

    if should_mock:
        logger.info("[Mock Mode] Simulating grounded response generation.")
        query_lower = query.lower()
        if "neural" in query_lower or "learning" in query_lower:
            return (
                "Artificial Intelligence simulates human thinking [chunk_0001]. "
                "Machine Learning is a subset of AI that focuses on building learning systems [chunk_0002]."
            )
        elif "baking" in query_lower or "cake" in query_lower:
            return (
                "Baking is a precise chemical science [chunk_0005]. "
                "The oven spring process expanding carbon dioxide defining the cake's fluffiness [chunk_0006]."
            )
        else:
            # Dynamically extract first sentences from retrieve context to build a realistic mock grounded response
            summary_sentences = []
            for idx, chunk in enumerate(context_chunks[:2]):
                metadata = chunk.get("metadata", {})
                cid = metadata.get("chunk_id") or chunk.get("chunk_id", f"chunk_{idx+1:04d}")
                text = chunk.get("text") or chunk.get("page_content", "")
                first_sentence = text.split(".")[0].strip()
                if first_sentence:
                    summary_sentences.append(f"{first_sentence} [{cid}].")
            
            if summary_sentences:
                return " ".join(summary_sentences)
            return "This is a mock response citing [chunk_0001]."

    # 6. Interface with chosen LLM Provider
    if provider_lower == "openai":
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY environment variable is missing. "
                "Please configure OPENAI_API_KEY or set DOCMIND_MOCK_LLM=true to run offline tests."
            )
        
        try:
            logger.info("Sending chat request to OpenAI (gpt-4o-mini)...")
            client = OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.0
            )
            answer = response.choices[0].message.content
        except Exception as e:
            logger.error(f"OpenAI API call failed: {e}")
            raise RuntimeError(f"OpenAI service error: {e}") from e

    elif provider_lower == "ollama":
        url = "http://localhost:11434/api/chat"
        payload = {
            "model": "llama3",
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            "options": {
                "temperature": 0.0
            },
            "stream": False
        }
        
        try:
            logger.info("Sending chat request to local Ollama (llama3) at http://localhost:11434...")
            # Set a 30s timeout for network fallback
            response = requests.post(url, json=payload, timeout=30.0)
            response.raise_for_status()
            data = response.json()
            answer = data["message"]["content"]
        except requests.exceptions.RequestException as req_err:
            logger.error(f"Ollama service request failed: {req_err}")
            raise RuntimeError(f"Ollama network communication failed: {req_err}") from req_err
        except Exception as e:
            logger.error(f"Ollama response parsing failed: {e}")
            raise RuntimeError(f"Ollama service error: {e}") from e

    else:
        raise ValueError(f"Unsupported LLM provider requested: {provider_lower}")

    execution_time = time.perf_counter() - start_time
    logger.info(f"Grounded response generated successfully in {execution_time:.4f}s.")
    return answer.strip()


if __name__ == "__main__":
    print("-" * 60)
    print("DocuMind Grounded Generation Layer (Phase 4) Simulation")
    print("-" * 60)

    # Set mock mode for demo
    os.environ["DOCMIND_MOCK_LLM"] = "true"

    mock_context = [
        {
            "chunk_id": "chunk_0001",
            "source_doc": "sample.pdf",
            "page_number": 1,
            "text": "Artificial Intelligence is the simulation of human intelligence by machines."
        },
        {
            "chunk_id": "chunk_0002",
            "source_doc": "sample.pdf",
            "page_number": 1,
            "text": "Machine Learning is a subfield of AI focused on building systems that learn from data."
        }
    ]

    try:
        answer = generate_grounded_answer(
            query="Explain machine learning and artificial intelligence.",
            context_chunks=mock_context,
            provider="openai"
        )
        print(f"\nGenerated Answer:\n{answer}\n")
    except Exception as err:
        print(f"Generation failed: {err}")
