import os
import re
import logging
from typing import List, Dict, Any, Optional
from langchain_community.document_loaders import PyPDFLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("DocuMind.Ingestion")

# Module-level caching for model and chunker to avoid re-initialization overhead
_embeddings: Optional[HuggingFaceEmbeddings] = None
_chunker: Optional[SemanticChunker] = None


def _get_semantic_chunker() -> SemanticChunker:
    """
    Initializes and returns the SemanticChunker with the HuggingFace 'all-MiniLM-L6-v2'
    embedding model, configured with the 95th percentile similarity threshold.
    Uses lazy loading to cache the initialized components at the module level.
    """
    global _embeddings, _chunker
    if _embeddings is None:
        logger.info("Loading HuggingFaceEmbeddings model: 'all-MiniLM-L6-v2'...")
        _embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            encode_kwargs={"normalize_embeddings": True}
        )
    if _chunker is None:
        logger.info("Initializing SemanticChunker with percentile threshold (95)...")
        _chunker = SemanticChunker(
            embeddings=_embeddings,
            breakpoint_threshold_type="percentile",
            breakpoint_threshold_amount=95
        )
    return _chunker


def clean_page_text(text: str) -> str:
    """
    Cleans up excessive whitespace and normalizes text structure page-by-page.
    Consecutive horizontal spaces/tabs are collapsed to a single space.
    Excessive consecutive newlines (3 or more) are collapsed to exactly 2 newlines
    to preserve logical paragraph boundaries.
    """
    if not text:
        return ""
    # Normalize line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Replace 3 or more consecutive newlines with 2 newlines (keeps paragraph separation)
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Replace multiple horizontal spaces/tabs with a single space
    text = re.sub(r"[ \t]+", " ", text)
    # Clean up spaces around newlines
    text = re.sub(r" \n", "\n", text)
    text = re.sub(r"\n ", "\n", text)
    return text.strip()


def ingest_and_chunk_pdf(file_path: str) -> List[Dict[str, Any]]:
    """
    Parses a PDF page-by-page, cleans whitespace, and splits it into semantically
    unified chunks with strict metadata mapping.

    Args:
        file_path (str): The absolute or relative path to the PDF document.

    Returns:
        List[Dict[str, Any]]: A list of chunk dictionaries, each containing:
            - 'chunk_id': zero-padded sequencing string (e.g., 'chunk_0001')
            - 'source_doc': filename of the PDF
            - 'page_number': 1-indexed page number of the original text
            - 'text': the extracted chunk text content
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Specified PDF file not found: {file_path}")
    if not file_path.lower().endswith(".pdf"):
        raise ValueError(f"File must be a PDF document. Provided: {file_path}")

    source_doc = os.path.basename(file_path)
    logger.info(f"Starting ingestion for file: {source_doc}")

    # 1. Parsing layout-aware page structures
    try:
        loader = PyPDFLoader(file_path)
        pages = loader.load()
    except Exception as e:
        logger.error(f"Failed to load or parse the PDF document '{source_doc}': {e}")
        raise RuntimeError(f"PDF parsing error: {e}") from e

    if not pages:
        logger.warning(f"No pages could be extracted from PDF: {source_doc}")
        return []

    # 2. Get/Initialize the Semantic Chunker
    try:
        chunker = _get_semantic_chunker()
    except Exception as e:
        logger.error(f"Failed to initialize embedding model or semantic chunker: {e}")
        raise RuntimeError(f"Semantic Chunking initialization error: {e}") from e

    result_chunks: List[Dict[str, Any]] = []
    chunk_counter = 1

    # 3. Ingestion Processing and Page-by-Page chunking
    for idx, page in enumerate(pages):
        # Extract page number metadata (safely defaulting to 1-indexed page loop value)
        page_num = page.metadata.get("page", idx) + 1
        
        # Clean excessive whitespace keeping paragraphs intact
        cleaned_text = clean_page_text(page.page_content)
        if not cleaned_text:
            logger.debug(f"Skipping page {page_num} since it is empty after cleaning.")
            continue

        # 4. Perform Semantic Chunking per page
        try:
            # Semantic splits on the cleaned text of the current page
            page_chunks = chunker.split_text(cleaned_text)
        except Exception as e:
            logger.warning(
                f"Semantic chunking failed on page {page_num} in '{source_doc}' due to: {e}. "
                f"Falling back to single chunk for this page."
            )
            page_chunks = [cleaned_text]

        # 5. Build structured dictionary results with strict metadata mapping
        for chunk_text in page_chunks:
            chunk_text = chunk_text.strip()
            if not chunk_text:
                continue

            result_chunks.append({
                "chunk_id": f"chunk_{chunk_counter:04d}",
                "source_doc": source_doc,
                "page_number": page_num,
                "text": chunk_text
            })
            chunk_counter += 1

    logger.info(
        f"Ingestion completed for '{source_doc}'. "
        f"Generated {len(result_chunks)} semantic chunks across {len(pages)} pages."
    )
    return result_chunks


if __name__ == "__main__":
    import sys
    
    print("-" * 60)
    print("DocuMind Ingestion & Smart Chunking (Phase 1) Usage Example")
    print("-" * 60)
    
    # Example usage:
    # If no file is provided as argument, we print usage information.
    if len(sys.argv) < 2:
        print("Usage: python ingestion.py <path_to_pdf>")
        print("\nTo test, provide a path to any PDF file. Example:")
        print("  python ingestion.py sample.pdf")
    else:
        target_pdf = sys.argv[1]
        try:
            chunks = ingest_and_chunk_pdf(target_pdf)
            print(f"\nSuccessfully processed! Total chunks generated: {len(chunks)}")
            
            # Print first few chunks as a sample
            sample_size = min(3, len(chunks))
            if sample_size > 0:
                print(f"\n--- Showing first {sample_size} chunks ---")
                for i in range(sample_size):
                    chunk = chunks[i]
                    print(f"\n[{chunk['chunk_id']}] Page {chunk['page_number']} | Source: {chunk['source_doc']}")
                    # Limit length of printed text for display
                    snippet = chunk['text'][:150] + "..." if len(chunk['text']) > 150 else chunk['text']
                    print(f"Content: {snippet}")
            print("-" * 60)
        except Exception as err:
            print(f"\nError occurred: {err}", file=sys.stderr)
            sys.exit(1)
