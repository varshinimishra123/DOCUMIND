import os
import logging
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.lib.styles import getSampleStyleSheet

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DocuMind.Verification")

TEST_PDF_PATH = "test_doc.pdf"


def create_sample_pdf(output_path: str):
    """
    Generates a 3-page sample PDF with distinct semantic topics on each page
    and intentional excessive whitespace to test cleaning and semantic split boundaries.
    """
    logger.info(f"Generating test PDF file at: {output_path}")
    doc = SimpleDocTemplate(output_path, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    # Page 1: Artificial Intelligence and Machine Learning
    story.append(Paragraph("<b>Artificial Intelligence and Machine Learning Introduction</b>", styles['Title']))
    story.append(Spacer(1, 12))
    story.append(Paragraph(
        "Artificial Intelligence (AI) refers to the simulation of human intelligence in machines "
        "that are programmed to think like humans and mimic their actions. The term may also be "
        "applied to any machine that exhibits traits associated with a human mind such as learning "
        "and problem-solving.",
        styles['BodyText']
    ))
    story.append(Spacer(1, 12))
    story.append(Paragraph(
        "Machine Learning (ML) is a subset of AI that focus on building systems that learn—or improve "
        "performance—based on the data they consume. Neural networks and deep learning are advanced subfields "
        "of ML that mimic the biological neural systems of the human brain to process data.",
        styles['BodyText']
    ))
    
    # Introduce page break to separate pages
    story.append(PageBreak())

    # Page 2: Marine Biology and Ecosystems (Distinct topic)
    story.append(Paragraph("<b>Exploring the Mysteries of Marine Biology</b>", styles['Title']))
    story.append(Spacer(1, 12))
    story.append(Paragraph(
        "Marine biology is the scientific study of marine life, organisms in the sea. Given that in biology "
        "many phyla, families and genera have some species that live in the sea and others that live on land, "
        "marine biology classifies species based on the environment rather than on taxonomy.",
        styles['BodyText']
    ))
    story.append(Spacer(1, 12))
    story.append(Paragraph(
        "Coral reefs form some of the most diverse ecosystems on Earth. They occupy less than 0.1% of the "
        "world's ocean surface, yet they provide a home for at least 25% of all marine species, including "
        "fish, mollusks, worms, crustaceans, echinoderms, sponges, tunicates and other cnidarians.",
        styles['BodyText']
    ))

    # Introduce page break to separate pages
    story.append(PageBreak())

    # Page 3: Culinary Arts and Baking (Distinct topic)
    story.append(Paragraph("<b>The Science of Baking the Perfect Chocolate Cake</b>", styles['Title']))
    story.append(Spacer(1, 12))
    story.append(Paragraph(
        "Baking is a precise science that requires careful measurement and understanding of chemical reactions. "
        "When baking a chocolate cake, the interaction between flour, sugar, eggs, fats, and leavening agents "
        "determines the final texture, crumb structure, and rise of the cake.",
        styles['BodyText']
    ))
    story.append(Spacer(1, 12))
    story.append(Paragraph(
        "Preheating the oven is a critical step. The initial blast of heat causes the carbon dioxide gas "
        "produced by the baking powder or soda to expand rapidly, creating tiny pockets of air inside the "
        "batter. This process, known as oven spring, defines the light and fluffy quality of the cake.",
        styles['BodyText']
    ))

    doc.build(story)
    logger.info("Test PDF successfully built.")


def verify():
    # 1. Create the test PDF
    create_sample_pdf(TEST_PDF_PATH)

    try:
        # 2. Import the ingestion function
        logger.info("Importing 'ingest_and_chunk_pdf' from ingestion.py...")
        from ingestion import ingest_and_chunk_pdf

        # 3. Process the test PDF
        logger.info("Executing ingest_and_chunk_pdf...")
        chunks = ingest_and_chunk_pdf(TEST_PDF_PATH)

        # 4. Perform Assertions
        assert isinstance(chunks, list), "Result should be a list"
        assert len(chunks) > 0, "Should generate at least one chunk"
        
        logger.info(f"Generated {len(chunks)} chunks.")
        
        for idx, chunk in enumerate(chunks):
            # Check dictionary structure
            assert isinstance(chunk, dict), f"Chunk {idx} is not a dictionary"
            assert "chunk_id" in chunk, f"Chunk {idx} is missing 'chunk_id'"
            assert "source_doc" in chunk, f"Chunk {idx} is missing 'source_doc'"
            assert "page_number" in chunk, f"Chunk {idx} is missing 'page_number'"
            assert "text" in chunk, f"Chunk {idx} is missing 'text'"

            # Validate metadata values
            expected_id = f"chunk_{idx + 1:04d}"
            assert chunk["chunk_id"] == expected_id, f"Invalid chunk_id sequence: {chunk['chunk_id']} (expected {expected_id})"
            assert chunk["source_doc"] == TEST_PDF_PATH, f"Invalid source_doc: {chunk['source_doc']}"
            assert chunk["page_number"] in [1, 2, 3], f"Invalid page_number: {chunk['page_number']}"
            assert len(chunk["text"].strip()) > 0, "Chunk text should not be empty or whitespace-only"

            # Check whitespace cleanup
            text = chunk["text"]
            assert "   " not in text, "Double spaces or extra whitespace should be cleaned"
            assert "\n\n\n" not in text, "Excessive newlines should be collapsed to maximum of two"

            # Print details of the validated chunk
            print(f"Validated: {chunk['chunk_id']} | Page: {chunk['page_number']} | Length: {len(text)}")
            print(f"Snippet: {text[:100]}...\n")

        print("=" * 60)
        print("ALL INGESTION AND CHUNKING TESTS PASSED SUCCESSFULLY!")
        print("=" * 60)

    finally:
        # Cleanup test document
        if os.path.exists(TEST_PDF_PATH):
            os.remove(TEST_PDF_PATH)
            logger.info(f"Cleaned up test file: {TEST_PDF_PATH}")


if __name__ == "__main__":
    verify()
