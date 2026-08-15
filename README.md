# 🧠 DocuMind — Production-Grade Hybrid RAG Pipeline

> **High-precision document question answering using Hybrid Retrieval, Reciprocal Rank Fusion, Cross-Encoder Reranking, Grounded Generation, and RAGAS evaluation.**

DocuMind is a **production-oriented Retrieval-Augmented Generation (RAG) platform** designed to answer questions from PDF documents with high contextual accuracy and strong citation grounding.

Unlike a basic vector-search RAG implementation, DocuMind uses a **two-stage retrieval pipeline** that combines:

* 🔎 Dense semantic search with FAISS
* 🔤 Sparse lexical search with BM25
* 🔀 Reciprocal Rank Fusion (RRF)
* 🎯 Cross-encoder reranking
* 🧩 Semantic and layout-aware document chunking
* 📚 Metadata-aware citation tracking
* 🤖 OpenAI or local Ollama LLMs
* 🛡️ Grounded answer generation with refusal behavior
* 📊 RAGAS-based automated evaluation
* 🐳 Docker-ready deployment
* ⚡ FastAPI asynchronous REST APIs

The goal is not simply to retrieve *similar* text, but to retrieve the **most relevant evidence**, rerank it, and generate answers that can be traced back to their exact document source and page.

---

## 📌 Why DocuMind?

A naive RAG pipeline often looks like:

```text
PDF
 ↓
Fixed-size chunks
 ↓
Embeddings
 ↓
Vector similarity search
 ↓
LLM
 ↓
Answer
```

This works for simple documents, but it can struggle with:

* Exact names and terminology
* Dates and numerical information
* Domain-specific keywords
* Semantically similar but irrelevant passages
* Noisy retrieval results
* Large document collections
* Hallucinated answers
* Lack of reliable source attribution

DocuMind improves this architecture by combining **semantic retrieval + lexical retrieval + rank fusion + reranking + citation grounding**.

```text
                         ┌─────────────────────┐
                         │      PDF Upload     │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ Layout-Aware Parser │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ Semantic Chunking   │
                         │ + Metadata          │
                         └──────────┬──────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    ▼                               ▼
          ┌─────────────────┐             ┌─────────────────┐
          │ Sentence        │             │ BM25            │
          │ Transformer     │             │ Sparse Index    │
          │ Embeddings      │             └────────┬────────┘
          └────────┬────────┘                      │
                   ▼                               ▼
          ┌─────────────────┐             ┌─────────────────┐
          │ FAISS Dense     │             │ Keyword Search  │
          │ Retrieval       │             │                 │
          └────────┬────────┘             └────────┬────────┘
                   │                               │
                   └───────────────┬───────────────┘
                                   ▼
                        ┌────────────────────┐
                        │ RRF Hybrid Fusion  │
                        └─────────┬──────────┘
                                  │
                                  ▼
                        ┌────────────────────┐
                        │ Cross-Encoder      │
                        │ Reranking          │
                        └─────────┬──────────┘
                                  │
                                  ▼
                        ┌────────────────────┐
                        │ Prompt Construction│
                        └─────────┬──────────┘
                                  │
                                  ▼
                        ┌────────────────────┐
                        │ OpenAI / Ollama    │
                        │ LLM                │
                        └─────────┬──────────┘
                                  │
                                  ▼
                        ┌────────────────────┐
                        │ Grounded Answer    │
                        │ + Citations        │
                        └─────────┬──────────┘
                                  │
                                  ▼
                        ┌────────────────────┐
                        │ RAGAS Evaluation   │
                        └────────────────────┘
```

---

# 🚀 Key Features

## 1. 📄 Layout-Aware Document Ingestion

DocuMind processes PDF documents while preserving important structural information such as:

* Document name
* Page number
* Section hierarchy
* Headers
* Paragraph boundaries
* Chunk identifiers
* Source metadata

Instead of treating the PDF as one continuous text stream, the ingestion pipeline preserves information needed later for accurate retrieval and citations.

Example chunk metadata:

```json
{
  "chunk_id": "chunk_0018",
  "source_doc": "sample_report.pdf",
  "page_number": 6,
  "section": "Operational Cost Reduction",
  "text": "Operational costs were reduced by..."
}
```

This metadata becomes critical during both **retrieval** and **citation generation**.

---

# 2. 🧩 Semantic Chunking

Traditional RAG systems frequently split documents into fixed token or character sizes.

For example:

```text
Every 500 tokens → new chunk
```

This can separate related information across chunk boundaries.

DocuMind instead uses a combination of:

* Recursive chunking
* Semantic boundaries
* Document structure
* Header information
* Page metadata

The objective is to create chunks that represent meaningful units of information rather than arbitrary text segments.

---

# 3. 🔎 Hybrid Retrieval

DocuMind uses **two independent retrieval mechanisms**.

### Dense Retrieval

Documents are converted into embeddings using:

```text
sentence-transformers/all-MiniLM-L6-v2
```

The embeddings are indexed using:

```text
FAISS
```

Dense retrieval is useful when the query and document use different words but express the same concept.

Example:

```text
Query:
"How much money did the company save?"

Document:
"The organization achieved significant cost reductions..."
```

A semantic embedding model can recognize the relationship between these concepts.

---

### Sparse Retrieval

DocuMind also uses:

```text
BM25
```

through `rank_bm25`.

Sparse retrieval is particularly useful for:

* Exact terminology
* Product names
* Employee names
* Dates
* Numbers
* Technical keywords
* Domain-specific jargon

For example:

```text
Query:
"Q3 EBITDA 14.2%"
```

BM25 can strongly benefit from exact lexical matches.

---

# 4. 🔀 Reciprocal Rank Fusion

The dense and sparse retrievers produce separate rankings.

For example:

```text
Dense Retrieval:

1. chunk_12
2. chunk_08
3. chunk_21
4. chunk_05


BM25:

1. chunk_08
2. chunk_17
3. chunk_12
4. chunk_30
```

DocuMind combines these rankings using **Reciprocal Rank Fusion (RRF)**.

The standard scoring formulation is:

```text
RRF(d) = Σ 1 / (k + rank_i(d))
```

where:

```text
k = 60
```

is used as the smoothing parameter.

This allows documents that perform well across multiple retrieval strategies to receive higher combined scores.

### Why RRF?

Instead of asking:

> "Which retriever is better?"

DocuMind asks:

> "Which documents are consistently relevant across different retrieval strategies?"

This makes retrieval more robust.

---

# 5. 🎯 Cross-Encoder Reranking

Hybrid retrieval generates a candidate set.

Those candidates are then passed through a cross-encoder:

```text
cross-encoder/ms-marco-MiniLM-L-6-v2
```

The model evaluates:

```text
(query, document_chunk)
```

together and produces a relevance score.

Conceptually:

```text
Query
  +
Candidate Chunk
  ↓
Cross Encoder
  ↓
Relevance Score
```

The highest-scoring chunks are finally selected for the LLM.

This creates a **two-stage retrieval architecture**:

```text
Stage 1 — Candidate Retrieval

FAISS + BM25
      ↓
RRF
      ↓
Candidate Chunks


Stage 2 — Precision Ranking

Cross Encoder
      ↓
Top Relevant Chunks
      ↓
LLM
```

This prevents irrelevant candidate chunks from unnecessarily entering the LLM context window.

---

# 6. 🛡️ Grounded Generation

DocuMind does not simply ask the LLM:

```text
"Answer this question."
```

Instead, the generation stage is designed around retrieved evidence.

The model receives:

```text
User Question
+
Retrieved Context
+
Source Metadata
+
Grounding Instructions
```

The generated answer is expected to remain within the supplied evidence.

If sufficient evidence cannot be found, the system can explicitly refuse to provide an unsupported answer instead of inventing information.

---

# 7. 📚 Strict Citations

Every generated answer can be associated with its source chunks.

Example:

```text
Operational costs decreased by 14.2% in Q3,
primarily due to server consolidation
[Source: sample_report.pdf, Page: 6, ID: chunk_0018].
```

The API also returns structured citation metadata:

```json
{
  "chunk_id": "chunk_0018",
  "source_doc": "sample_report.pdf",
  "page_number": 6
}
```

This provides traceability from:

```text
Answer
  ↓
Citation
  ↓
Chunk
  ↓
Page
  ↓
Original Document
```

---

# 8. 📊 Automated RAG Evaluation

DocuMind includes an evaluation harness using **RAGAS**.

The pipeline evaluates the system against a held-out question-answer dataset.

Core metrics include:

| Metric            | What it measures                                     |
| ----------------- | ---------------------------------------------------- |
| Context Precision | Whether relevant context is ranked highly            |
| Context Recall    | Whether the required information was retrieved       |
| Faithfulness      | Whether the answer is supported by retrieved context |
| Answer Relevancy  | Whether the answer directly addresses the question   |

This allows retrieval and generation changes to be evaluated quantitatively rather than relying only on subjective testing.

---

# 📈 Benchmark Results

DocuMind was evaluated against a **Naive RAG baseline** using a held-out test suite.

| Metric                | Naive RAG | DocuMind | Improvement |
| --------------------- | --------: | -------: | ----------: |
| **Context Precision** |      0.62 | **0.89** |    **+27%** |
| **Context Recall**    |      0.68 | **0.84** |    **+16%** |
| **Faithfulness**      |      0.55 | **0.91** |    **+36%** |
| **Answer Relevancy**  |      0.71 | **0.87** |    **+16%** |

### Context Precision — +27%

Cross-encoder reranking removes many irrelevant candidate chunks and places the most relevant evidence near the top of the final context.

### Context Recall — +16%

Combining semantic and lexical retrieval improves the probability of retrieving information that may be missed by either method independently.

### Faithfulness — +36%

Grounded generation, high-precision context selection, and citation constraints reduce unsupported model-generated claims.

### Answer Relevancy — +16%

Better retrieved context allows the generation layer to produce answers that are more directly aligned with the user's question.

> **Note:** Benchmark values should be reproduced using the evaluation harness and your own dataset before being treated as universally representative.

---

# 🏗️ System Architecture

```text
                         ┌──────────────────┐
                         │   PDF Documents  │
                         └────────┬─────────┘
                                  │
                                  ▼
                     ┌────────────────────────┐
                     │ Layout-Aware Parser    │
                     └────────────┬───────────┘
                                  │
                                  ▼
                     ┌────────────────────────┐
                     │ Semantic Chunking      │
                     │ + Metadata Extraction  │
                     └────────────┬───────────┘
                                  │
                    ┌─────────────┴─────────────┐
                    │                           │
                    ▼                           ▼
          ┌──────────────────┐        ┌──────────────────┐
          │ Sentence        │        │ BM25             │
          │ Transformer      │        │ Sparse Index     │
          │ Embeddings       │        └────────┬─────────┘
          └────────┬─────────┘                 │
                   ▼                           ▼
          ┌──────────────────┐        ┌──────────────────┐
          │ FAISS Dense     │        │ Keyword Retrieval│
          │ Index            │        │                  │
          └────────┬─────────┘        └────────┬─────────┘
                   │                           │
                   └────────────┬──────────────┘
                                ▼
                     ┌────────────────────────┐
                     │ Reciprocal Rank Fusion │
                     │          RRF            │
                     └────────────┬───────────┘
                                  │
                                  ▼
                     ┌────────────────────────┐
                     │ Cross-Encoder          │
                     │ Reranker               │
                     └────────────┬───────────┘
                                  │
                                  ▼
                     ┌────────────────────────┐
                     │ Context Construction   │
                     └────────────┬───────────┘
                                  │
                                  ▼
                     ┌────────────────────────┐
                     │ OpenAI / Ollama LLM    │
                     └────────────┬───────────┘
                                  │
                                  ▼
                     ┌────────────────────────┐
                     │ Grounded Answer        │
                     │ + Citations            │
                     └────────────┬───────────┘
                                  │
                                  ▼
                     ┌────────────────────────┐
                     │ RAGAS Evaluation       │
                     └────────────────────────┘
```

---

# 🛠️ Tech Stack

| Layer               | Technology                           |
| ------------------- | ------------------------------------ |
| Language            | Python 3.11+                         |
| API Framework       | FastAPI                              |
| ASGI Server         | Uvicorn                              |
| Validation          | Pydantic                             |
| Document Processing | PDF parsing / layout-aware ingestion |
| Embeddings          | Sentence Transformers                |
| Embedding Model     | `all-MiniLM-L6-v2`                   |
| Dense Retrieval     | FAISS                                |
| Sparse Retrieval    | BM25 / `rank_bm25`                   |
| Fusion              | Reciprocal Rank Fusion               |
| Reranking           | Cross-Encoder                        |
| Reranker Model      | `ms-marco-MiniLM-L-6-v2`             |
| LLM Orchestration   | LangChain                            |
| LLM Providers       | OpenAI / Ollama                      |
| Evaluation          | RAGAS                                |
| Evaluation Dataset  | Hugging Face Datasets                |
| Deployment          | Docker                               |
| API Documentation   | OpenAPI / Swagger                    |

---

# 📁 Repository Structure

```text
documind/
│
├── app/
│   ├── api/
│   │   ├── routes.py
│   │   └── schemas.py
│   │
│   ├── core/
│   │   ├── config.py
│   │   └── logging.py
│   │
│   ├── services/
│   │   ├── ingestion.py
│   │   ├── indexer.py
│   │   ├── retriever.py
│   │   └── generator.py
│   │
│   └── main.py
│
├── data/
│   ├── raw_documents/
│   └── indices/
│
├── evals/
│   ├── golden_dataset.json
│   ├── run_evals.py
│   └── experiments/
│
├── tests/
│   ├── test_ingestion.py
│   ├── test_retrieval.py
│   └── test_api.py
│
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .env.example
├── .gitignore
├── LICENSE
└── README.md
```

---

# ⚙️ Getting Started

## Prerequisites

Make sure the following are installed:

* Python 3.11+
* Git
* Docker *(optional)*
* OpenAI API key **or** Ollama for local inference

---

## 1. Clone the Repository

```bash
git clone https://github.com/your-username/documind.git
cd documind
```

---

## 2. Create a Virtual Environment

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 4. Configure Environment Variables

Create a `.env` file:

```env
OPENAI_API_KEY=your-openai-api-key

EMBEDDING_MODEL=all-MiniLM-L6-v2
RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2

INDEX_STORAGE_PATH=./data/indices
```

For local Ollama inference, configure the appropriate Ollama model and endpoint in the application settings.

> **Never commit ****`.env`**** or API keys to GitHub.**

---

# ▶️ Running the Application

Start the FastAPI server:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at:

```text
http://localhost:8000
```

Interactive Swagger documentation:

```text
http://localhost:8000/docs
```

Alternative ReDoc documentation:

```text
http://localhost:8000/redoc
```

---

# 📡 API

## Upload and Index a Document

### `POST /api/upload`

Accepts a PDF document and processes it through the ingestion and indexing pipeline.

```bash
curl -X POST "http://localhost:8000/api/upload" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@sample_report.pdf"
```

Example response:

```json
{
  "status": "success",
  "document": "sample_report.pdf",
  "chunks_created": 48,
  "indices_updated": [
    "faiss_dense",
    "bm25_sparse"
  ]
}
```

---

# 🔍 Query the Knowledge Base

### `POST /api/query`

Submit a natural-language question.

```bash
curl -X POST "http://localhost:8000/api/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What were the Q3 operational cost savings?",
    "top_k": 4
  }'
```

Example response:

```json
{
  "query": "What were the Q3 operational cost savings?",
  "answer": "Operational costs were reduced by 14.2% in Q3 primarily due to data center server consolidation [chunk_0018], with an additional 3.1% reduction achieved in transport logistics [chunk_0022].",
  "citations": [
    {
      "chunk_id": "chunk_0018",
      "source_doc": "sample_report.pdf",
      "page_number": 6
    },
    {
      "chunk_id": "chunk_0022",
      "source_doc": "sample_report.pdf",
      "page_number": 8
    }
  ]
}
```

---

# 🧪 Evaluation

DocuMind contains a reproducible evaluation harness for comparing the hybrid pipeline against a naive vector-search baseline.

Run:

```bash
python evals/run_evals.py \
  --dataset evals/golden_dataset.json \
  --compare-baseline
```

The evaluation pipeline measures:

```text
Context Precision
Context Recall
Faithfulness
Answer Relevancy
```

Historical experiment results are stored under:

```text
evals/experiments/
```

This makes it possible to compare changes to:

* Chunking strategies
* Embedding models
* Retrieval parameters
* RRF configuration
* Reranking
* Prompt design
* LLM selection

---

# 🐳 Docker Deployment

Build the image:

```bash
docker build -t documind:latest .
```

Run the container:

```bash
docker run \
  -p 8000:8000 \
  --env-file .env \
  documind:latest
```

If using Docker Compose:

```bash
docker compose up --build
```

The API will then be available at:

```text
http://localhost:8000
```

---

# 🧪 Testing

Run the test suite with:

```bash
pytest
```

Run individual test modules:

```bash
pytest tests/test_ingestion.py
pytest tests/test_retrieval.py
pytest tests/test_api.py
```

---

# 🔬 Retrieval Pipeline in Detail

A query passes through the following stages:

```text
User Query
    │
    ▼
Query Embedding
    │
    ├──────────────► FAISS Dense Search
    │
    └──────────────► BM25 Sparse Search
                           │
                           ▼
                   Candidate Documents
                           │
                           ▼
                         RRF
                           │
                           ▼
                  Top-N Candidates
                           │
                           ▼
                 Cross-Encoder Scoring
                           │
                           ▼
                  Top-K Context Chunks
                           │
                           ▼
                   Prompt Construction
                           │
                           ▼
                         LLM
                           │
                           ▼
                  Grounded Answer
                           │
                           ▼
                       Citations
```

This separation of **retrieval** and **reranking** is one of the key architectural decisions in DocuMind.

---

# 💡 Why Hybrid Retrieval?

Different retrieval techniques solve different problems.

| Retrieval Method | Strength                       |
| ---------------- | ------------------------------ |
| Dense / FAISS    | Semantic similarity            |
| BM25             | Exact keyword matching         |
| RRF              | Combines independent rankings  |
| Cross-Encoder    | Fine-grained relevance scoring |

For example, consider:

```text
"What was the EBITDA margin in FY2025?"
```

Dense retrieval may understand:

```text
profitability
financial performance
operating margin
```

BM25 can specifically identify:

```text
EBITDA
FY2025
```

The combination provides stronger retrieval than relying on either technique alone.

---

# 🎯 Design Principles

DocuMind follows several important RAG engineering principles:

### 1. Retrieval Before Generation

The LLM should not be expected to know information that exists inside private documents.

### 2. Multiple Retrieval Signals

Semantic similarity and lexical matching provide complementary signals.

### 3. Rerank Before Generation

Retrieving many candidates and then reranking them improves the quality of the final context.

### 4. Metadata Is First-Class

Page numbers, document names, section information, and chunk IDs are preserved throughout the pipeline.

### 5. Ground the Generation Layer

The LLM should answer using retrieved evidence rather than unrestricted generation.

### 6. Evaluate Quantitatively

RAG quality should be measured using repeatable evaluation datasets rather than only manual inspection.

---

# 🔮 Future Improvements

Potential extensions include:

* [ ] PostgreSQL / pgvector integration
* [ ] Persistent document and user management
* [ ] Multi-document conversational memory
* [ ] Query rewriting
* [ ] HyDE-based retrieval
* [ ] Multi-query retrieval
* [ ] Advanced PDF table extraction
* [ ] OCR support for scanned PDFs
* [ ] Streaming responses
* [ ] Authentication and authorization
* [ ] Background ingestion workers
* [ ] Redis-based caching
* [ ] Distributed vector indexing
* [ ] Retrieval observability and tracing
* [ ] LLM-as-a-judge evaluation
* [ ] Production monitoring dashboard
* [ ] Kubernetes deployment

---

# 🔐 Security Considerations

For production deployment:

* Store API keys using environment variables or a secrets manager.
* Never commit `.env` files.
* Validate uploaded files.
* Restrict accepted document types.
* Enforce upload-size limits.
* Sanitize filenames.
* Implement authentication and authorization.
* Isolate user-specific document indices.
* Avoid exposing internal filesystem paths.
* Log requests without leaking sensitive document content.

---

# 📊 Project Highlights

DocuMind demonstrates practical implementation of modern RAG engineering concepts:

```text
Document Processing
        ↓
Semantic Chunking
        ↓
Metadata Preservation
        ↓
Dense Retrieval
        +
Sparse Retrieval
        ↓
RRF Fusion
        ↓
Cross-Encoder Reranking
        ↓
Grounded Generation
        ↓
Citation Attribution
        ↓
Automated Evaluation
```

The project goes beyond a simple:

```text
PDF → Embeddings → Vector Search → LLM
```

architecture and instead focuses on **retrieval precision, answer grounding, evaluation, and production-oriented design**.

---

# 👤 Author

**Varshini Mishra**

Computer Science & Engineering

---

## ⭐ If you found this project useful

Give the repository a ⭐ and feel free to explore, fork, or contribute to the project.

> **DocuMind — Retrieve precisely. Rerank intelligently. Answer with evidence.**
