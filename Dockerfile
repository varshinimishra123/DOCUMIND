# ==========================================
# STAGE 1: Build & Package Installation Stage
# ==========================================
FROM python:3.11-slim AS builder

# Set build environment settings
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install basic system build dependencies required for compiling Python extension packages (like rank-bm25 or faiss-cpu components if needed)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment for app dependencies
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy dependency manifests
WORKDIR /build
COPY requirements.txt .

# Install dependencies inside the virtual environment
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt


# ==========================================
# STAGE 2: Clean Runtime Execution Stage
# ==========================================
FROM python:3.11-slim AS runtime

# Set execution environment settings
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH"

# Copy the prepared virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv

# Create a non-root system user and group for security enforcement
RUN groupadd -g 999 docmind && \
    useradd -r -u 999 -g docmind -m -d /home/docmind docmind_user

# Set working directory
WORKDIR /app

# Copy application codebase
COPY ingestion.py storage.py retrieval.py generation.py main.py ./

# Create directories for persistent uploads and indices, and assign ownership to the non-root user
RUN mkdir -p uploads docmind_index && \
    chown -R docmind_user:docmind /app

# Drop root privileges and switch execution context to the non-root user
USER docmind_user

# Expose microservice port
EXPOSE 8000

# Exec array format to start Uvicorn securely
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
