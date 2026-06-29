# Base image: official Python 3.11 slim variant
# "slim" excludes unnecessary OS packages, keeping the image smaller.
# We pin to 3.11 specifically because that's what we developed against --
# Python 3.12+ has some behavioral differences with certain dependencies.
FROM python:3.11-slim

# Set working directory inside the container
WORKDIR /app

# Install system dependencies that some Python packages need to compile.
# --no-install-recommends keeps the image lean by skipping optional packages.
# We clean up the apt cache in the same layer to avoid bloating image size
# (each RUN creates a new layer; combining commands minimizes layer count).
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first, before the rest of the code.
# This is a critical Docker optimization: Docker caches each layer.
# If requirements.txt hasn't changed, Docker reuses the cached pip install
# layer even if your code changed -- so you don't reinstall all packages
# on every code change. Copying requirements separately exploits this.
COPY requirements.txt .

# Install Python dependencies
# --no-cache-dir: don't store the pip download cache inside the image,
# keeps image size smaller since we don't need to reinstall during a run.
RUN pip install --no-cache-dir -r requirements.txt

# Ragas 0.3.9 has a bug where it unconditionally imports
# langchain_community.chat_models.vertexai which was removed in
# langchain-community 0.4.x. Create a stub to satisfy the import
# since we never actually use Vertex AI.
RUN python -c "\
import langchain_community.chat_models as cm; \
import os; \
stub = os.path.join(os.path.dirname(cm.__file__), 'vertexai.py'); \
open(stub, 'w').write('# stub\nclass ChatVertexAI: pass\n'); \
print('Vertex AI stub created at', stub)"

# Copy the rest of the project code
# .dockerignore (created below) prevents .env, .venv, chroma_db,
# data/raw etc. from being copied into the image.
COPY . .

# Create directories that need to exist at runtime
# data/raw and chroma_db are mounted as volumes in docker-compose,
# but creating them here ensures the image works standalone too.
RUN mkdir -p data/raw data/processed chroma_db docs/learning_notes

# Environment variables with safe defaults
# PYTHONUNBUFFERED: print statements appear immediately in Docker logs
# rather than being buffered -- critical for seeing pipeline progress.
# PYTHONDONTWRITEBYTECODE: don't create .pyc files inside the container.
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Default command: run the RAG pipeline with tracing
# Override at runtime: docker run glp1 python src/evaluate.py
CMD ["python", "src/run_with_tracing.py"]