# Multi-stage build for minimal image size
# Stage 1: Build dependencies
FROM python:3.11-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Pre-download Wav2Vec2 model to cache (reduces cold start time)
RUN python -c "from transformers import Wav2Vec2Model, Wav2Vec2Processor; Wav2Vec2Processor.from_pretrained('facebook/wav2vec2-base-960h'); Wav2Vec2Model.from_pretrained('facebook/wav2vec2-base-960h')"

# Stage 2: Runtime (minimal)
FROM python:3.11-slim

WORKDIR /app

# Install only runtime dependencies (FFmpeg for pydub)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy installed packages from builder
COPY --from=builder /root/.local /root/.local
COPY --from=builder /root/.cache/huggingface /root/.cache/huggingface
ENV PATH=/root/.local/bin:$PATH

# Copy only necessary application files
COPY app.py .
COPY audio/ ./audio/
COPY model/ ./model/
COPY utils/ ./utils/

# Environment
ENV PYTHONUNBUFFERED=1
ENV TRANSFORMERS_CACHE=/root/.cache/huggingface
ENV PRODUCTION=true

EXPOSE 8000

# Use uvicorn directly - simpler and more reliable on Railway
# --timeout-keep-alive 120: Keep connections alive for slow clients
CMD uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000} --timeout-keep-alive 120
