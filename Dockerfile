# Base image with vLLM, CUDA, and PyTorch pre-installed
FROM runpod/worker-v1-vllm:latest

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Set working directory
WORKDIR /app

# Copy application code
COPY app/ /app/

# Install additional Python dependencies
RUN pip install --no-cache-dir -r /app/requirements.txt

# Set the RunPod handler entrypoint
ENV RUNPOD_HANDLER="app.handler"

# Pre-download the model to reduce cold start time
# This is optional but highly recommended for production
ARG MODEL_NAME=codefactory4791/intent-classification-qwen
ARG QUANTIZATION=none

# Download model during build (comment out to skip pre-downloading)
RUN python3 -c "\
from transformers import AutoTokenizer, AutoModelForSequenceClassification; \
import os; \
model_name = os.getenv('MODEL_NAME', '${MODEL_NAME}'); \
print(f'Pre-downloading model: {model_name}'); \
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True); \
model = AutoModelForSequenceClassification.from_pretrained(model_name, trust_remote_code=True); \
print('Model pre-download complete'); \
"

# Health check endpoint (optional, for debugging)
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python3 -c "import sys; sys.exit(0)"

# Expose port for debugging (RunPod manages networking in production)
EXPOSE 8000

# The base image already defines CMD to start the worker
# No need to override unless using a custom entrypoint

