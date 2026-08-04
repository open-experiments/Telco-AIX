FROM registry.access.redhat.com/ubi9/python-311:latest AS base
USER root

# Install system dependencies
RUN dnf update -y && \
    dnf install -y xz && \
    dnf clean all

# Install Python dependencies for vLLM compatibility
RUN pip install --upgrade pip && \
    pip install huggingface-hub transformers

# Copy the compressed model and extraction script
COPY model/qwen-finetuned-model.tar.xz /tmp/
COPY extract_model.py .

# Set environment variables
ENV COMPRESSED_MODEL=/tmp/qwen-finetuned-model.tar.xz
ENV MODEL_DIR=/models

# Extract and organize model files
RUN python extract_model.py

# Verify model structure
RUN ls -la /models/ && \
    test -f /models/config.json && \
    test -f /models/tokenizer.json && \
    echo "Model verification successful"

# Final image containing only the essential model files
FROM registry.access.redhat.com/ubi9/ubi-micro:9.8

# Copy the extracted model files from the base container
COPY --from=base /models /models

# Set proper ownership and permissions
USER 1001

# Add labels for container metadata
LABEL name="qwen-finetuned-intent-classifier" \
      version="1.0" \
      description="Fine-tuned Qwen 4B model for intent classification" \
      maintainer="fenar@yahoo.com"
