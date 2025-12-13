# Dockerfile for Manual ML Model Deployment
# This demonstrates all the configuration needed without Truss

# Choose base image - need to pick correct Python version
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Set environment variables
# Need to configure these manually for ML workloads
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TRANSFORMERS_CACHE=/app/.cache \
    HF_HOME=/app/.cache

# Install system dependencies
# Often need to debug which system packages are required
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for Docker layer caching)
COPY requirements.txt .

# Install Python dependencies
# This can fail due to version conflicts - need to debug manually
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app.py .

# Create cache directory for model weights
RUN mkdir -p /app/.cache

# Expose port - need to remember to match with app.py
EXPOSE 8080

# Health check - need to configure manually
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Run the application
# In production: configure workers, timeouts, etc.
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "1", "--timeout", "120", "app:app"]
