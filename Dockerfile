# Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Environment settings
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MODEL_VERSION=0.1

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and model files
COPY src/ ./src
COPY models/model_v${MODEL_VERSION}.joblib ./models/model_v${MODEL_VERSION}.joblib
COPY models/feature_list.json ./models/feature_list.json

# Expose port and run API
EXPOSE 8000
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
