FROM python:3.11-slim
WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
ARG MODEL_VERSION=unknown
ENV MODEL_VERSION=${MODEL_VERSION}

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src
COPY models/model.joblib ./models/model.joblib
COPY models/feature_list.json ./models/feature_list.json
COPY artifacts/meta.json ./artifacts/meta.json

EXPOSE 8000
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]