FROM python:3.11-slim

WORKDIR /app

# System deps for scikit-learn + bs4
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Build the TF-IDF index at image-build time so cold-start is fast.
# KB edits post-deploy trigger /admin/reindex which rebuilds in-place.
RUN python -m app.train.build_index || true

EXPOSE 8001

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://127.0.0.1:8001/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001"]
