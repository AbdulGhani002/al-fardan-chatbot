FROM python:3.11-slim

WORKDIR /app

# System deps:
# - build-essential: scikit-learn wheels + bs4 occasionally need C deps
# - curl: healthcheck probe
# - libgomp1: onnxruntime requires libgomp (OpenMP) at runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the fastembed ONNX model at image-build time. This puts
# the ~130MB model (BAAI/bge-small-en-v1.5) into the image rather than
# the container runtime, so cold-start is instant and production never
# hits an outbound HTTPS failure during boot. Cache lands at
# /root/.cache/fastembed/ which is already on PATH for fastembed.
RUN python -c "from fastembed import TextEmbedding; \
    TextEmbedding(model_name='BAAI/bge-small-en-v1.5'); \
    print('[build] fastembed model cached')"

COPY . .

# Build the TF-IDF + dense index at image-build time so cold-start is fast.
# dense build embeds the full KB (~2000 entries × 384 dims) and pickles;
# typical time 15-30s on CPU. `|| true` so a transient failure still
# ships a container — it'll rebuild in-memory on first request.
RUN python -m app.train.build_index || true

EXPOSE 8001

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://127.0.0.1:8001/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001"]
