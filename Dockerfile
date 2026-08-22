# Container for Hugging Face Spaces (sdk: docker). Also runs anywhere else.
#
# Three deliberate choices:
#   - two stages, because hnswlib ships no manylinux wheel and has to be
#     compiled; the compiler has no business in the runtime image;
#   - the encoder is baked in at build time, so a cold start does not spend
#     ~120 MB of download before answering the first question;
#   - the prebuilt index is copied in rather than built, because ingest takes
#     ~10 minutes of embedding and does not belong in an image build.

FROM python:3.12-slim AS build
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /src
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir --prefix=/install .

FROM python:3.12-slim
COPY --from=build /install /usr/local

RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    HF_HOME=/home/user/.cache/huggingface \
    PYTHONUNBUFFERED=1
WORKDIR /app

# Bake the ONNX encoder into the image (runs as `user`, so the cache is readable).
RUN python -c "from vrag.index.embedder import get_embedder; get_embedder()"

COPY --chown=user:user data/index ./data/index

# Spaces routes to 7860. INDEX_DIR is absolute because the installed package no
# longer sits two levels under the repo root the way a source checkout does.
ENV HOST=0.0.0.0 \
    PORT=7860 \
    INDEX_DIR=/app/data/index \
    LOG_JSON=true
EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s \
    CMD python -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:7860/api/v1/health')"

CMD ["vrag", "serve"]
