FROM python:3.12-slim-trixie
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
ARG FUNES_SOURCE_REV=57e9d8179203b7152ee2336d534046471c4290a2
ARG FUNES_BINARY_SHA256=5af4859dd8a76066594c50cea207437aaaa4889b7476cdcb451ecf7a3aa11999
ARG FUNES_ARCHIVE_SHA256=346da45929ac747b23f67915d2e01f0ccdd425216129419feb6e22b4c6a2df8f
ARG FUNES_BINARY_URL=https://raw.githubusercontent.com/pjpjq/funes/7d452a2d90fb3a10f3e722204bddc9441be75446/space/bin/funes-x86_64-linux.bin.zst
LABEL org.opencontainers.image.source="https://github.com/pjpjq/funes" \
      org.opencontainers.image.revision="${FUNES_SOURCE_REV}"
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl zstd libstdc++6 \
    && curl --fail --show-error --location --retry 5 --retry-all-errors --connect-timeout 15 --max-time 120 \
        https://raw.githubusercontent.com/trufflesecurity/trufflehog/main/scripts/install.sh \
        | bash -s -- -b /usr/local/bin v3.95.5 \
    && test -x /usr/local/bin/trufflehog \
    && rm -rf /var/lib/apt/lists/*
RUN curl --fail --show-error --location --retry 5 --retry-all-errors --connect-timeout 15 --max-time 300 \
        "${FUNES_BINARY_URL}" -o /tmp/funes.bin.zst \
    && printf '%s  %s\n' "${FUNES_ARCHIVE_SHA256}" /tmp/funes.bin.zst | sha256sum -c - \
    && zstd -q -d /tmp/funes.bin.zst -o /usr/local/bin/funes \
    && printf '%s  %s\n' "${FUNES_BINARY_SHA256}" /usr/local/bin/funes | sha256sum -c - \
    && chmod 0755 /usr/local/bin/funes \
    && /usr/local/bin/funes --version \
    && rm /tmp/funes.bin.zst
COPY space/server.py /app/server.py
COPY space/canonical_ab.py /app/space/canonical_ab.py
COPY service /app/service
RUN pip install --no-cache-dir -r /app/service/requirements.txt
ENV FUNES_HOME=/data/.funes \
    HF_HOME=/data/huggingface \
    HF_HUB_CACHE=/data/huggingface/hub \
    FUNES_HUB_CACHE_BUCKET=bolikoto/funes-runtime-cache \
    FUNES_HUB_CACHE_MOUNT=/mnt/funes-hub-cache \
    FUNES_HUB_CACHE_INTERVAL=300 \
    FUNES_HUB_CACHE_MAX_BYTES=34359738368 \
    PYTHONUNBUFFERED=1 \
    PORT=7860 \
    FUNES_EMBEDDING_PROVIDER=voyage \
    FUNES_EMBEDDING_MODEL=voyage-4-lite \
    FUNES_EMBEDDING_DIMENSIONS=1024 \
    FUNES_EMBEDDING_SCHEMA_VERSION=2 \
    FUNES_RERANK_PROVIDER=none \
    FUNES_RERANK_MODEL=rerank-3-lite \
    FUNES_NATIVE_FALLBACK=false \
    FUNES_RETRIEVAL_LANGUAGE_MODE=raw \
    FUNES_CANONICAL_INDEX_ACTIVE_INTERVAL=1 \
    FUNES_CANONICAL_INDEX_IDLE_INTERVAL=300 \
    FUNES_CANONICAL_INDEX_BATCH=512 \
    FUNES_CANONICAL_INDEX_REQUEST_ROWS=64 \
    FUNES_CANONICAL_INDEX_MAX_CHARS=192000 \
    FUNES_CANONICAL_INDEX_MIN_REQUEST_INTERVAL=1 \
    FUNES_VOYAGE_CONCURRENCY=4 \
    FUNES_VOYAGE_MAX_REQUEST_TOKENS=9000 \
    FUNES_VOYAGE_TOKENS_PER_MINUTE=120000 \
    FUNES_VOYAGE_MIN_REQUEST_INTERVAL=0
RUN mkdir -p /data/.funes/sources
WORKDIR /app
EXPOSE 7860
CMD ["python", "/app/server.py"]
