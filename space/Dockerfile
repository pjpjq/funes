FROM rust:1.94-bookworm AS build
WORKDIR /src
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl libprotobuf-dev protobuf-compiler \
    && curl -sSfL https://raw.githubusercontent.com/trufflesecurity/trufflehog/main/scripts/install.sh \
        | sh -s -- -b /usr/local/bin v3.95.5

# The HF cpu-basic builder previously stopped during an unbounded parallel Rust
# build, while one job exceeded the platform's cold-build timeout. Two jobs keep
# memory bounded and leave enough parallelism to finish the dependency graph.
ENV CARGO_BUILD_JOBS=2 \
    CARGO_INCREMENTAL=0 \
    CARGO_PROFILE_RELEASE_LTO=off
COPY Cargo.toml Cargo.lock ./
RUN cargo fetch --locked
COPY build.rs ./
COPY src ./src
COPY scripts/automation/funes-index.sh scripts/automation/funes-push.sh ./scripts/automation/
COPY integrations/claude-plugin/.claude-plugin/marketplace.json ./integrations/claude-plugin/.claude-plugin/marketplace.json
COPY integrations/claude-plugin/funes/.claude-plugin/plugin.json ./integrations/claude-plugin/funes/.claude-plugin/plugin.json
COPY integrations/codex/SKILL.md ./integrations/codex/SKILL.md
COPY integrations/pi/index.ts integrations/pi/package.json ./integrations/pi/
RUN cargo build --locked --release --no-default-features --features blas --bin funes

FROM python:3.12-slim-bookworm
COPY --from=build /src/target/release/funes /usr/local/bin/funes
COPY --from=build /usr/local/bin/trufflehog /usr/local/bin/trufflehog
COPY space/server.py /app/server.py
COPY service /app/service
RUN pip install --no-cache-dir -r /app/service/requirements.txt
ENV FUNES_HOME=/data/.funes \
    HF_HOME=/data/huggingface \
    HF_HUB_CACHE=/data/huggingface/hub \
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
    FUNES_CANONICAL_INDEX_IDLE_INTERVAL=300
RUN mkdir -p /data/.funes/sources
WORKDIR /app
EXPOSE 7860
CMD ["python", "/app/server.py"]
