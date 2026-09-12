---
title: Funes Unified Agent Memory
sdk: docker
app_port: 7860
---

# Funes unified memory service

This Space exposes the durable HTTP API backed by the private Hub dataset in
`FUNES_MEMORY` (or `FUNES_STORAGE_REPO`).  The dataset contains compressed raw
source shards and append-only deltas; the Space rebuilds SQLite FTS/BM25 on
restart.  Raw text is retained as the source of truth; `retrieval_text` is a
derived shadow and is never returned unless `RETURN_RETRIEVAL_TEXT=true`.

Public probes:

* `GET /health`
* `GET /ready`

Bearer-authenticated endpoints:

* `POST /ingest` — idempotent source-identity upsert, durable only after Hub upload
* `POST /search` or `/recall` — unified Codex/Pi/Claude search with optional filters
* `POST /get` or `GET /get?id=...` — original raw record
* `GET /sources`, `GET /sync/status`
* `POST /reindex`, `POST /sync`

Set Space Secrets (never commit values): `HF_TOKEN` and `FUNES_API_TOKEN`.
Set Variables: `FUNES_MEMORY=bolikoto/funes-memory-data`,
`FUNES_REQUIRE_DURABLE_ACK=true`, `FUNES_ALLOW_EMPTY_REMOTE=false`,
`FUNES_RESTORE_BATCH=5000`, and optionally the OpenAI-compatible translation
variables `TRANSLATION_BASE_URL`, `TRANSLATION_API_KEY`, `TRANSLATION_MODEL`.
