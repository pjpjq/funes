# Funes HTTP compatibility service

`service/server.py` remains a standard-library compatibility implementation for
local tests and migrations. It is **not** the production retrieval engine. The
Space/Docker entry point is `space/server.py`, a thin bridge over native Funes.
Native Funes owns Lance vector search, BM25, reranking, recency, neighbors and
the TruffleHog push gate.

## API

- `GET /health`, `GET /ready` (no auth)
- Bearer-authenticated `POST /ingest`, `/search`, `/recall`, `/get`, `/reindex`, `/sync`
- Bearer-authenticated `GET /sources`, `/sync/status`

Set `FUNES_AUTH_TOKEN` in local compatibility deployments. Ingest documents with
`raw_text` and optional identity/version/metadata fields. Responses include
`raw_text`; logs never include authorization or document bodies.

For production use `FUNES_MEMORY` with the native bridge and `HF_TOKEN`; the HF
Hub Lance dataset is the durable source of truth.

Chinese queries use the optional OpenAI-compatible translation endpoint configured by `TRANSLATION_BASE_URL`, `TRANSLATION_API_KEY`, and `TRANSLATION_MODEL`. Failed translation falls back to the original query.
