# Funes HTTP compatibility service

`service/server.py` provides the source-of-truth sidecar used by the production
Space and remains independently usable for local tests and migrations. The
Space/Docker entry point is `space/server.py`; native Funes still owns Lance vector
search, BM25, reranking, recency and neighbors.

## API

- `GET /health`, `GET /ready` (no auth)
- Bearer-authenticated `POST /ingest`, `/search`, `/recall`, `/get`, `/reindex`, `/sync`, `/sources/check`
- Bearer-authenticated `GET /sources`, `/sync/status`

Set `FUNES_AUTH_TOKEN` in local compatibility deployments. Ingest documents with
`raw_text` and optional identity/version/metadata fields. Responses include
`raw_text`; logs never include authorization or document bodies.
`/sources/check` accepts at most 5,000 source identities and returns only the
ordered `present`/`missing` identity lists; it never returns stored text.

For production, configure `FUNES_STORAGE_REPO`, `HF_TOKEN`, and
`FUNES_STORAGE_KEY`. Snapshot and delta objects are compressed then encrypted with
streaming AES-256-GCM before upload. Plain `raw_text` never enters the Hub repo;
restore fails closed when the key is absent or authentication fails. `FUNES_MEMORY`
remains the separately rebuildable native retrieval index.

Chinese queries use the optional OpenAI-compatible translation endpoint configured by `TRANSLATION_BASE_URL`, `TRANSLATION_API_KEY`, and `TRANSLATION_MODEL`. Failed translation falls back to the original query.
