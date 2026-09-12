# Funes HTTP Space

A low-dependency HTTP facade for durable Funes memory storage. It binds to `0.0.0.0:7860` and stores SQLite/FTS5 data under `FUNES_DATA_DIR` (default `/data`).

## API

- `GET /health`, `GET /ready` (no auth)
- Bearer-authenticated `POST /ingest`, `/search`, `/recall`, `/get`, `/reindex`, `/sync`
- Bearer-authenticated `GET /sources`, `/sync/status`

Set `FUNES_AUTH_TOKEN` in Space secrets. Ingest documents with `raw_text` and optional identity/version/metadata fields. Responses include `raw_text`; logs never include authorization or document bodies.

Set `FUNES_STORAGE_REPO` (private dataset repo) and `HF_TOKEN` to enable JSONL snapshot recovery/upload. Snapshots preserve source identity and metadata; the SQLite FTS index is rebuilt locally.

Chinese queries use the optional OpenAI-compatible translation endpoint configured by `TRANSLATION_BASE_URL`, `TRANSLATION_API_KEY`, and `TRANSLATION_MODEL`. Failed translation falls back to the original query.
