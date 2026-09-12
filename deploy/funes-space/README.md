---
title: Funes Unified Agent Memory
sdk: docker
app_port: 7860
---

# Funes unified agent memory

This Docker Space exposes the low-memory HTTP facade in `service/`:

- public `GET /health` and `GET /ready`;
- bearer-authenticated `POST /ingest`, `/search`, `/recall`, `/get`, `/reindex`, `/sync`;
- bearer-authenticated `GET /sources` and `GET /sync/status`.

The raw source text remains the durable source of truth. `retrieval_text`, BM25/FTS data and
translation shadows are derived and rebuildable. Set `FUNES_STORAGE_REPO` to a private dataset
and `HF_TOKEN` to restore/upload the JSONL snapshot across Space restarts.

Required Space Secrets/Variables (values never belong in this repository):

- `FUNES_API_TOKEN`
- `FUNES_STORAGE_REPO`
- `HF_TOKEN`

Optional translation secrets: `TRANSLATION_BASE_URL`, `TRANSLATION_API_KEY`, `TRANSLATION_MODEL`.
