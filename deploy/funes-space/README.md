---
title: Funes Unified Agent Memory
sdk: docker
app_port: 7860
---

# Funes unified agent memory

This Docker Space combines the native `funes` retrieval engine with the
source-of-truth sidecar in `service/server.py`. Native Funes remains the derived
Lance vector/BM25/reranking index. Original raw text, source identity, metadata,
and translation cache are compressed and encrypted with streaming AES-256-GCM
before they enter the private Hub source dataset.

- `GET /health` and `GET /ready` (a private Space also needs the HF bearer at the
  front door; `/ready` additionally checks the application bearer);
- bearer-authenticated `POST /ingest`, `/search`, `/recall`, and `/get`.

Source discovery and backfill remain local-daemon operations (`funes sync status`,
`funes sources`). The daemon uploads durable raw/source records; the Space canonical reconciler
incrementally rebuilds and publishes the derived native index. The Space never accesses the Mac
filesystem directly.

Set `FUNES_MEMORY` to the rebuildable private Funes dataset and
`FUNES_STORAGE_REPO` to the separate private encrypted-source dataset. The
container filesystem is only a warm cache.

Required Space Secrets/Variables (values never belong in this repository):

- `FUNES_API_TOKEN`
- `FUNES_MEMORY`
- `FUNES_STORAGE_KEY`
- `FUNES_STORAGE_REPO`
- `HF_TOKEN`

Optional translation secrets: `TRANSLATION_BASE_URL`, `TRANSLATION_API_KEY`, `TRANSLATION_MODEL`.
