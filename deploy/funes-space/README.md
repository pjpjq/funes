---
title: Funes Unified Agent Memory
sdk: docker
app_port: 7860
---

# Funes unified agent memory

This Docker Space exposes a thin HTTP/MCP-compatible facade over the native
`funes` binary in `space/server.py`.  Native Funes remains the source of search
truth: Lance vector search, BM25, reranking, recency, neighbor expansion, and
the existing TruffleHog fail-closed push gate.

- `GET /health` and `GET /ready` (a private Space also needs the HF bearer at the
  front door; `/ready` additionally checks the application bearer);
- bearer-authenticated `POST /ingest`, `/search`, `/recall`, and `/get`.

Source discovery, backfill, index rebuild, and sync status are local-daemon/CLI
operations (`funes sync status`, `funes sources`, `funes index --yes`, then
`funes push <memory> --force-reindex`); the Space
does not pretend to access the Mac filesystem.

The raw source text remains the durable source of truth. Set `FUNES_MEMORY` to a private
Funes HF dataset (`owner/name` or `hf://datasets/...`) and `HF_TOKEN` to use native
CAS-protected push/recovery across Space restarts. The container filesystem is only a
warm cache.

Required Space Secrets/Variables (values never belong in this repository):

- `FUNES_API_TOKEN`
- `FUNES_MEMORY`
- `HF_TOKEN`

Optional translation secrets: `TRANSLATION_BASE_URL`, `TRANSLATION_API_KEY`, `TRANSLATION_MODEL`.
