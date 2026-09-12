---
title: Funes Memory Bridge
sdk: docker
app_port: 7860
---

# Funes HTTP bridge

The Space is a thin authenticated bridge over the upstream Funes CLI. Durable data
lives in the HF Hub dataset named by `FUNES_MEMORY`; the container cache is rebuildable.

* `GET /health` is public.
* `GET /ready` and `POST /search`, `/recall`, `/get`, `/ingest` require `Authorization: Bearer FUNES_API_TOKEN` when the secret is configured.

Configure Space Secrets (never commit values): `HF_TOKEN`, `FUNES_MEMORY`,
`FUNES_API_TOKEN`, and optional `TRANSLATION_BASE_URL`, `TRANSLATION_API_KEY`,
`TRANSLATION_MODEL`.
