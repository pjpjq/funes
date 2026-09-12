---
title: Funes Memory Bridge
sdk: docker
app_port: 7860
---

# Funes HTTP bridge

The Space is a thin authenticated bridge over the upstream Funes CLI. Durable data
lives in the HF Hub Lance dataset named by `FUNES_MEMORY`; the container cache is
rebuildable. `/ingest` returns success only after native `funes push` commits the
batch durably; a failed push returns 503 so the local sync queue retries.

* `GET /health` is public at the application layer. For a private Space, send
  the HF `Authorization` bearer to pass the Space front door.
* `GET /ready` and `POST /search`, `/recall`, `/get`, `/ingest` require the
  application bearer in `X-Funes-Authorization` (the HF bearer remains the
  separate front-door credential).

Configure Space Secrets (never commit values): `HF_TOKEN`, `FUNES_MEMORY`,
`FUNES_API_TOKEN`, and optional `FUNES_RETRIEVAL_LANGUAGE_MODE` (`raw`, `auto`,
`translate`), `TRANSLATION_BASE_URL`, `TRANSLATION_API_KEY`, `TRANSLATION_MODEL`.
