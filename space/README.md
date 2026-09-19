---
title: Funes Memory Bridge
sdk: docker
app_port: 7860
---

# Funes HTTP bridge

The Space combines the upstream Funes retrieval engine with an authoritative source
sidecar. Derived Lance data lives in `FUNES_MEMORY`; original `raw_text`, stable source
identity, metadata, and translation cache live as AES-256-GCM encrypted snapshots in
the private dataset named by `FUNES_STORAGE_REPO`. Container caches are rebuildable.
`/ingest` returns success only after the encrypted source delta is durable. The sync
client sends `Prefer: respond-async`; the Space returns `202` plus an authenticated
operation URL while the Hub commit is running, and the client keeps its local queue
until polling returns `200` with `durable=true`. Requests without that preference keep
the synchronous compatibility behavior. A failed upload returns 503 so the local queue
retries.

* `GET /health` is public at the application layer. For a private Space, send
  the HF `Authorization` bearer to pass the Space front door.
* `GET /ready`, `GET /ingest/operations/<id>`, and `POST /search`, `/recall`, `/get`, `/ingest` require the
  application bearer in `X-Funes-Authorization` (the HF bearer remains the
  separate front-door credential).

Configure Space Secrets (never commit values): `HF_TOKEN`, `FUNES_MEMORY`,
`FUNES_API_TOKEN`, `FUNES_STORAGE_REPO`, and `FUNES_STORAGE_KEY`. The storage key
may initially equal the existing API token but should be kept unchanged if the API
token is later rotated. Optional retrieval settings are
`FUNES_RETRIEVAL_LANGUAGE_MODE` (`raw`, `auto`, `translate`),
`TRANSLATION_BASE_URL`, `TRANSLATION_API_KEY`, and `TRANSLATION_MODEL`.
`FUNES_INGEST_OPERATION_TIMEOUT` defaults to 1800 seconds; a worker that exceeds
it triggers a fail-stop process restart so an unkillable Python thread cannot
hold the only durable-write slot forever.
