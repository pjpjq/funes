# Chinese retrieval shadow

The unified service keeps `raw_text` as the source of truth and stores an optional
English `retrieval_text` shadow.  Translation is enabled only when the CJK ratio is
at least `TRANSLATE_CHINESE_THRESHOLD` (default `0.15`); failures fall back to the
raw text and never block ingestion.  Set `TRANSLATION_BASE_URL`,
`TRANSLATION_API_KEY`, and `TRANSLATION_MODEL` for an OpenAI-compatible provider.

## Reproducible benchmark

Use the benchmark fixture and run it against a clean service twice:

```bash
PYTHONPATH=. python -m pytest -q tests_sync service/tests
```

For a provider-backed comparison, ingest the 50+ Chinese fixture memories with
`FUNES_RETRIEVAL_LANGUAGE_MODE=raw`, record Recall@1/3/5 for the 20 queries, then
repeat with `language_mode=auto`.  The HTTP API always returns `raw_text`; set
`RETURN_RETRIEVAL_TEXT=true` only when inspecting derived text.

The checked-in service tests cover the required CPA/Northflank,
Tailscale/Pi, Gemini reasoning preference, and mixed
`previous_response_id`/`chatcmpl-*` entities.  Provider-dependent Recall@k values
must be recorded only after a real translation provider and embedding backend are
configured; no fabricated scores are included here.
