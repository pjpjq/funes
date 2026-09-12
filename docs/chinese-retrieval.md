# Chinese retrieval shadow

The unified service keeps `raw_text` as the source of truth and stores an optional
English `retrieval_text` shadow.  Translation is enabled only when the CJK ratio is
at least `TRANSLATE_CHINESE_THRESHOLD` (default `0.15`); failures fall back to the
raw text and never block ingestion.  Set `TRANSLATION_BASE_URL`,
`TRANSLATION_API_KEY`, and `TRANSLATION_MODEL` for an OpenAI-compatible provider.

## Reproducible benchmark

Run the checked-in 60-memory/20-query fixture:

```bash
/Users/pwd/anaconda3/bin/python scripts/benchmark_chinese_retrieval.py
```

The run writes `docs/chinese-retrieval-results.json` and produced:

| mode | Recall@1 | Recall@3 | Recall@5 |
| --- | ---: | ---: | ---: |
| raw Chinese → English-only BGE/FTS proxy | 0.45 | 0.65 | 0.75 |
| English retrieval shadow → same proxy | 0.75 | 0.75 | 1.00 |

These are **offline ASCII-token overlap proxy** measurements for the deployed
FTS path, not claimed BGE embedding scores. They are reproducible without a
translation secret and prevent invented provider results. Once an
OpenAI-compatible provider is configured, run the same fixture against the HTTP
service with `FUNES_RETRIEVAL_LANGUAGE_MODE=raw` and `auto` to obtain
provider-backed numbers. The HTTP API always returns `raw_text`; set
`RETURN_RETRIEVAL_TEXT=true` only when inspecting derived text.

The checked-in service tests cover the required CPA/Northflank,
Tailscale/Pi, Gemini reasoning preference, and mixed
`previous_response_id`/`chatcmpl-*` entities. Provider-dependent Recall@k values
must be recorded only after a real translation provider and embedding backend are
configured; the proxy values above are clearly labelled and are not substituted
for that run.
