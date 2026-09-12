# Chinese retrieval shadow

The unified service keeps `raw_text` as the source of truth and may store an optional
English `retrieval_text` shadow. Translation is enabled only when the CJK ratio is at least
`TRANSLATE_CHINESE_THRESHOLD` (default `0.15`); failures fall back to raw text and never block
ingestion. Set `TRANSLATION_BASE_URL`, `TRANSLATION_API_KEY`, and `TRANSLATION_MODEL` for an
OpenAI-compatible provider.

## Reproducible synthetic proxy

Run the checked-in 60-memory/20-query fixture from any checkout:

```bash
python3 scripts/benchmark_chinese_retrieval.py
```

The run writes `docs/chinese-retrieval-results.json` and reports two **synthetic ASCII-token
overlap proxies**:

| mode | Recall@1 | Recall@3 | Recall@5 |
| --- | ---: | ---: | ---: |
| synthetic raw Chinese ASCII-token proxy | 0.45 | 0.65 | 0.75 |
| synthetic English-shadow ASCII-token proxy | 0.75 | 0.75 | 1.00 |

These numbers are deterministic fixture measurements only. The script does **not** invoke Funes,
Lance, the deployed service, an FTS index, an embedding model, or BGE; they must not be described as
a Funes/BGE benchmark or as provider-backed retrieval quality. The result JSON records this boundary
with `benchmark_kind`, `not_a_real_funes_or_bge_benchmark`, and separate `synthetic_*_proxy` keys.

## Real E2E benchmark scaffold

A real run must index the same fixture through a real Funes binary/local memory and a configured
embedding/retrieval backend. The script intentionally does not guess those paths or credentials.
Pass an explicit command; its status and capped stdout/stderr are recorded under `e2e` and are kept
separate from the synthetic metrics:

```bash
# First index the fixture with your real Funes setup, then pass one argv-style command:
python3 scripts/benchmark_chinese_retrieval.py \
  --e2e-command 'funes recall CPA 第二轮上下文'
```

`--e2e-command` is parsed without a shell, so operators such as `&&` are not interpreted;
use a small checked-in wrapper command if setup requires multiple steps. `e2e.status` is `not_run` unless `--e2e-command` is supplied. A completed command is evidence that
the command ran, not evidence that its Recall@k is comparable; a real E2E harness should export its
own ranked results and report the backend/model, corpus, query set, and latency separately.

The HTTP API always returns `raw_text`; set `RETURN_RETRIEVAL_TEXT=true` only when inspecting derived
text. Provider-dependent Recall@k values must be recorded only after a real translation provider
and embedding backend are configured; the synthetic proxy values above are not substitutes.

## Real native Funes E2E (recorded 2026-09-12)

The same 60-memory/20-query fixture was indexed into an isolated temporary `FUNES_HOME` and
queried through one long-lived `funes mcp local` process. This exercises the native Lance vector +
BM25 + reranker path (default `BAAI/bge-small-en-v1.5`), not the SQLite compatibility service.
The shadow arm uses the deployed bridge's deterministic `auto` fallback because no external
translation provider secret was configured; it is therefore a real backend shadow test, not a
provider-quality claim. Full rows are in `docs/chinese-retrieval-e2e-results.json`.

| mode | Recall@1 | Recall@3 | Recall@5 | mean query seconds |
| --- | ---: | ---: | ---: | ---: |
| native Funes, raw Chinese query | 0.85 | 0.90 | 0.90 | 0.278 |
| native Funes, deterministic English shadow query | 0.65 | 0.90 | 0.95 | 0.226 |

The result is intentionally not presented as “shadow always wins”: it improves Recall@5 on this
fixture but lowers Recall@1. A provider-backed Chinese translation comparison remains a follow-up
when `TRANSLATION_BASE_URL`, `TRANSLATION_API_KEY`, and `TRANSLATION_MODEL` are configured.
