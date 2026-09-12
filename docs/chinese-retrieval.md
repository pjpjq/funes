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
fixture but lowers Recall@1. The provider-backed comparison below is the authoritative follow-up.

## Provider-backed native runner

`scripts/benchmark_chinese_retrieval_provider.py` closes the provider/E2E gap without changing a
real memory. It reads the provider credential only from an environment variable, normalizes the
same 60 memories and 20 queries through the OpenAI-compatible
`https://router.huggingface.co/v1/chat/completions` endpoint with
`Qwen/Qwen3-4B-Instruct-2507`, and builds two isolated temporary native Funes memories through
`funes ingest-docs <jsonl> --memory <local-path>`:

1. **Before:** canonical `retrieval_text` is the raw Chinese fixture text; queries are raw Chinese.
2. **After:** canonical `retrieval_text` is only the document-prompt English shadow; queries use the
   query-specific prompt and fall back to the raw query when entity, number, or length validation
   rejects the provider output.

Every canonical row includes `source_identity`, `source_version`, `retrieval_text`, `content_hash`,
`updated_at`, and `metadata`. Neither arm emits a synthetic transcript or a `raw_text` sidecar, so
the After native index never contains the raw Chinese document text. Each arm has its own temporary
`FUNES_HOME` and explicit local memory path; recall serves that exact path. Both arms request `k=5`,
`candidates=12`, `neighbors=0`, and `half_life=0` for the same 20 queries from one warm MCP process
per memory. The output records Recall@1/@3/@5, per-query ranks, raw/effective provider outputs,
model, endpoint, both prompt versions and hashes, backend, and latency in
`docs/chinese-retrieval-provider-results.json`; it never records the credential. Run the offline
invariant check first, then the real comparison:

```bash
python3 scripts/benchmark_chinese_retrieval_provider.py --self-check
python3 scripts/benchmark_chinese_retrieval_provider.py --token-env HF_TOKEN
```

Provider outputs are atomically checkpointed to the result path **before** canonical ingestion. If
the ingestion/model network path fails, rerunning the same command reuses that checkpoint and makes
no more provider calls. `--provider-only` explicitly stops at that checkpoint.

### Execution evidence (2026-09-13)

The full v2 run completed 24 distinct document normalizations, 20 query rewrites, both canonical
indexes, and all 40 native recall calls. One query rewrite changed the technical entity `Mac` to
`mac`; validation rejected it and used the original Chinese query. The recorded credential scan is
clean (`credential_value_recorded=false`, and no configured token value occurs in the JSON).

| mode | Recall@1 | Recall@3 | Recall@5 | index seconds | mean query seconds |
| --- | ---: | ---: | ---: | ---: | ---: |
| raw Chinese | 0.75 | 0.90 | 0.90 | 1.817 | 0.272 |
| English retrieval shadow | 0.70 | 0.90 | 0.90 | 1.469 | 0.256 |
| delta (shadow - raw) | -0.05 | 0.00 | 0.00 | -0.348 | -0.016 |

The English-only arm ties raw Chinese at Recall@3 and Recall@5 but remains 0.05 lower at Recall@1;
therefore this result does not support a blanket claim that translation improves retrieval. The
deployed HTTP service mitigates that limitation by fusing raw-query, rewritten-query, and native
rankings with RRF, while this controlled benchmark intentionally isolates the two representations.
