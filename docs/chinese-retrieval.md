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

## Provider-backed native runner

`scripts/benchmark_chinese_retrieval_provider.py` closes the provider/E2E gap without changing a
real memory. It reads the provider credential only from an environment variable, normalizes the
same 60 memories and 20 queries through the OpenAI-compatible
`https://router.huggingface.co/v1/chat/completions` endpoint with
`Qwen/Qwen3-4B-Instruct-2507`, and builds two isolated temporary native Funes memories through
`funes ingest-docs <jsonl> --memory <local-path>`:

1. **Before:** canonical `retrieval_text` is the raw Chinese fixture text; queries are raw Chinese.
2. **After:** canonical `retrieval_text` is only the provider-generated English shadow; queries use
   provider-generated shadows.

Every canonical row includes `source_identity`, `source_version`, `retrieval_text`, `content_hash`,
`updated_at`, and `metadata`. Neither arm emits a synthetic transcript or a `raw_text` sidecar, so
the After native index never contains the raw Chinese document text. Each arm has its own temporary
`FUNES_HOME` and explicit local memory path; recall serves that exact path. Both arms request `k=5`,
`candidates=12`, `neighbors=0`, and `half_life=0` for the same 20 queries from one warm MCP process
per memory. The output records Recall@1/@3/@5, per-query ranks, provider outputs, model, endpoint,
prompt hash, backend, and latency in `docs/chinese-retrieval-provider-results.json`; it never records
the credential. Run the offline invariant check first, then the real comparison:

```bash
python3 scripts/benchmark_chinese_retrieval_provider.py --self-check
python3 scripts/benchmark_chinese_retrieval_provider.py --token-env HF_TOKEN
```

Provider outputs are atomically checkpointed to the result path **before** canonical ingestion. If
the ingestion/model network path fails, rerunning the same command reuses that checkpoint and makes
no more provider calls. `--provider-only` explicitly stops at that checkpoint.

### Execution evidence (2026-09-13)

The HF Router probe returned HTTP 200, and the full provider stage completed all 44 unique inputs
(60 memories contain repeated distractors). Native indexing then exited 1 before either arm could be
measured because the active `target/release/funes` was an ONNX-only build and downloading
`onnx/model.onnx` failed with `connection reset by peer (os error 54)`. That attempt predated the
checkpoint fix, so its generated text could not be recovered. A cached default-BLAS binary was
independently verified with a one-memory native index (`exit 0`). **No provider-backed Recall@k is
currently available or claimed from the failed run.** The runner now preserves provider output
across exactly this failure; a successful canonical-ingestion run is still required before reporting
provider Recall@k.
