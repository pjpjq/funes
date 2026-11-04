# Configuration and local files

funes has no global configuration file. A read's memory is the one it names, else the one this
host is bound to (`funes bind`), else the local memory; the hooks publish to the bound memory. The
files below hold the local derived memory, incremental state, the binding, and integration wiring.

## The funes home

`FUNES_HOME` changes funes's state directory; the default is `~/.funes`.

```bash
FUNES_HOME=/tmp/funes-demo funes index ./traces
FUNES_HOME=/tmp/funes-demo funes recall "what changed"
```

Use the same value on every command that should see that isolated memory. This is useful for demos,
benchmarks, and tests because it leaves the normal local memory untouched.

| Path below `$FUNES_HOME` | Purpose |
| --- | --- |
| `memory/chunks.lance/` | The local Lance memory: passages, provenance, embeddings, and search indexes. |
| `state.json` | Per-source incremental indexing state. |
| `index-coverage.json` | The last sweep's small coverage snapshot used by `funes status`. |
| `pushed/` | Per-remote receipts used to report this host's pending push coverage. |
| `bound` | The memory this host is bound to, as typed to `funes bind`; absent when the host is local. |

The memory and indexing state are derived from the original agent transcripts and can be rebuilt,
and push receipts can be recreated by running `funes push <memory>`.

`FUNES_HOME` does **not** relocate agent configuration, installed integrations, model caches, or
the install id (`~/.funes/install-id`). Those paths must remain stable after an agent records them.

## Agent integration files

`funes add` writes or registers these user-wide files; `funes remove <agent>` removes the matching
registration and funes-owned files/entries:

| Agent | Files or configuration |
| --- | --- |
| Claude Code | Hooks-only plugin under `~/.funes/agents/claude/`; registered through Claude's plugin commands. |
| Codex | Plugin carrying the skill and the hooks under `~/.funes/agents/codex/`; registered through Codex's plugin commands. |
| Hermes | Plugin carrying the automation hooks at `~/.hermes/plugins/funes/`; enabled through hermes' plugin commands. |
| pi | Extension under `~/.funes/agents/pi/`, with its copy of the bound memory. |

Beside each integration's directory, `~/.funes/agents/<id>.json` records what `funes add`
installed there and where the files came from. A funes from before the host binding noted a memory
beside each agent, as `<id>.memory`; the next `funes update` or bare `funes add` moves that into
the binding.

What each integration puts there, and which of its agent's events it hooks, is in the
integration's README — the maintained ones are in
[funes-integrations](https://github.com/huggingface/funes-integrations#readme). Hook logs sit
beside the installed scripts as `funes-sync.log`.

## Authentication

Private-memory reads and all Hub writes need a Hugging Face token. funes uses the first non-empty
token in this order:

1. `HF_TOKEN`
2. `HUGGING_FACE_HUB_TOKEN`
3. `HUGGINGFACE_TOKEN`
4. the token file, written by `hf auth login`: `$HF_TOKEN_PATH` if set, else `$HF_HOME/token`
   (`HF_HOME` defaults to `~/.cache/huggingface`)

`HF_HUB_DISABLE_IMPLICIT_TOKEN`, set to anything, makes funes use none of them, as it does hf-hub.

A token used only for recall needs read access; `push` needs write
access to the target dataset repository. Public-memory recall needs no token.

## Model and remote caches

When local embedding or local reranking is selected, the backend downloads its pinned models into the standard
Hugging Face cache (`$HF_HUB_CACHE` when set, else `$HF_HOME/hub`, else
`~/.cache/huggingface/hub`). The optional ONNX build
downloads its reranker there too, and its embedders into fastembed's `.fastembed_cache` under the
process working directory unless configured by that library.
Voyage inference uses its native HTTP API instead of a local model cache.

Remote `hf://` recall also uses the standard hf-hub file cache. `HF_HUB_CACHE` can relocate that
cache; `HF_HOME` relocates the broader Hugging Face home. See [hub-caching.md](hub-caching.md) for the
file-grained cache design and cold-versus-warm behavior.

## Retrieval providers

Embedding and reranking are independent runtime choices:

| Operation | Provider | Contract |
| --- | --- | --- |
| Embedding | `local` | Existing pinned `BAAI/bge-small-en-v1.5`, 384 dimensions, schema version `1`; no API credential. This is the standalone CLI default. |
| Embedding | `voyage` | Defaults to `voyage-4-lite`, 1024 dimensions, schema version `2`; original multilingual documents use `input_type=document` and queries use `input_type=query`. This is the production image default. |
| Reranking | `none` | Keep RRF order; this is the default, including production. |
| Reranking | `local` | Use the existing pinned local cross-encoder. |
| Reranking | `voyage` | Use Voyage rerank, default model `rerank-3-lite`. |

Every new memory stores the complete embedding profile — provider, model, dimensions, schema
version, and a fingerprint covering those fields plus the document/query modes, normalization, and
distance metric. Recall validates all of it before provider loading or query embedding, and refuses
to mix incompatible spaces even if their vector widths match. Legacy memories without the complete
profile remain recognized only as the known local BGE/E5 384-dimensional spaces. Changing a profile therefore
requires rebuilding the derived memory from its raw transcripts.

Voyage receives original text rather than translated or English shadow text when
`FUNES_RETRIEVAL_LANGUAGE_MODE=raw`. Keep `VOYAGE_API_KEY` only in a secret manager or process
environment: never commit it, write it into these files, or include it in logs.

## Environment reference

| Variable | Effect |
| --- | --- |
| `FUNES_HOME` | Local memory and funes state directory; default `~/.funes`. |
| `FUNES_BIN` | Binary path recorded in supported MCP registrations and used by the pi bridge. Hook workers instead find `funes` on `PATH` or in common install directories. |
| `FUNES_MEMORY` | Per-run memory override understood by the pi extension; otherwise its copy of the host's binding is used. |
| `FUNES_INTEGRATIONS` | Directory `funes add` installs an agent integration from, instead of the published one. funes can't vouch for those files, so it asks you to confirm before running them — once, until they change. |
| `FUNES_INDEX_MEMORY` | Optional blue/green build target for the Space canonical reconciler. It defaults to `FUNES_MEMORY`; set it to a new private dataset when changing embedding spaces. |
| `FUNES_NATIVE_PRIMARY` | Legacy opt-in. When `true`, macOS also runs a local native index/push helper. Keep the default `false` when the remote Space has a canonical reconciler. |
| `FUNES_NATIVE_FALLBACK` | Remote-read fallback policy. `false` disables substitution of the offline local memory; production sets `false`. The standalone default is enabled. |
| `FUNES_EMBEDDING_PROVIDER` | Runtime embedding provider: `local` or `voyage`. Standalone default: `local`; production image default: `voyage`. |
| `FUNES_EMBEDDING_MODEL` | Voyage embedding model; default `voyage-4-lite`. Local embedding uses the model stored in the memory (BGE or multilingual E5). |
| `FUNES_EMBEDDING_DIMENSIONS` | Voyage output width: `256`, `512`, `1024`, or `2048`; default `1024`. Local embedding remains fixed at `384`. |
| `FUNES_EMBEDDING_SCHEMA_VERSION` | Embedding contract version. Voyage requires `2`; local remains fixed at `1`. |
| `FUNES_INDEX_EMBEDDING_PROVIDER` | Optional build-target provider. Defaults to the active `FUNES_EMBEDDING_PROVIDER`. |
| `FUNES_INDEX_EMBEDDING_MODEL` | Optional build-target model. Defaults to the active model, or the provider default when the build provider changes. |
| `FUNES_INDEX_EMBEDDING_DIMENSIONS` | Optional build-target vector width. Defaults to the active width, or the provider default when the build provider changes. |
| `FUNES_INDEX_EMBEDDING_SCHEMA_VERSION` | Optional build-target schema version. Defaults to the active version, or the provider default when the build provider changes. |
| `FUNES_RERANK_PROVIDER` | Runtime reranker: `none`, `local`, or `voyage`; default `none`, including production. |
| `FUNES_RERANK_MODEL` | Voyage rerank model; default `rerank-3-lite`. Ignored by `none` and `local`. |
| `VOYAGE_API_KEY` | Required when either selected provider is `voyage`; secret-only process environment or secret-manager value. Never commit or log it. |
| `FUNES_RETRIEVAL_LANGUAGE_MODE` | Retrieval text mode. Production sets `raw`, preserving original multilingual documents and queries without translation or shadow text. |
| `FUNES_SYNC_INTERVAL` | Continuous reconciliation interval in seconds; default `300`. |
| `FUNES_SYNC_BATCH` | Maximum records per HTTP ingest segment; default `50`. |
| `FUNES_SYNC_MAX_BATCH_BYTES` | Maximum serialized source bytes per ingest segment; default `16777216` (16 MiB). A single already-chunked record is never dropped. |
| `FUNES_REMOTE_TIMEOUT` | Total deadline for starting and polling one durable ingest operation; default `900` seconds. |
| `FUNES_REMOTE_ATTEMPT_TIMEOUT` | Bounded per-request network attempt timeout for ingest POST and polling requests; default `30` seconds (safe range `1..55`s). |
| `FUNES_REMOTE_MAX_RESPONSE_BYTES` | Maximum ingest/status response body; default `1048576` (1 MiB). |
| `FUNES_HTTP_GZIP` | Compress HTTP ingest bodies when beneficial; enabled by default. |
| `FUNES_INGEST_OPERATION_TIMEOUT` | Space-side durable worker deadline; default `1800` seconds. A stall fail-stops the process for platform restart without ACKing local data. |
| `FUNES_CANONICAL_INDEX_TIMEOUT` | Subprocess deadline for one canonical index batch; default `1800` seconds. |
| `FUNES_API_TOKEN` | Bearer token for the compatibility HTTP bridge; never written to launchd plist, loaded from Keychain on macOS. |
| `FUNES_TRUFFLEHOG` | Explicit TruffleHog binary for secret scanning. Index-time redaction is best-effort; push and scrub scanning fail closed. |
| `HF_TOKEN`, `HUGGING_FACE_HUB_TOKEN`, `HUGGINGFACE_TOKEN` | Hugging Face authentication, in the precedence shown above. |
| `HF_TOKEN_PATH` | The token file, when it is not `$HF_HOME/token`. |
| `HF_HUB_DISABLE_IMPLICIT_TOKEN` | Set to anything: no token is used, from the variables or a file. |
| `HF_HOME` | Hugging Face home: the token file, and the default backend's model cache. |
| `HF_HUB_CACHE` | Hugging Face Hub file-cache location, including cached remote-memory files. |
| `HF_HUB_USER_AGENT_ORIGIN` | Origin hf-hub appends to the User-Agent of every Hub request. funes adds `funes/<version>` and `install/<id>` to it, keeping an origin already set, and the agent integrations add `agent/<harness>`. |
| `DO_NOT_TRACK`, `FUNES_NO_TELEMETRY` | Either set keeps the install id out of the User-Agent. |
| `NO_COLOR` | Disable ANSI color in human-facing terminal output. |
| `COLUMNS` | Human-rendering width, clamped to 40–120 columns. |

The host's binding is `$FUNES_HOME/bound`, written by `funes bind` alone, which `funes add <agent>
<memory>` runs first. Each agent's registration carries a copy, kept in step by `bind` re-running
the setups. `funes bind local` unbinds; `funes remove <agent>` removes that agent integration
without touching the binding or the memory.

When changing embedding spaces in the Docker Space, never point the new provider at the old Lance
dataset. Keep `FUNES_MEMORY` and its active profile unchanged, set `FUNES_INDEX_MEMORY` plus the four
`FUNES_INDEX_EMBEDDING_*` values to a new private dataset, and let the canonical reconciler rebuild it
from the encrypted durable raw source. The authenticated readiness payload reports separate
`active_index`, `build_index`, and `canonical_index.cutover_ready` state. Only after the build is
complete and optimized should the Space configuration switch `FUNES_MEMORY` and the active
`FUNES_EMBEDDING_*` values together. The old dataset remains an immediate rollback target.
