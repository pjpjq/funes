# Configuration and local files

funes has no global configuration file. A command's memory is selected explicitly, baked into an
agent registration by `funes add`, or left at the default local memory. The files below hold the
local derived memory, incremental state, and integration wiring.

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

The memory and indexing state are derived from the original agent transcripts and can be rebuilt,
and push receipts can be recreated by running `funes push <memory>`.

`FUNES_HOME` does **not** relocate agent configuration, installed integrations, or model caches.
Those paths must remain stable after an agent records them.

## Agent integration files

`funes add` writes or registers these user-wide files; `funes remove <agent>` removes the matching
registration and funes-owned files/entries:

| Agent | Files or configuration |
| --- | --- |
| Claude Code | Hooks-only plugin under `~/.funes/integrations/claude-plugin`; registered through Claude's plugin commands. |
| Codex | `~/.codex/hooks.json` and scripts under `~/.codex/hooks/`. |
| Hermes | `~/.hermes/config.yaml`, `~/.hermes/shell-hooks-allowlist.json`, and scripts under `~/.hermes/hooks/`. |
| pi | Extension and optional memory binding under `~/.funes/integrations/pi/`. |

See [automation.md](automation.md) for how these files are merged and which events they handle.
Hook logs sit beside the installed scripts as `funes-sync.log`.

## Authentication

Private-memory reads and all Hub writes need a Hugging Face token. funes uses the first non-empty
token in this order:

1. `HF_TOKEN`
2. `HUGGING_FACE_HUB_TOKEN`
3. `HUGGINGFACE_TOKEN`
4. `~/.cache/huggingface/token`, written by `hf auth login`

A token used only for recall needs read access; `push` needs write
access to the target dataset repository. Public-memory recall needs no token.

## Model and remote caches

The default inference backend downloads its pinned embedder and reranker into the standard
Hugging Face cache (`$HF_HOME/hub`, or `~/.cache/huggingface/hub`). The optional ONNX build uses
fastembed's `.fastembed_cache` under the process working directory unless configured by that
library.

Remote `hf://` recall also uses the standard hf-hub file cache. `HF_HUB_CACHE` can relocate that
cache; `HF_HOME` relocates the broader Hugging Face home. See [hub-caching.md](hub-caching.md) for the
file-grained cache design and cold-versus-warm behavior.

## Environment reference

| Variable | Effect |
| --- | --- |
| `FUNES_HOME` | Local memory and funes state directory; default `~/.funes`. |
| `FUNES_BIN` | Binary path recorded in supported MCP registrations and used by the pi bridge. Hook workers instead find `funes` on `PATH` or in common install directories. |
| `FUNES_MEMORY` | Per-run memory override understood by the pi extension; otherwise its binding from `funes add pi [memory]` is used. |
| `FUNES_NATIVE_PRIMARY` | When `true`, `funes-sync` delegates production index/push to native Funes instead of the migration HTTP path. |
| `FUNES_SYNC_INTERVAL` | Continuous reconciliation interval in seconds; default `300`. |
| `FUNES_SYNC_BATCH` | Maximum records per HTTP ingest segment; default `50`. |
| `FUNES_SYNC_MAX_BATCH_BYTES` | Maximum serialized source bytes per ingest segment; default `16777216` (16 MiB). A single already-chunked record is never dropped. |
| `FUNES_REMOTE_TIMEOUT` | Total deadline for starting and polling one durable ingest operation; default `900` seconds. |
| `FUNES_REMOTE_MAX_RESPONSE_BYTES` | Maximum ingest/status response body; default `1048576` (1 MiB). |
| `FUNES_HTTP_GZIP` | Compress HTTP ingest bodies when beneficial; enabled by default. |
| `FUNES_INGEST_OPERATION_TIMEOUT` | Space-side durable worker deadline; default `1800` seconds. A stall fail-stops the process for platform restart without ACKing local data. |
| `FUNES_API_TOKEN` | Bearer token for the compatibility HTTP bridge; never written to launchd plist, loaded from Keychain on macOS. |
| `FUNES_TRUFFLEHOG` | Explicit TruffleHog binary for secret scanning. Index-time redaction is best-effort; push and scrub scanning fail closed. |
| `HF_TOKEN`, `HUGGING_FACE_HUB_TOKEN`, `HUGGINGFACE_TOKEN` | Hugging Face authentication, in the precedence shown above. |
| `HF_HOME` | Hugging Face home, including the default backend's model cache. |
| `HF_HUB_CACHE` | Hugging Face Hub file-cache location, including cached remote-memory files. |
| `NO_COLOR` | Disable ANSI color in human-facing terminal output. |
| `COLUMNS` | Human-rendering width, clamped to 40–120 columns. |

Bindings passed to `funes add` live in the agent's own registration or integration files; there is
no hidden “active remote” in `$FUNES_HOME`. Re-run `funes add <agent> [memory]` to change one, or
`funes remove <agent>` to remove that agent integration without deleting the memory.
