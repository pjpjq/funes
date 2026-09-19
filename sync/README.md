# Unified sync daemon

`python -m sync backfill` discovers Codex, Pi, Claude Code and persistent instruction files,
parses them into stable chunks, stores state in `~/.local/share/funes-sync/sync.db`, and queues
batches for a durable remote ACK. The production HF Space persists those raw/source records and its
canonical reconciler builds the derived native Lance vector + BM25 index. The default macOS install
therefore runs only the lightweight HTTP daemon. Set `FUNES_NATIVE_PRIMARY=true` only for a legacy
remote that still requires a local `funes index`/`funes push` helper. `FUNES_API_TOKEN` is read only
from the environment/Keychain. Use `python -m sync run` for polling, `install/uninstall` for a macOS
LaunchAgent, and `python -m sync.mcp_bridge` as a stdio recall/get bridge.

The daemon also performs a resumable, remote-scoped source inventory through
authenticated `POST /sources/check`. Only missing identities are re-queued; use
`python -m sync reconcile` to force the same check after rebuilding a remote.
