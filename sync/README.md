# Unified sync daemon

`python -m sync backfill` discovers Codex, Pi, Claude Code and persistent instruction files,
parses them into stable chunks, stores state in `~/.local/share/funes-sync/sync.db`, and queues
batches for a durable remote ACK. For production HF deployment set `FUNES_NATIVE_PRIMARY=true`:
the daemon then delegates indexing/publishing to the native `funes index`/`funes push` pipeline
(Lance vector + BM25 + rerank + TruffleHog gate). `FUNES_API_TOKEN` is read only from the
environment/Keychain. Use `python -m sync run` for polling, `install/uninstall` for a macOS
LaunchAgent, and `python -m sync.mcp_bridge` as a stdio recall/get bridge.
