# Unified sync daemon

`python -m sync backfill` discovers Codex, Pi, Claude Code and persistent instruction files,
parses them into stable chunks, stores state in `~/.local/share/funes-sync/sync.db`, and queues
batches for `POST $FUNES_REMOTE_URL/ingest`. `FUNES_API_TOKEN` is read only from the environment.
Use `python -m sync run` for polling, `install/uninstall` for a macOS LaunchAgent, and
`python -m sync.mcp_bridge` as a stdio recall/get bridge.
