#!/bin/sh
# Native historical backfill and periodic reconciliation for a shared Funes
# memory.  The caller supplies FUNES_NATIVE_MEMORY and a Hub token through the
# environment/Keychain; no credential is written here.
set -eu

PATH="/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin:$HOME/.local/bin:$PATH"
export PATH
STATE_DIR="${FUNES_SYNC_STATE_DIR:-$HOME/.local/share/funes-sync}"
LOCK="$STATE_DIR/native-backfill.lockfile"
LOG="${FUNES_NATIVE_BACKFILL_LOG:-$HOME/Library/Logs/funes-native-backfill.log}"
BIN="${FUNES_BIN:-$HOME/.local/bin/funes}"
REMOTE="${FUNES_NATIVE_MEMORY:?FUNES_NATIVE_MEMORY is required}"
WARM_HELPER="${FUNES_NATIVE_WARM_HELPER:-$HOME/.local/share/funes-sync/warm-space.py}"
PYTHON="${PYTHON:-python3}"
PUSH_EVERY="${FUNES_NATIVE_BACKFILL_PUSH_EVERY:-4}"
RECONCILE_INTERVAL="${FUNES_NATIVE_BACKFILL_RECONCILE_INTERVAL:-300}"

case "$PUSH_EVERY" in ''|*[!0-9]*|0) PUSH_EVERY=4 ;; esac
case "$RECONCILE_INTERVAL" in ''|*[!0-9]*|0) RECONCILE_INTERVAL=300 ;; esac
mkdir -p "$STATE_DIR" "$(dirname "$LOG")"
if [ ! -x /usr/bin/lockf ]; then
  printf '%s lockf unavailable\n' "$(date '+%Y-%m-%dT%H:%M:%S%z')" >>"$LOG"
  exit 1
fi
# Hold a kernel advisory lock on fd 9 for the whole script.  The file may
# survive a crash, but the lock cannot: the kernel releases it with the last
# inherited descriptor, so PID reuse and stale-directory ABA races are absent.
exec 9>"$LOCK"
/usr/bin/lockf -s -t 0 9 || exit 0

if [ ! -x "$BIN" ]; then
  printf '%s missing funes binary\n' "$(date '+%Y-%m-%dT%H:%M:%S%z')" >>"$LOG"
  exit 1
fi

# Prefer the macOS Keychain; the zshrc fallback is for the explicit local
# setup requested by the operator and is never echoed or persisted by this
# script.
if [ -z "${HF_TOKEN:-}" ] && command -v security >/dev/null 2>&1; then
  HF_TOKEN="$(security find-generic-password -a "${USER:-$(id -un)}" -s funes-hf-token -w 2>/dev/null || true)"
  [ -n "$HF_TOKEN" ] && export HF_TOKEN
fi
if [ -z "${FUNES_API_TOKEN:-}" ] && command -v security >/dev/null 2>&1; then
  FUNES_API_TOKEN="$(security find-generic-password -a "${USER:-$(id -un)}" -s funes-api-token -w 2>/dev/null || true)"
  [ -n "$FUNES_API_TOKEN" ] && export FUNES_API_TOKEN
fi
if [ -z "${HF_TOKEN:-}" ] && [ -r "$HOME/.zshrc" ] && command -v zsh >/dev/null 2>&1; then
  HF_TOKEN="$(zsh -c 'source "$HOME/.zshrc" >/dev/null 2>&1; printf %s "${FUNES_HF_TOKEN:-${HF_TOKEN:-}}"' 2>/dev/null || true)"
  [ -n "$HF_TOKEN" ] && export HF_TOKEN
fi
if [ -z "${FUNES_API_TOKEN:-}" ] && [ -r "$HOME/.zshrc" ] && command -v zsh >/dev/null 2>&1; then
  FUNES_API_TOKEN="$(zsh -c 'source "$HOME/.zshrc" >/dev/null 2>&1; printf %s "${FUNES_API_TOKEN:-}"' 2>/dev/null || true)"
  [ -n "$FUNES_API_TOKEN" ] && export FUNES_API_TOKEN
fi
[ -n "${HF_TOKEN:-}" ] || { printf '%s HF token unavailable\n' "$(date '+%Y-%m-%dT%H:%M:%S%z')" >>"$LOG"; exit 1; }
export FUNES_TRUFFLEHOG="${FUNES_TRUFFLEHOG:-$HOME/.local/bin/trufflehog}"

CURRENT_TMP=""
cleanup_tmp() {
  case "${CURRENT_TMP:-}" in
    "$STATE_DIR"/native-backfill.*.out)
      "$PYTHON" -c 'import os,sys; os.path.exists(sys.argv[1]) and os.unlink(sys.argv[1])' "$CURRENT_TMP" 2>/dev/null || true
      ;;
  esac
}
trap 'cleanup_tmp' EXIT
trap 'exit 129' HUP
trap 'exit 130' INT
trap 'exit 143' TERM

index_source() {
  index_harness="$1"
  shift
  CURRENT_TMP="$STATE_DIR/native-backfill.$$.out"
  if "$BIN" index "$@" --harness "$index_harness" --yes >"$CURRENT_TMP" 2>&1; then
    cat "$CURRENT_TMP" >>"$LOG"
    grep -Eq 'chunks=[1-9][0-9]*' "$CURRENT_TMP" && changed=1 || true
  else
    printf '%s harness=%s index_failed\n' "$(date '+%Y-%m-%dT%H:%M:%S%z')" "$index_harness" >>"$LOG"
  fi
  cleanup_tmp
  CURRENT_TMP=""
}

ITER_FILE="$STATE_DIR/native-backfill.iteration"
while :; do
  iteration="$(cat "$ITER_FILE" 2>/dev/null || printf '0')"
  case "$iteration" in ''|*[!0-9]*) iteration=0 ;; esac
  iteration=$((iteration + 1)); printf '%s\n' "$iteration" >"$ITER_FILE"
  changed=0
  # Explicit roots use the unit-major path: each session is parsed once for
  # text/tool-use/tool-result instead of three tier-major passes.
  CODEX_ROOT="${CODEX_HOME:-$HOME/.codex}"
  [ -d "$CODEX_ROOT/sessions" ] && index_source codex "$CODEX_ROOT/sessions"
  PI_ROOT="${PI_CODING_AGENT_DIR:-$HOME/.pi/agent}"
  [ -d "$PI_ROOT/sessions" ] && index_source pi "$PI_ROOT/sessions"
  CLAUDE_ROOT="${CLAUDE_CONFIG_DIR:-$HOME/.claude}"
  [ -d "$CLAUDE_ROOT/projects" ] && index_source claude "$CLAUDE_ROOT/projects"
  [ -f "$HOME/.hermes/state.db" ] && index_source hermes "$HOME/.hermes/state.db"
  # Native Funes intentionally keeps one canonical auto-discovery root per
  # harness.  Index the additional locations used by older/current clients
  # with the same official parser so archived and subagent sessions are not
  # silently omitted.
  for source_path in "$CODEX_ROOT/archived_sessions" "$CODEX_ROOT/subagents"; do
    [ -d "$source_path" ] && index_source codex "$source_path"
  done
  for source_path in "$HOME/.pi/sessions" "${PI_CODING_AGENT_SESSION_DIR:-}" "${PI_CODING_AGENT_DIR:-}" "${PI_SESSION_DIR:-}"; do
    [ -d "$source_path" ] && index_source pi "$source_path"
  done
  for source_path in "$CLAUDE_ROOT/history"; do
    [ -d "$source_path" ] && index_source claude "$source_path"
  done
  push_now=0
  [ "$changed" -eq 1 ] && [ $((iteration % PUSH_EVERY)) -eq 0 ] && push_now=1
  if [ "$changed" -eq 1 ] && ! "$BIN" status 2>/dev/null | grep -q 'pending indexing:'; then push_now=1; fi
  if [ "$push_now" -eq 1 ]; then
    if "$BIN" push "$REMOTE" --yes >>"$LOG" 2>&1; then
      # The remote snapshot changed. Ask the Space to refresh its native MCP
      # worker in the background so the next agent query does not pay a cold
      # remote-index load through the ingress timeout.
      if [ -r "$WARM_HELPER" ] && [ -n "${FUNES_REMOTE_URL:-}" ] && [ -n "${FUNES_API_TOKEN:-}" ]; then
        "$PYTHON" "$WARM_HELPER" >/dev/null 2>&1 || true
      fi
    else
      printf '%s push_failed; local index retained\n' "$(date '+%Y-%m-%dT%H:%M:%S%z')" >>"$LOG"
    fi
  fi
  if ! "$BIN" status 2>/dev/null | grep -q 'pending indexing:'; then
    printf '%s native backfill complete; next reconciliation in %ss\n' "$(date '+%Y-%m-%dT%H:%M:%S%z')" "$RECONCILE_INTERVAL" >>"$LOG"
    sleep "$RECONCILE_INTERVAL"
  else
    sleep "${FUNES_NATIVE_BACKFILL_SLEEP:-15}"
  fi
done
