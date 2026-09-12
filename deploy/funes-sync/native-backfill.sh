#!/bin/sh
# Native historical backfill and periodic reconciliation for a shared Funes
# memory.  The caller supplies FUNES_NATIVE_MEMORY and a Hub token through the
# environment/Keychain; no credential is written here.
set -eu

PATH="/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin:$HOME/.local/bin:$PATH"
export PATH
STATE_DIR="${FUNES_SYNC_STATE_DIR:-$HOME/.local/share/funes-sync}"
LOCK="$STATE_DIR/native-backfill.lock"
LOG="${FUNES_NATIVE_BACKFILL_LOG:-$HOME/Library/Logs/funes-native-backfill.log}"
BIN="${FUNES_BIN:-$HOME/.local/bin/funes}"
REMOTE="${FUNES_NATIVE_MEMORY:?FUNES_NATIVE_MEMORY is required}"
PUSH_EVERY="${FUNES_NATIVE_BACKFILL_PUSH_EVERY:-4}"
RECONCILE_INTERVAL="${FUNES_NATIVE_BACKFILL_RECONCILE_INTERVAL:-300}"

case "$PUSH_EVERY" in ''|*[!0-9]*|0) PUSH_EVERY=4 ;; esac
case "$RECONCILE_INTERVAL" in ''|*[!0-9]*|0) RECONCILE_INTERVAL=300 ;; esac
mkdir -p "$STATE_DIR" "$(dirname "$LOG")"
if ! mkdir "$LOCK" 2>/dev/null; then exit 0; fi
cleanup() { rmdir "$LOCK" 2>/dev/null || true; }
trap cleanup EXIT INT TERM

if [ ! -x "$BIN" ]; then
  printf '%s missing funes binary\n' "$(date '+%Y-%m-%dT%H:%M:%S%z')" >>"$LOG"
  exit 1
fi

# Prefer the macOS Keychain; the zshrc fallback is for the explicit local
# setup requested by the operator and is never echoed or persisted by this
# script.
if command -v security >/dev/null 2>&1; then
  HF_TOKEN="$(security find-generic-password -a "${USER:-$(id -un)}" -s funes-hf-token -w 2>/dev/null || true)"
  [ -n "$HF_TOKEN" ] && export HF_TOKEN
fi
if [ -z "${HF_TOKEN:-}" ] && [ -r "$HOME/.zshrc" ] && command -v zsh >/dev/null 2>&1; then
  HF_TOKEN="$(zsh -c 'source "$HOME/.zshrc" >/dev/null 2>&1; printf %s "${FUNES_HF_TOKEN:-${HF_TOKEN:-}}"' 2>/dev/null || true)"
  [ -n "$HF_TOKEN" ] && export HF_TOKEN
fi
[ -n "${HF_TOKEN:-}" ] || { printf '%s HF token unavailable\n' "$(date '+%Y-%m-%dT%H:%M:%S%z')" >>"$LOG"; exit 1; }
export FUNES_TRUFFLEHOG="${FUNES_TRUFFLEHOG:-$HOME/.local/bin/trufflehog}"

ITER_FILE="$STATE_DIR/native-backfill.iteration"
while :; do
  iteration="$(cat "$ITER_FILE" 2>/dev/null || printf '0')"
  case "$iteration" in ''|*[!0-9]*) iteration=0 ;; esac
  iteration=$((iteration + 1)); printf '%s\n' "$iteration" >"$ITER_FILE"
  changed=0
  for harness in codex pi claude; do
    tmp="$STATE_DIR/native-backfill.$$.out"
    if "$BIN" index --harness "$harness" >"$tmp" 2>&1; then
      cat "$tmp" >>"$LOG"
      grep -Eq 'chunks=[1-9][0-9]*' "$tmp" && changed=1 || true
    else
      printf '%s harness=%s index_failed\n' "$(date '+%Y-%m-%dT%H:%M:%S%z')" "$harness" >>"$LOG"
    fi
    python3 -c 'import os,sys; os.unlink(sys.argv[1])' "$tmp" 2>/dev/null || true
  done
  push_now=0
  [ "$changed" -eq 1 ] && [ $((iteration % PUSH_EVERY)) -eq 0 ] && push_now=1
  if [ "$changed" -eq 1 ] && ! "$BIN" status 2>/dev/null | grep -q 'pending indexing:'; then push_now=1; fi
  if [ "$push_now" -eq 1 ]; then
    "$BIN" push "$REMOTE" --yes >>"$LOG" 2>&1 || printf '%s push_failed; local index retained\n' "$(date '+%Y-%m-%dT%H:%M:%S%z')" >>"$LOG"
  fi
  if ! "$BIN" status 2>/dev/null | grep -q 'pending indexing:'; then
    printf '%s native backfill complete; next reconciliation in %ss\n' "$(date '+%Y-%m-%dT%H:%M:%S%z')" "$RECONCILE_INTERVAL" >>"$LOG"
    sleep "$RECONCILE_INTERVAL"
  else
    sleep "${FUNES_NATIVE_BACKFILL_SLEEP:-15}"
  fi
done
