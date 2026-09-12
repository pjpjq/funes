from __future__ import annotations
import hashlib, os, socket
from dataclasses import dataclass
from pathlib import Path
try:
    import tomllib
except ImportError:
    tomllib = None

@dataclass
class Config:
    home: Path
    state_dir: Path
    config_path: Path
    remote_url: str = "http://127.0.0.1:7860"
    interval: int = 300
    batch_size: int = 50
    concurrency: int = 2
    device_id: str = ""
    enabled: bool = True
    initial_backfill: bool = True
    auto_discover: bool = True
    source_codex: bool = True
    source_pi: bool = True
    retrieval_language_mode: str = "auto"
    native_memory: str = ""
    native_primary: bool = False
    native_bin: str = ""
    memory_only: bool = False
    def __post_init__(self):
        raw = self.device_id or os.environ.get("FUNES_DEVICE_ID") or socket.gethostname()
        self.device_id = "dev-" + hashlib.sha256(("funes:" + raw).encode()).hexdigest()[:20]
    @classmethod
    def load(cls, home: Path|None = None) -> "Config":
        home = Path(home or os.environ.get("HOME", "~")).expanduser()
        cfg = Path(os.environ.get("FUNES_CONFIG", home / ".config/funes/config.toml")).expanduser()
        data = {}
        if cfg.exists() and tomllib:
            try:
                with cfg.open("rb") as f: data = tomllib.load(f)
            except (OSError, ValueError): data = {}
        section = data.get("sync", {})
        remote = data.get("remote", {})
        retrieval = data.get("retrieval", {})
        sources = data.get("sources", {})
        codex_source = sources.get("codex", {}).get("enabled", section.get("source_codex", True))
        pi_source = sources.get("pi", {}).get("enabled", section.get("source_pi", True))
        # Keep the config compatible with the documented TOML shape while accepting
        # the older flat `[sync]` form used by the first bridge release.
        if not section:
            section = data
        truth=lambda v: str(v).lower() not in ("0", "false", "no", "off")
        return cls(home, Path(os.environ.get("FUNES_STATE_DIR", home / ".local/share/funes-sync")).expanduser(), cfg,
                   os.environ.get("FUNES_REMOTE_URL", remote.get("url", section.get("remote_url", "http://127.0.0.1:7860"))),
                   int(os.environ.get("FUNES_SYNC_INTERVAL", section.get("interval", 300))),
                   int(os.environ.get("FUNES_SYNC_BATCH", section.get("batch_size", 50))),
                   int(os.environ.get("FUNES_SYNC_CONCURRENCY", section.get("concurrency", 2))),
                   os.environ.get("FUNES_DEVICE_ID", section.get("device_id", retrieval.get("device_id", remote.get("device_id", "")))),
                   truth(os.environ.get("FUNES_SYNC_ENABLED", section.get("enabled", True))),
                   truth(os.environ.get("FUNES_SYNC_INITIAL_BACKFILL", section.get("initial_backfill", True))),
                   truth(os.environ.get("FUNES_SYNC_AUTO_DISCOVER", section.get("auto_discover", True))),
                   truth(os.environ.get("FUNES_SOURCE_CODEX", os.environ.get("FUNES_SYNC_SOURCE_CODEX", codex_source))),
                   truth(os.environ.get("FUNES_SOURCE_PI", os.environ.get("FUNES_SYNC_SOURCE_PI", pi_source))),
                   os.environ.get("FUNES_RETRIEVAL_LANGUAGE_MODE", retrieval.get("language_mode", section.get("retrieval_language_mode", "auto"))),
                   os.environ.get("FUNES_MEMORY", remote.get("memory", "")),
                   truth(os.environ.get("FUNES_NATIVE_PRIMARY", section.get("native_primary", False))),
                   os.environ.get("FUNES_BIN", section.get("native_bin", "")),
                   truth(os.environ.get("FUNES_MEMORY_ONLY", section.get("memory_only", False))))
    def ensure(self):
        self.state_dir.mkdir(parents=True, exist_ok=True)
