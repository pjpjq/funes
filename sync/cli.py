from __future__ import annotations
import argparse, importlib.util, json, logging, os, re, shutil, signal, subprocess, sys
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

try:
    import tomllib
except ImportError:
    tomllib = None

from .config import Config
from .daemon import SyncDaemon
from .discovery import discover_sources
from .store import Store
from .launchd import StateMigrationBlocked, install, plist_path, plist_paths, restart as restart_agents, start as start_agents, status as launchd_status, stop as stop_agents, uninstall


def _safe_locator(value):
    text=str(value or "")
    try:
        if text.startswith("//"):
            parsed=urlsplit(text)
            if not parsed.hostname:
                return "<configured>"
            host=parsed.hostname
            if ":" in host and not host.startswith("["):
                host=f"[{host}]"
            port=f":{parsed.port}" if parsed.port is not None else ""
            return urlunsplit(("",f"{host}{port}",parsed.path,"",""))
        if "://" not in text:
            if re.match(r"^[A-Za-z][A-Za-z0-9+.-]*:/",text):
                return "<configured>"
            safe=text.split("?",1)[0].split("#",1)[0]
            if "@" in safe:
                safe=safe.rsplit("@",1)[1]
            return safe
        parsed=urlsplit(text)
        if not parsed.hostname:
            return "<configured>"
        host=parsed.hostname
        if ":" in host and not host.startswith("["):
            host=f"[{host}]"
        port=f":{parsed.port}" if parsed.port is not None else ""
        return urlunsplit((parsed.scheme,f"{host}{port}",parsed.path,"",""))
    except (TypeError, ValueError):
        return "<configured>"


def _config_diagnostic(path):
    path=Path(path)
    result={"path":str(path),"exists":path.is_file(),"parseable":False}
    if not result["exists"]:
        return result
    if tomllib is None:
        result["error"]="tomllib_unavailable"
        return result
    try:
        with path.open("rb") as handle:
            data=tomllib.load(handle)
        result["parseable"]=isinstance(data,dict)
    except (OSError, ValueError) as exc:
        result["error"]=type(exc).__name__
    return result


def _path_diagnostic(path):
    path=Path(path).expanduser()
    return {"path":str(path),"exists":path.exists(),"readable":os.access(path,os.R_OK)}


def _source_path_diagnostics(cfg):
    codex=Path(os.environ.get("CODEX_HOME",cfg.home/".codex")).expanduser()
    codex_paths=(codex/"sessions",codex/"archived_sessions",codex/"subagents",codex/"sessions"/"archive")
    claude=Path(os.environ.get("CLAUDE_CONFIG_DIR",cfg.home/".claude")).expanduser()
    claude_paths=(claude/"projects",claude/"history",claude/"memory",claude/"memories")
    pi_paths=[]
    for key in ("PI_CODING_AGENT_SESSION_DIR","PI_CODING_AGENT_DIR","PI_SESSION_DIR"):
        if os.environ.get(key):
            pi_paths.append(Path(os.environ[key]).expanduser())
    pi_paths.extend((cfg.home/".pi/agent/sessions",cfg.home/".pi/sessions",cfg.home/".pi"))
    seen=set()
    unique_pi=[]
    for path in pi_paths:
        value=str(path)
        if value not in seen:
            seen.add(value)
            unique_pi.append(path)
    return {"codex":[_path_diagnostic(path) for path in codex_paths],"pi":[_path_diagnostic(path) for path in unique_pi],"claude":[_path_diagnostic(path) for path in claude_paths]}


def _launch_agent_diagnostics(cfg):
    try:
        return launchd_status(config=cfg)
    except (OSError, RuntimeError) as exc:
        return {path.stem:{"path":str(path),"installed":path.exists(),"loaded":None,"error":type(exc).__name__} for path in plist_paths(cfg.home)}


def _auth_ready(client):
    try:
        headers=client._auth_headers()
    except Exception:
        return False
    return bool(headers.get("Authorization") or headers.get("X-Funes-Authorization"))


def _remote_ready(client):
    try:
        return bool(client.health())
    except Exception:
        return False


def _translation_diagnostic(cfg):
    mode=str(cfg.retrieval_language_mode or "").strip().lower()
    mode_valid=mode in {"raw","auto","translate"}
    configured={
        "base_url":bool(os.environ.get("TRANSLATION_BASE_URL")),
        "api_key":bool(os.environ.get("TRANSLATION_API_KEY")),
        "model":bool(os.environ.get("TRANSLATION_MODEL")),
    }
    provider_configured=all(configured.values())
    provider_partial=any(configured.values()) and not provider_configured
    return {"mode":mode if mode_valid else "invalid","mode_valid":mode_valid,"provider_required":mode=="translate","provider_configured":provider_configured,"provider_partial":provider_partial,"settings":configured}


def _watcher_diagnostic():
    try:
        available=importlib.util.find_spec("watchdog") is not None
    except (ImportError, ValueError):
        available=False
    return {"dependency":"watchdog","available":available,"polling_fallback":True}


def _native_diagnostic(cfg):
    configured=cfg.native_bin or os.environ.get("FUNES_BIN") or ""
    candidate=configured or shutil.which("funes") or str(cfg.home/".local/bin/funes")
    if candidate and os.path.sep not in candidate:
        candidate=shutil.which(candidate) or candidate
    binary=Path(candidate).expanduser() if candidate else None
    funes_home=Path(os.environ.get("FUNES_HOME",cfg.home/".funes")).expanduser()
    index_path=funes_home/"memory/chunks.lance"
    return {"enabled":bool(cfg.native_primary),"memory_configured":bool(cfg.native_memory),"binary_path":str(binary) if binary else "","binary_found":bool(binary and binary.is_file()),"binary_executable":bool(binary and binary.is_file() and os.access(binary,os.X_OK)),"index_path":str(index_path),"index_exists":index_path.is_dir()}

def main(argv=None):
    ap=argparse.ArgumentParser(prog="funes-sync")
    sub=ap.add_subparsers(dest="cmd",required=True)
    for n in ("backfill","run","drain","status","sources","doctor","install","uninstall","start","stop","restart","logs","mcp"): sub.add_parser(n)
    reindex = sub.add_parser("reindex")
    scope = reindex.add_mutually_exclusive_group(required=True)
    scope.add_argument("--retrieval-text", action="store_true")
    scope.add_argument("--all", action="store_true")
    ap.add_argument("--root", type=str, help="override the home directory for a dry-run")
    a=ap.parse_args(argv)
    selected_home=Path(a.root).expanduser() if a.root else None
    try:
        cfg=Config.load(selected_home)
    except (AttributeError, OSError, TypeError, ValueError) as exc:
        if a.cmd!="doctor":
            raise
        home=selected_home or Path(os.environ.get("HOME","~")).expanduser()
        config_path=Path(os.environ.get("FUNES_CONFIG",home/".config/funes/config.toml")).expanduser()
        config_check=_config_diagnostic(config_path)
        config_check.update({"loadable":False,"load_error":type(exc).__name__})
        print(json.dumps({"ok":False,"config":config_check},ensure_ascii=False,indent=2))
        return 2
    if a.cmd=="install":
        try:
            installed=install(config=cfg)
        except StateMigrationBlocked as exc:
            print(json.dumps(exc.as_dict(), ensure_ascii=False, indent=2), file=sys.stderr)
            return 2
        for path in installed: print(path)
        from .integrations import install_all
        print(json.dumps(install_all(cfg.home), ensure_ascii=False, indent=2))
        return 0
    if a.cmd=="uninstall":
        for path in uninstall(config=cfg): print(path)
        return 0
    if a.cmd in ("start","stop","restart"):
        action={"start":start_agents,"stop":stop_agents,"restart":restart_agents}[a.cmd]
        try:
            managed=action(config=cfg)
        except StateMigrationBlocked as exc:
            print(json.dumps(exc.as_dict(), ensure_ascii=False, indent=2), file=sys.stderr)
            return 2
        for path in managed: print(path)
        return 0
    if a.cmd=="reindex":
        from .client import SyncClient
        selected="all" if a.all else "retrieval_text"
        print(json.dumps(SyncClient(cfg).reindex(selected),ensure_ascii=False,indent=2))
        return 0
    store=Store(config=cfg)
    try:
        if a.cmd in ("backfill","run"):
            d=SyncDaemon(cfg,store)
            d.run(once=a.cmd=="backfill")
        elif a.cmd=="drain":
            d=SyncDaemon(cfg,store)
            print(json.dumps({"flushed": d.drain(), "remaining": store.pending_count()}, ensure_ascii=False))
        elif a.cmd=="status":
            from .client import SyncClient
            client=SyncClient(cfg)
            remote_ready=_remote_ready(client)
            remote_url=_safe_locator(cfg.remote_url)
            status=store.stats(); status.update({"remote_url":remote_url,"remote_status":{"url":remote_url,"ready":remote_ready},"native_memory":_safe_locator(cfg.native_memory),"native_primary":cfg.native_primary,"remote_ready":remote_ready,"device_id":cfg.device_id,"interval":cfg.interval,"launch_agents":_launch_agent_diagnostics(cfg)})
            print(json.dumps(status,ensure_ascii=False,indent=2))
        elif a.cmd=="sources": print(json.dumps([s.as_dict() for s in discover_sources(cfg)],ensure_ascii=False,indent=2))
        elif a.cmd=="doctor":
            from .client import SyncClient
            client=SyncClient(cfg)
            auth_ready=_auth_ready(client)
            remote_ready=_remote_ready(client)
            remote_url=_safe_locator(cfg.remote_url)
            launch_agents=_launch_agent_diagnostics(cfg)
            stats=store.stats()
            checks={"state_dir":str(cfg.state_dir),"db":str(store.path),"remote_url":remote_url,"native_memory":_safe_locator(cfg.native_memory),"native_primary":cfg.native_primary,"token_configured":auth_ready,"auth_ready":auth_ready,"sources":stats["sources"],"remote_ready":remote_ready,"remote_status":{"url":remote_url,"auth_ready":auth_ready,"ready":remote_ready},"config":_config_diagnostic(cfg.config_path),"source_paths":_source_path_diagnostics(cfg),"translation":_translation_diagnostic(cfg),"watcher":_watcher_diagnostic(),"launch_agent":str(plist_path(cfg.home)),"launch_agents":[str(path) for path in plist_paths(cfg.home)],"launch_agent_status":launch_agents,"native":_native_diagnostic(cfg)}
            print(json.dumps(checks,ensure_ascii=False,indent=2))
        elif a.cmd=="logs":
            p=cfg.home/"Library/Logs/funes-sync.log"; print(p.read_text(errors="replace")[-10000:] if p.exists() else "")
        elif a.cmd=="mcp":
            from .mcp_bridge import serve
            serve(store)
    finally: store.close()
    return 0
if __name__=="__main__": raise SystemExit(main())
