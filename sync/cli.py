from __future__ import annotations
import argparse, json, logging, os, signal, subprocess, sys
from pathlib import Path
from .config import Config
from .daemon import SyncDaemon
from .discovery import discover_sources
from .store import Store
from .launchd import StateMigrationBlocked, install, plist_path, plist_paths, restart as restart_agents, start as start_agents, status as launchd_status, stop as stop_agents, uninstall

def main(argv=None):
    ap=argparse.ArgumentParser(prog="funes-sync")
    sub=ap.add_subparsers(dest="cmd",required=True)
    for n in ("backfill","run","drain","status","sources","doctor","install","uninstall","start","stop","restart","logs","mcp"): sub.add_parser(n)
    reindex = sub.add_parser("reindex")
    scope = reindex.add_mutually_exclusive_group(required=True)
    scope.add_argument("--retrieval-text", action="store_true")
    scope.add_argument("--all", action="store_true")
    ap.add_argument("--root", type=str, help="override the home directory for a dry-run")
    a=ap.parse_args(argv); cfg=Config.load(Path(a.root).expanduser() if a.root else None)
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
            status=store.stats(); status.update({"remote_url":cfg.remote_url,"native_memory":cfg.native_memory,"native_primary":cfg.native_primary,"remote_ready":SyncClient(cfg).health(),"device_id":cfg.device_id,"interval":cfg.interval,"launch_agents":launchd_status(config=cfg)})
            print(json.dumps(status,ensure_ascii=False,indent=2))
        elif a.cmd=="sources": print(json.dumps([s.as_dict() for s in discover_sources(cfg)],ensure_ascii=False,indent=2))
        elif a.cmd=="doctor":
            from .client import SyncClient
            checks={"state_dir":str(cfg.state_dir),"db":str(store.path),"remote_url":cfg.remote_url,"native_memory":cfg.native_memory,"native_primary":cfg.native_primary,"token_configured":bool(os.environ.get("FUNES_API_TOKEN")),"sources":store.stats()["sources"],"remote_ready":SyncClient(cfg).health(),"launch_agent":str(plist_path(cfg.home)),"launch_agents":[str(path) for path in plist_paths(cfg.home)]}
            print(json.dumps(checks,ensure_ascii=False,indent=2))
        elif a.cmd=="logs":
            p=cfg.home/"Library/Logs/funes-sync.log"; print(p.read_text(errors="replace")[-10000:] if p.exists() else "")
        elif a.cmd=="mcp":
            from .mcp_bridge import serve
            serve(store)
    finally: store.close()
    return 0
if __name__=="__main__": raise SystemExit(main())
