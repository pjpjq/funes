from __future__ import annotations
import argparse, json, logging, os, signal, subprocess, sys
from pathlib import Path
from .config import Config
from .daemon import SyncDaemon
from .discovery import discover_sources
from .store import Store
from .launchd import install, uninstall, plist_path

def main(argv=None):
    ap=argparse.ArgumentParser(prog="funes-sync")
    sub=ap.add_subparsers(dest="cmd",required=True)
    for n in ("backfill","run","status","sources","doctor","install","uninstall","start","stop","restart","logs","mcp"): sub.add_parser(n)
    ap.add_argument("--root", type=str, help="override the home directory for a dry-run")
    a=ap.parse_args(argv); cfg=Config.load(Path(a.root).expanduser() if a.root else None); store=Store(config=cfg)
    try:
        if a.cmd in ("backfill","run"):
            d=SyncDaemon(cfg,store)
            d.run(once=a.cmd=="backfill")
        elif a.cmd=="status":
            from .client import SyncClient
            status=store.stats(); status.update({"remote_url":cfg.remote_url,"remote_ready":SyncClient(cfg).health(),"device_id":cfg.device_id,"interval":cfg.interval})
            print(json.dumps(status,ensure_ascii=False,indent=2))
        elif a.cmd=="sources": print(json.dumps([s.as_dict() for s in discover_sources(cfg)],ensure_ascii=False,indent=2))
        elif a.cmd=="doctor":
            from .client import SyncClient
            checks={"state_dir":str(cfg.state_dir),"db":str(store.path),"remote_url":cfg.remote_url,"token_configured":bool(os.environ.get("FUNES_API_TOKEN")),"sources":store.stats()["sources"],"remote_ready":SyncClient(cfg).health(),"launch_agent":str(plist_path(cfg.home))}
            print(json.dumps(checks,ensure_ascii=False,indent=2))
        elif a.cmd=="install":
            print(install(cfg.home))
            from .integrations import install_all
            print(json.dumps(install_all(cfg.home), ensure_ascii=False, indent=2))
        elif a.cmd=="uninstall": print(uninstall(cfg.home))
        elif a.cmd in ("start","stop","restart"):
            path=plist_path(cfg.home); uid=str(os.getuid())
            if a.cmd in ("stop","restart"):
                subprocess.run(["launchctl","bootout",f"gui/{uid}",str(path)],check=False,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
            if a.cmd in ("start","restart"):
                if not path.exists(): install(cfg.home)
                subprocess.run(["launchctl","bootstrap",f"gui/{uid}",str(path)],check=False,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
            print(path)
        elif a.cmd=="logs":
            p=cfg.home/"Library/Logs/funes-sync.log"; print(p.read_text(errors="replace")[-10000:] if p.exists() else "")
        elif a.cmd=="mcp":
            from .mcp_bridge import serve
            serve(store)
    finally: store.close()
    return 0
if __name__=="__main__": raise SystemExit(main())
