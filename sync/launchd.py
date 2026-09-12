from __future__ import annotations
import os, plistlib, subprocess, sys
from pathlib import Path
LABEL="com.funes.sync"
def plist_path(home=None): return Path(home or os.environ.get("HOME","~")).expanduser()/"Library/LaunchAgents"/(LABEL+".plist")
def render_plist(python=None):
    python=python or os.environ.get("PYTHON", sys.executable)
    root=Path(__file__).resolve().parents[1]
    launcher=root / "bin" / "funes-sync"
    return {"Label":LABEL,"ProgramArguments":[str(launcher),"run"],"EnvironmentVariables":{"PYTHONPATH":str(root)},"WorkingDirectory":str(root),"RunAtLoad":True,"KeepAlive":True,"StandardOutPath":str(Path.home()/"Library/Logs/funes-sync.log"),"StandardErrorPath":str(Path.home()/"Library/Logs/funes-sync.err.log")}
def install(home=None):
    p=plist_path(home); p.parent.mkdir(parents=True,exist_ok=True)
    if p.exists():
        try:
            existing=plistlib.loads(p.read_bytes())
        except Exception as exc:
            raise RuntimeError(f"refusing to overwrite unreadable LaunchAgent {p}: {exc}")
        if existing.get("Label") != LABEL:
            raise RuntimeError(f"refusing to overwrite unrelated LaunchAgent {p}")
    # This file is owned by us; never overwrite unrelated launch agents.
    with p.open("wb") as f: plistlib.dump(render_plist(),f)
    try:
        uid=str(os.getuid())
        subprocess.run(["launchctl","bootstrap",f"gui/{uid}",str(p)],check=False,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    except OSError: pass
    return p
def uninstall(home=None):
    p=plist_path(home)
    try:
        uid=str(os.getuid())
        subprocess.run(["launchctl","bootout",f"gui/{uid}",str(p)],check=False,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    except OSError: pass
    if p.exists():
        try:
            current=plistlib.loads(p.read_bytes())
        except Exception as exc:
            raise RuntimeError(f"refusing to remove unreadable LaunchAgent {p}: {exc}")
        if current.get("Label") == LABEL:
            p.unlink()
    return p
