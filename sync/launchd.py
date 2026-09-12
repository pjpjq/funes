from __future__ import annotations
import os, plistlib, shutil, subprocess, sys
from pathlib import Path
LABEL="com.funes.sync"
KEYCHAIN_SERVICE="funes-api-token"
HF_KEYCHAIN_SERVICE="funes-hf-token"
def plist_path(home=None): return Path(home or os.environ.get("HOME","~")).expanduser()/"Library/LaunchAgents"/(LABEL+".plist")

def persist_keychain_token() -> bool:
    """Store and verify the bearer token without putting it in launchd plist data."""
    token = os.environ.get("FUNES_API_TOKEN")
    security = shutil.which("security")
    if not token or not security:
        return False
    account = os.environ.get("USER") or str(os.getuid())
    # `security` has no stdin password mode; the value is passed only to this
    # short-lived process and is never written to a file or logged.
    saved = subprocess.run(
        [security, "add-generic-password", "-U", "-a", account, "-s", KEYCHAIN_SERVICE, "-w", token],
        check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    if saved.returncode != 0:
        raise RuntimeError("unable to store FUNES_API_TOKEN in macOS Keychain")
    check = subprocess.run(
        [security, "find-generic-password", "-a", account, "-s", KEYCHAIN_SERVICE, "-w"],
        check=False, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
    )
    if check.returncode != 0 or check.stdout.rstrip("\n") != token:
        raise RuntimeError("macOS Keychain verification failed for FUNES_API_TOKEN")
    return True

def persist_keychain_credentials() -> None:
    """Persist configured API and Hub tokens; neither is written to the plist."""
    persist_keychain_token()
    hub = os.environ.get("FUNES_HF_TOKEN")
    security = shutil.which("security")
    if not hub or not security:
        return
    account = os.environ.get("USER") or str(os.getuid())
    saved = subprocess.run([security, "add-generic-password", "-U", "-a", account, "-s", HF_KEYCHAIN_SERVICE, "-w", hub], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if saved.returncode != 0:
        raise RuntimeError("unable to store FUNES_HF_TOKEN in macOS Keychain")
    check = subprocess.run([security, "find-generic-password", "-a", account, "-s", HF_KEYCHAIN_SERVICE, "-w"], check=False, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    if check.returncode != 0 or check.stdout.rstrip("\n") != hub:
        raise RuntimeError("macOS Keychain verification failed for FUNES_HF_TOKEN")
def render_plist(python=None):
    python=python or os.environ.get("PYTHON", sys.executable)
    root=Path(__file__).resolve().parents[1]
    launcher=root / "bin" / "funes-sync"
    env={"PYTHONPATH":str(root)}
    # The launcher loads FUNES_API_TOKEN from macOS Keychain at runtime; never
    # persist the bearer token in a world-readable plist.
    for key in ("FUNES_REMOTE_URL",):
        value=os.environ.get(key)
        if value:
            env[key]=value
    return {"Label":LABEL,"ProgramArguments":[str(launcher),"run"],"EnvironmentVariables":env,"WorkingDirectory":str(root),"RunAtLoad":True,"KeepAlive":True,"StandardOutPath":str(Path.home()/"Library/Logs/funes-sync.log"),"StandardErrorPath":str(Path.home()/"Library/Logs/funes-sync.err.log")}
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
    persist_keychain_credentials()
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
