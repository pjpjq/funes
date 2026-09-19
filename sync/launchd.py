from __future__ import annotations
import os, plistlib, shutil, sqlite3, subprocess, sys, tempfile
from pathlib import Path
from .config import Config
try:
    import tomllib
except ImportError:
    tomllib = None

LABEL="com.funes.sync"
NATIVE_LABEL="com.funes.native-backfill"
KEYCHAIN_SERVICE="funes-api-token"
HF_KEYCHAIN_SERVICE="funes-hf-token"
KEYCHAIN_TIMEOUT=8


class StateMigrationBlocked(RuntimeError):
    def __init__(self, migrations):
        self.migrations=tuple(migrations)
        known=[item["pending"] for item in self.migrations if item["pending"] is not None]
        self.pending=sum(known) if len(known) == len(self.migrations) else None
        pending="unknown" if self.pending is None else str(self.pending)
        super().__init__(f"state migration blocked: old queue has {pending} pending records")

    def as_dict(self):
        first=self.migrations[0]
        return {"error":"state_migration_blocked", "warning":str(self), "pending":self.pending, "old_state_dir":first["old_state_dir"], "new_state_dir":first["new_state_dir"], "migrations":list(self.migrations)}


def plist_path(home=None, label=LABEL):
    return Path(home or os.environ.get("HOME","~")).expanduser()/"Library/LaunchAgents"/(label+".plist")


def native_plist_path(home=None):
    return plist_path(home, NATIVE_LABEL)


def plist_paths(home=None):
    return plist_path(home), native_plist_path(home)


def helper_paths(config):
    return config.state_dir/"native-backfill.sh", config.state_dir/"warm-space.py"


def _keychain_run(args, **kwargs):
    try:
        return subprocess.run(args, timeout=KEYCHAIN_TIMEOUT, **kwargs)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("Keychain command timed out after 8 seconds") from exc

def persist_keychain_token() -> bool:
    """Store and verify the bearer token without putting it in launchd plist data."""
    token = os.environ.get("FUNES_API_TOKEN")
    security = shutil.which("security")
    if not token or not security:
        return False
    account = os.environ.get("USER") or str(os.getuid())
    # Supplying `-w <value>` exposes the secret in process arguments.  The
    # macOS security CLI prompts twice when `-w` is the final option, so feed
    # both prompts over stdin instead.
    saved = _keychain_run(
        [security, "add-generic-password", "-U", "-a", account, "-s", KEYCHAIN_SERVICE, "-w"],
        input=f"{token}\n{token}\n", encoding="utf-8",
        check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    if saved.returncode != 0:
        raise RuntimeError("unable to store FUNES_API_TOKEN in macOS Keychain")
    check = _keychain_run(
        [security, "find-generic-password", "-a", account, "-s", KEYCHAIN_SERVICE, "-w"],
        check=False, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
    )
    if check.returncode != 0 or check.stdout.rstrip("\n") != token:
        raise RuntimeError("macOS Keychain verification failed for FUNES_API_TOKEN")
    return True

def persist_keychain_credentials() -> None:
    """Persist configured API and Hub tokens; neither is written to the plist."""
    persist_keychain_token()
    hub = os.environ.get("FUNES_HF_TOKEN") or os.environ.get("HF_TOKEN")
    security = shutil.which("security")
    if not hub or not security:
        return
    account = os.environ.get("USER") or str(os.getuid())
    saved = _keychain_run([security, "add-generic-password", "-U", "-a", account, "-s", HF_KEYCHAIN_SERVICE, "-w"], input=f"{hub}\n{hub}\n", encoding="utf-8", check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if saved.returncode != 0:
        raise RuntimeError("unable to store FUNES_HF_TOKEN in macOS Keychain")
    check = _keychain_run([security, "find-generic-password", "-a", account, "-s", HF_KEYCHAIN_SERVICE, "-w"], check=False, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    if check.returncode != 0 or check.stdout.rstrip("\n") != hub:
        raise RuntimeError("macOS Keychain verification failed for FUNES_HF_TOKEN")


def _config(home=None, config=None):
    if isinstance(home, Config):
        if config is not None:
            raise TypeError("config supplied twice")
        return home
    return config or Config.load(Path(home).expanduser() if home is not None else None)


def _config_section(config):
    if not tomllib or not config.config_path.exists():
        return {}
    try:
        with config.config_path.open("rb") as handle:
            data=tomllib.load(handle)
    except (OSError, ValueError):
        return {}
    section=data.get("sync", {})
    return section if section else data


def _tuning(config, env_key, previous, config_key=None, config_attr=None):
    explicit=os.environ.get(env_key)
    if explicit:
        return explicit
    section=_config_section(config) if config_key else {}
    configured=section.get(config_key) if isinstance(section, dict) and config_key else None
    if configured is not None and str(configured):
        return str(configured)
    if config_attr:
        value=getattr(config, config_attr)
        default=Config.__dataclass_fields__[config_attr].default
        if value != default:
            return str(value)
    old=previous.get(env_key) if isinstance(previous, dict) else None
    if isinstance(old, (str, int, float)) and str(old):
        return str(old)
    if config_attr:
        return str(getattr(config, config_attr))
    return ""


def render_plist(python=None, config=None, previous_env=None):
    python=python or os.environ.get("PYTHON", sys.executable)
    config=config or Config.load()
    root=Path(__file__).resolve().parents[1]
    launcher=root / "bin" / "funes-sync"
    # launchd does not guarantee that HOME/USER are inherited for a GUI
    # LaunchAgent.  The launcher uses both values when resolving the login
    # Keychain, so make the lookup deterministic without persisting secrets.
    # launchd does not inherit the interactive shell's PATH.  Pin the
    # interpreter selected at install time so the daemon does not silently
    # fall back to Apple's system Python (which may lack the user's packages
    # and has different Keychain/runtime behavior).
    previous_env=previous_env or {}
    env={"PYTHON":str(Path(python).expanduser()), "PYTHONPATH":str(root), "HOME":str(config.home), "USER":os.environ.get("USER") or str(os.getuid()), "FUNES_CONFIG":str(config.config_path), "FUNES_REMOTE_URL":config.remote_url, "FUNES_MEMORY_ONLY":"1", "FUNES_STATE_DIR":str(config.state_dir), "FUNES_SYNC_BATCH":_tuning(config, "FUNES_SYNC_BATCH", previous_env, "batch_size", "batch_size"), "FUNES_SYNC_MAX_BATCH_BYTES":_tuning(config, "FUNES_SYNC_MAX_BATCH_BYTES", previous_env, "max_batch_bytes", "max_batch_bytes")}
    # The launcher loads FUNES_API_TOKEN from macOS Keychain at runtime; never
    # persist the bearer token in a world-readable plist.
    for key, config_key in (
        ("FUNES_REMOTE_TIMEOUT", "remote_timeout"),
        ("FUNES_REMOTE_ATTEMPT_TIMEOUT", "remote_attempt_timeout"),
        ("FUNES_REMOTE_TRANSIENT_RETRIES", "remote_transient_retries"),
    ):
        value=_tuning(config, key, previous_env, config_key)
        if value:
            env[key]=value
    return {"Label":LABEL,"ProgramArguments":[str(launcher),"run"],"EnvironmentVariables":env,"WorkingDirectory":str(root),"RunAtLoad":True,"KeepAlive":True,"StandardOutPath":str(config.home/"Library/Logs/funes-sync.log"),"StandardErrorPath":str(config.home/"Library/Logs/funes-sync.err.log")}


def render_native_plist(config, python=None, previous_env=None):
    if not config.native_memory:
        raise RuntimeError("native memory is not configured")
    python=python or os.environ.get("PYTHON", sys.executable)
    root=Path(__file__).resolve().parents[1]
    native_script, warm_helper=helper_paths(config)
    native_bin=config.native_bin or shutil.which("funes") or config.home/".local/bin/funes"
    env={"PYTHON":str(Path(python).expanduser()), "HOME":str(config.home), "USER":os.environ.get("USER") or str(os.getuid()), "FUNES_BIN":str(Path(native_bin).expanduser()), "FUNES_NATIVE_MEMORY":config.native_memory, "FUNES_SYNC_STATE_DIR":str(config.state_dir), "FUNES_REMOTE_URL":config.remote_url, "FUNES_NATIVE_BACKFILL_RECONCILE_INTERVAL":str(config.interval), "FUNES_NATIVE_WARM_HELPER":str(warm_helper)}
    previous_env=previous_env or {}
    for key, config_key in (("FUNES_NATIVE_BACKFILL_PUSH_EVERY", "native_backfill_push_every"), ("FUNES_NATIVE_BACKFILL_SLEEP", "native_backfill_sleep")):
        value=_tuning(config, key, previous_env, config_key)
        if value:
            env[key]=value
    return {"Label":NATIVE_LABEL,"ProgramArguments":[str(native_script)],"EnvironmentVariables":env,"WorkingDirectory":str(root),"RunAtLoad":True,"KeepAlive":True,"ThrottleInterval":30,"StandardOutPath":str(config.home/"Library/Logs/funes-native-backfill.log"),"StandardErrorPath":str(config.home/"Library/Logs/funes-native-backfill.err.log")}


def _read_plists(paths):
    result={}
    for p, label in paths:
        if not p.exists():
            continue
        try:
            existing=plistlib.loads(p.read_bytes())
        except Exception as exc:
            raise RuntimeError(f"refusing to overwrite unreadable LaunchAgent {p}: {exc}")
        if existing.get("Label") != label:
            raise RuntimeError(f"refusing to overwrite unrelated LaunchAgent {p}")
        result[label]=existing
    return result


def _validate_plists(paths):
    _read_plists(paths)


def _atomic_write(path, content, mode):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary=tempfile.mkstemp(prefix=f".{path.name}.tmp-", dir=path.parent)
    temporary_path=Path(temporary)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        temporary_path.chmod(mode)
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _launchctl(action, value):
    uid=str(os.getuid())
    try:
        result=subprocess.run(["launchctl",action,f"gui/{uid}",str(value)],check=False,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    except OSError as exc:
        raise RuntimeError(f"launchctl {action} is unavailable") from exc
    if result.returncode != 0:
        raise RuntimeError(f"launchctl {action} failed with exit code {result.returncode}")


def _launchctl_service(action, label):
    uid=str(os.getuid())
    try:
        result=subprocess.run(
            ["launchctl",action,f"gui/{uid}/{label}"],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except OSError as exc:
        raise RuntimeError(f"launchctl {action} is unavailable") from exc
    if result.returncode != 0:
        raise RuntimeError(f"launchctl {action} failed with exit code {result.returncode}")


def _launchctl_status(label):
    uid=str(os.getuid())
    try:
        result=subprocess.run(["launchctl","print",f"gui/{uid}/{label}"],check=False,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    except OSError as exc:
        raise RuntimeError("launchctl print is unavailable") from exc
    if result.returncode == 113:
        return False
    if result.returncode != 0:
        raise RuntimeError(f"launchctl print failed with exit code {result.returncode}")
    return True


def _owned_plists(config):
    owned=_all_owned_plists(config)
    return owned if config.native_primary else owned[:1]


def _all_owned_plists(config):
    return tuple(zip(plist_paths(config.home), (LABEL, NATIVE_LABEL)))


def _managed_paths(config):
    return tuple(path for path, _label in _owned_plists(config))


def _plist_environment(plist):
    env=plist.get("EnvironmentVariables", {}) if isinstance(plist, dict) else {}
    return env if isinstance(env, dict) else {}


def _state_path(value, home, working_directory=None):
    text=str(value)
    if text == "~":
        return home
    if text.startswith("~/"):
        return home/text[2:]
    path=Path(text).expanduser()
    if path.is_absolute():
        return path
    base=_state_path(working_directory, home) if working_directory else home
    return base/path


def _same_path(left, right):
    try:
        if Path(left).exists() and Path(right).exists():
            return os.path.samefile(left, right)
    except OSError:
        pass
    try:
        return Path(left).resolve(strict=False) == Path(right).resolve(strict=False)
    except OSError:
        return os.path.abspath(left) == os.path.abspath(right)


def _installed_helper_paths(config, existing):
    """Return current helpers plus old helpers proven to belong to our plist."""
    paths=list(helper_paths(config))
    plist=existing.get(NATIVE_LABEL, {})
    env=_plist_environment(plist)
    state_value=env.get("FUNES_SYNC_STATE_DIR")
    if not state_value:
        return tuple(paths)
    old_home=_state_path(env.get("HOME", config.home), config.home)
    working=plist.get("WorkingDirectory")
    state_dir=_state_path(state_value, old_home, working)
    expected=(state_dir/"native-backfill.sh", state_dir/"warm-space.py")
    args=plist.get("ProgramArguments")
    references=(
        args[0] if isinstance(args,list) and args else None,
        env.get("FUNES_NATIVE_WARM_HELPER"),
    )
    for reference, candidate in zip(references, expected):
        if not reference:
            continue
        referenced=_state_path(reference, old_home, working)
        if (
            os.path.abspath(referenced) == os.path.abspath(candidate)
            and not any(_same_path(candidate, current) for current in paths)
        ):
            paths.append(candidate)
    return tuple(paths)


def _pending_queue_count(state_dir):
    database_path=state_dir/"sync.db"
    try:
        database_path.lstat()
    except FileNotFoundError:
        return 0
    except OSError as exc:
        raise RuntimeError(f"unable to verify pending queue in {state_dir}") from exc
    try:
        resolved=database_path.resolve(strict=True)
        if not resolved.is_file():
            raise RuntimeError(f"unable to verify pending queue in {state_dir}")
        uri=resolved.as_uri()+"?mode=ro"
        with sqlite3.connect(uri, uri=True, timeout=1) as database:
            database.execute("PRAGMA query_only=ON")
            exists=database.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='queue'").fetchone()
            if not exists:
                return 0
            return int(database.execute("SELECT count(*) FROM queue").fetchone()[0])
    except (OSError, sqlite3.Error) as exc:
        raise RuntimeError(f"unable to verify pending queue in {state_dir}") from exc


def _resolve_old_state(state_dir):
    try:
        state_dir.lstat()
    except FileNotFoundError:
        return state_dir.resolve()
    except OSError as exc:
        raise RuntimeError(f"unable to resolve old state directory {state_dir}") from exc
    try:
        return state_dir.resolve(strict=True)
    except OSError as exc:
        raise RuntimeError(f"unable to resolve old state directory {state_dir}") from exc


def _check_state_migration(config, existing):
    candidates=[]
    for label, key in ((LABEL, "FUNES_STATE_DIR"), (NATIVE_LABEL, "FUNES_SYNC_STATE_DIR")):
        plist=existing.get(label, {})
        env=_plist_environment(plist)
        value=env.get(key)
        if value:
            old_home=_state_path(env.get("HOME", config.home), config.home)
            candidates.append(_state_path(value, old_home, plist.get("WorkingDirectory")))
    new_state=config.state_dir.resolve()
    migrations=[]
    seen=set()
    for old_state in candidates:
        try:
            normalized=_resolve_old_state(old_state)
        except RuntimeError:
            display=old_state.absolute()
            if display not in seen:
                seen.add(display)
                migrations.append({"old_state_dir":str(display), "new_state_dir":str(new_state), "pending":None, "reason":"queue_unreadable"})
            continue
        if normalized == new_state or normalized in seen:
            continue
        seen.add(normalized)
        try:
            pending=_pending_queue_count(normalized)
            reason="pending_queue" if pending else "empty_queue"
        except RuntimeError:
            pending=None
            reason="queue_unreadable"
        if pending:
            migrations.append({"old_state_dir":str(normalized), "new_state_dir":str(new_state), "pending":pending, "reason":reason})
        elif pending is None:
            migrations.append({"old_state_dir":str(normalized), "new_state_dir":str(new_state), "pending":None, "reason":reason})
    if migrations:
        raise StateMigrationBlocked(migrations)


def _snapshot_files(paths):
    snapshots={}
    for path in paths:
        if path.exists():
            snapshots[path]=(path.read_bytes(), path.stat().st_mode & 0o7777)
        else:
            snapshots[path]=(None, None)
    return snapshots


def _restore_files(snapshots):
    for path, (content, mode) in snapshots.items():
        if content is None:
            if path.exists():
                path.unlink()
            continue
        _atomic_write(path, content, mode)


def _loaded_state(owned):
    return {label:_launchctl_status(label) for _path, label in owned}


def _stop_owned(owned):
    for path, label in reversed(owned):
        if _launchctl_status(label):
            if path.exists():
                _launchctl("bootout", path)
            else:
                _launchctl_service("bootout", label)


def _rollback_transaction(owned, snapshots, original_loaded):
    errors=[]
    for path, label in reversed(owned):
        try:
            if _launchctl_status(label):
                _launchctl("bootout", path)
        except Exception as exc:
            errors.append(f"stop {label}: {type(exc).__name__}")
    try:
        _restore_files(snapshots)
    except Exception as exc:
        errors.append(f"restore files: {type(exc).__name__}")
    for path, label in owned:
        try:
            loaded=_launchctl_status(label)
            if loaded:
                _launchctl("bootout", path)
            if original_loaded[label]:
                _launchctl("bootstrap", path)
        except Exception as exc:
            errors.append(f"restore {label}: {type(exc).__name__}")
    return errors


def _start_owned(owned):
    started=[]
    try:
        for path, label in owned:
            if not _launchctl_status(label):
                _launchctl("bootstrap", path)
                started.append((path, label))
    except RuntimeError as exc:
        rollback_error=None
        for path, _label in reversed(started):
            try:
                _launchctl("bootout", path)
            except RuntimeError as current:
                rollback_error=current
        if rollback_error is not None:
            raise RuntimeError(f"{exc}; startup rollback failed: {rollback_error}") from exc
        raise


def install(home=None, *, config=None, python=None):
    config=_config(home, config)
    main_path, native_path=plist_paths(config.home)
    native_script, warm_helper=helper_paths(config)
    owned=_owned_plists(config)
    all_owned=_all_owned_plists(config)
    existing=_read_plists(all_owned)
    _check_state_migration(config, existing)
    root=Path(__file__).resolve().parents[1]
    source_script=root/"deploy/funes-sync/native-backfill.sh"
    source_helper=root/"deploy/funes-sync/warm-space.py"
    script_content=source_script.read_bytes()
    helper_content=source_helper.read_bytes()
    installed_helpers=_installed_helper_paths(config, existing)
    main_env=_plist_environment(existing.get(LABEL, {}))
    native_env=_plist_environment(existing.get(NATIVE_LABEL, {}))
    main_content=plistlib.dumps(render_plist(python, config, main_env))
    native_content=(
        plistlib.dumps(render_native_plist(config, python, native_env))
        if config.native_primary
        else None
    )
    snapshots=_snapshot_files((*installed_helpers, main_path, native_path))
    original_loaded=_loaded_state(all_owned)
    for path, label in all_owned:
        if original_loaded[label] and snapshots[path][0] is None and (path,label) in owned:
            raise RuntimeError(f"cannot safely replace loaded LaunchAgent without plist {path}")
        if original_loaded[label] and snapshots[path][0] is None:
            # A loaded legacy helper without its plist cannot be restored
            # exactly. In non-native mode it is an unwanted service, so fail
            # safe by stopping it and excluding it from rollback restoration.
            original_loaded[label]=False
    try:
        _stop_owned(all_owned)
        _check_state_migration(config, existing)
        # This file is owned by us; never overwrite unrelated launch agents.
        persist_keychain_credentials()
        _atomic_write(main_path, main_content, 0o600)
        if native_content is not None:
            _atomic_write(native_script, script_content, 0o700)
            _atomic_write(warm_helper, helper_content, 0o700)
            _atomic_write(native_path, native_content, 0o600)
        for path in installed_helpers:
            current_helper=any(
                _same_path(path, current) for current in (native_script, warm_helper)
            )
            if (native_content is None or not current_helper) and path.exists():
                path.unlink()
        if native_content is None:
            for path in (native_path,):
                if path.exists():
                    path.unlink()
        _start_owned(owned)
    except Exception as exc:
        rollback_errors=_rollback_transaction(all_owned, snapshots, original_loaded)
        if rollback_errors:
            raise RuntimeError(f"{exc}; rollback failed: {'; '.join(rollback_errors)}") from exc
        raise
    return _managed_paths(config)


def start(home=None, *, config=None):
    config=_config(home, config)
    paths=_managed_paths(config)
    if not all(path.exists() for path in paths) or (
        not config.native_primary
        and (
            native_plist_path(config.home).exists()
            or _launchctl_status(NATIVE_LABEL)
        )
    ):
        return install(config=config)
    owned=_owned_plists(config)
    _validate_plists(owned)
    _start_owned(owned)
    return paths


def stop(home=None, *, config=None):
    config=_config(home, config)
    paths=_managed_paths(config)
    owned=_all_owned_plists(config)
    _validate_plists(owned)
    _stop_owned(owned)
    return paths


def restart(home=None, *, config=None):
    config=_config(home, config)
    paths=_managed_paths(config)
    if not all(path.exists() for path in paths) or (
        not config.native_primary
        and (
            native_plist_path(config.home).exists()
            or _launchctl_status(NATIVE_LABEL)
        )
    ):
        return install(config=config)
    owned=_owned_plists(config)
    _validate_plists(owned)
    snapshots=_snapshot_files((*helper_paths(config), *paths))
    original_loaded=_loaded_state(owned)
    try:
        _stop_owned(owned)
        _start_owned(owned)
    except Exception as exc:
        rollback_errors=_rollback_transaction(owned, snapshots, original_loaded)
        if rollback_errors:
            raise RuntimeError(f"{exc}; rollback failed: {'; '.join(rollback_errors)}") from exc
        raise
    return paths


def status(home=None, *, config=None):
    config=_config(home, config)
    return {label:{"path":str(path), "installed":path.exists(), "loaded":_launchctl_status(label)} for path, label in _all_owned_plists(config)}


def uninstall(home=None, *, config=None):
    config=_config(home, config)
    paths=plist_paths(config.home)
    all_owned=_all_owned_plists(config)
    existing=_read_plists(all_owned)
    installed_helpers=_installed_helper_paths(config, existing)
    stop(config=config)
    for path in paths:
        if path.exists():
            path.unlink()
    for path in installed_helpers:
        if path.exists():
            path.unlink()
    return paths
