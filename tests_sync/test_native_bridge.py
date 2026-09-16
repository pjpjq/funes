from pathlib import Path

from sync.config import Config
from sync.native import NativeFunes


def test_native_adapter_uses_upstream_index_and_push(tmp_path):
    calls = []

    class Result:
        returncode = 0

    def runner(args, **kwargs):
        calls.append(args)
        return Result()

    cfg = Config(Path(tmp_path), Path(tmp_path) / "state", Path(tmp_path) / "config", native_memory="owner/memory", native_primary=True, native_bin="/usr/local/bin/funes")
    result = NativeFunes(cfg, runner=runner).sync(("codex", "pi"))
    assert result.ok
    assert calls == [
        ["/usr/local/bin/funes", "index", "--harness", "codex", "--yes"],
        ["/usr/local/bin/funes", "index", "--harness", "pi", "--yes"],
        ["/usr/local/bin/funes", "push", "owner/memory", "--yes", "--force-reindex"],
    ]
