import json
from types import SimpleNamespace

import pytest

import sync.cli as cli
import sync.client as client_module


def test_reindex_requires_exactly_one_scope(monkeypatch, tmp_path, capsys):
    config = SimpleNamespace(home=tmp_path)
    monkeypatch.setattr(cli.Config, "load", classmethod(lambda cls, home=None: config))
    calls = []

    class Client:
        def __init__(self, selected):
            assert selected is config

        def reindex(self, scope):
            calls.append(scope)
            return {"queued": True, "durable": True, "scope": scope, "generation": 7}

    monkeypatch.setattr(client_module, "SyncClient", Client)

    with pytest.raises(SystemExit) as missing:
        cli.main(["reindex"])
    assert missing.value.code == 2
    with pytest.raises(SystemExit) as conflicting:
        cli.main(["reindex", "--retrieval-text", "--all"])
    assert conflicting.value.code == 2

    assert cli.main(["reindex", "--retrieval-text"]) == 0
    assert json.loads(capsys.readouterr().out)["scope"] == "retrieval_text"
    assert cli.main(["reindex", "--all"]) == 0
    assert json.loads(capsys.readouterr().out)["scope"] == "all"
    assert calls == ["retrieval_text", "all"]
