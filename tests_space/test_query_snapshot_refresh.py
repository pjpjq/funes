import json
import os
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import space.server as bridge

from service.roles import QuerySnapshot, QuerySnapshotRefresher


def snapshot(version=1, sha='a'):
    return QuerySnapshot.parse({'schema_version': 1, 'memory': 'owner/name',
                                'hf_sha': sha * 40, 'lance_version': version}, 'owner/name')


@pytest.mark.parametrize('field,value', [('hf_sha', 'main'), ('lance_version', 0),
                                       ('lance_version', True), ('schema_version', 2),
                                       ('memory', 'other/name'), ('lance_version', 1 << 64)])
def test_invalid_snapshot(field, value):
    data = dict(schema_version=1, memory='owner/name', hf_sha='a' * 40, lance_version=1)
    data[field] = value
    with pytest.raises(ValueError):
        QuerySnapshot.parse(data, 'owner/name')


def test_only_lance_changes_warm_and_failure_keeps_active():
    values = iter([snapshot(), snapshot(1, 'b'), snapshot(2, 'c'), snapshot(2, 'd')])
    install = Mock(side_effect=[None, RuntimeError('candidate failed'), None])
    refresher = QuerySnapshotRefresher(lambda: next(values), install)
    assert refresher.poll_once()
    active = refresher.active
    assert not refresher.poll_once()
    assert not refresher.poll_once()
    assert refresher.active == active
    assert refresher.poll_once()
    assert refresher.active == snapshot(2, 'd')
    assert install.call_count == 3


def test_child_environment_does_not_mutate_global(monkeypatch):
    monkeypatch.delenv('FUNES_QUERY_HF_REVISION', raising=False)
    env = snapshot().environment({'untouched': 'yes'})
    assert env['FUNES_QUERY_HF_REVISION'] == 'a' * 40
    assert env['FUNES_QUERY_LANCE_VERSION'] == '1'
    assert 'FUNES_QUERY_HF_REVISION' not in os.environ


def test_same_version_can_recover_without_changing_snapshot():
    install = Mock()
    refresher = QuerySnapshotRefresher(lambda: snapshot(), install)
    refresher.poll_once()
    refresher.refresh_requested.set()
    assert refresher.poll_once()
    assert install.call_count == 2


def test_single_candidate_and_coalesced_refresh():
    entered, finish = threading.Event(), threading.Event()
    def install(value):
        entered.set()
        assert finish.wait(2)
    installer = Mock(side_effect=install)
    refresher = QuerySnapshotRefresher(lambda: snapshot(), installer, interval=60)
    refresher.start()
    try:
        assert entered.wait(2)
        for _ in range(20):
            refresher.wake.set()
        assert installer.call_count == 1
        finish.set()
    finally:
        finish.set()
        refresher.close()
    assert not refresher.thread.is_alive()


def test_failed_bridge_candidate_preserves_old_worker_and_config(monkeypatch):
    monkeypatch.setenv('FUNES_SERVICE_ROLE', 'query')
    monkeypatch.setattr(bridge, 'SERVICE_STOP', threading.Event())
    old = SimpleNamespace(process=SimpleNamespace(poll=lambda: None), close=Mock())
    original_config = bridge._native_worker_config()
    monkeypatch.setattr(bridge, 'MCP_WORKER', old)
    monkeypatch.setattr(bridge, '_MCP_WORKER_CONFIG', original_config)
    candidate = Mock()
    candidate.recall.side_effect = RuntimeError('open failed')
    monkeypatch.setattr(bridge, 'NativeMcpWorker', Mock(return_value=candidate))
    with pytest.raises(RuntimeError):
        bridge._refresh_native_worker(snapshot(2))
    assert bridge.MCP_WORKER is old
    assert bridge._MCP_WORKER_CONFIG == original_config
    candidate.close.assert_called_once()
    assert bridge._native_search_serviceable({'state': 'error'})


def test_failed_hot_refresh_keeps_bridge_ready_200_and_old_worker_open(monkeypatch):
    monkeypatch.setenv('FUNES_SERVICE_ROLE', 'query')
    monkeypatch.setattr(bridge, 'SERVICE_STOP', threading.Event())
    monkeypatch.setattr(bridge, 'REMOTE', 'owner/name')
    monkeypatch.setattr(bridge, '_WARM_STATE', {'state': 'ready'})
    monkeypatch.setattr(bridge, 'source_readiness_state',
                        lambda: {'configured': True, 'ready': True})
    old = SimpleNamespace(process=SimpleNamespace(poll=lambda: None), close=Mock())
    monkeypatch.setattr(bridge, 'MCP_WORKER', old)
    original_config = bridge._native_worker_config()
    monkeypatch.setattr(bridge, '_MCP_WORKER_CONFIG', original_config)
    candidate = Mock()
    candidate.recall.side_effect = RuntimeError('candidate failed')
    monkeypatch.setattr(bridge, 'NativeMcpWorker', Mock(return_value=candidate))
    monkeypatch.setattr(bridge, 'native_worker_status', lambda: {})
    with pytest.raises(RuntimeError):
        bridge._install_query_snapshot(snapshot(2))
    assert bridge.ready_payload()[0] == 200
    assert bridge.search_ready_payload()[0] == 200
    assert bridge.MCP_WORKER is old
    assert bridge._MCP_WORKER_CONFIG == original_config
    old.close.assert_not_called()
    candidate.close.assert_called_once()


def test_query_request_never_resolves_snapshot_or_spawns_worker(monkeypatch):
    monkeypatch.setenv('FUNES_SERVICE_ROLE', 'query')
    old = SimpleNamespace(query_snapshot=snapshot())
    monkeypatch.setattr(bridge, 'MCP_WORKER', old)
    describe = Mock(side_effect=AssertionError('metadata in request'))
    monkeypatch.setattr(bridge, '_describe_query_snapshot', describe)
    assert bridge.native_worker() is old
    describe.assert_not_called()


def test_background_metadata_budget_allows_observed_cold_open_without_changing_query_budget(monkeypatch):
    monkeypatch.setattr(bridge, 'REMOTE', 'owner/name')
    query_timeout = bridge.MCP_TIMEOUT
    descriptor = dict(schema_version=1, memory='owner/name', hf_sha='a' * 40, lance_version=79552)
    run = Mock(return_value=(0, json.dumps(descriptor), ''))
    monkeypatch.setattr(bridge, 'run', run)
    assert bridge._describe_query_snapshot().lance_version == 79552
    run.assert_called_once_with('query-snapshot', 'owner/name', timeout=60)
    assert bridge.MCP_TIMEOUT == query_timeout
