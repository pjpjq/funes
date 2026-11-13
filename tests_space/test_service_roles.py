import inspect
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from service.roles import service_role
from service.server import App
from service.server import Store
from service.postgres import PostgresStore
import space.server as bridge


def test_roles_fail_closed(monkeypatch):
    monkeypatch.delenv('FUNES_SERVICE_ROLE', raising=False)
    assert service_role() == 'combined'
    monkeypatch.setenv('FUNES_SERVICE_ROLE', 'typo')
    with pytest.raises(ValueError):
        service_role()


@pytest.mark.parametrize('role', ['query', 'writer'])
def test_split_requires_pg(monkeypatch, role):
    monkeypatch.setenv('FUNES_SERVICE_ROLE', role)
    monkeypatch.delenv('FUNES_POSTGRES_DSN', raising=False)
    with pytest.raises(ValueError, match='FUNES_POSTGRES_DSN'):
        App()


def test_query_app_never_starts_background(monkeypatch):
    import service.postgres as pg
    monkeypatch.setenv('FUNES_SERVICE_ROLE', 'query')
    monkeypatch.setenv('FUNES_POSTGRES_DSN', 'offline-fake')
    store = SimpleNamespace(close=Mock(), verify_schema=Mock())
    monkeypatch.setattr(pg, 'PostgresStore', Mock(return_value=store))
    start = Mock(side_effect=AssertionError('writer thread started'))
    monkeypatch.setattr('threading.Thread.start', start)
    app = App()
    assert app.restore_done.is_set()
    assert app.reconcile_thread is None
    assert app.reindex_thread is None
    app.close()
    store.close.assert_called_once()


def test_writer_cannot_warm(monkeypatch):
    monkeypatch.setenv('FUNES_SERVICE_ROLE', 'writer')
    monkeypatch.setattr(bridge, '_warm_native_memory', Mock(side_effect=AssertionError('warm')))
    assert bridge.request_warm()['state'] == 'disabled'
    with pytest.raises(RuntimeError, match='writer'):
        bridge.native_worker()


def test_query_disables_bucket_checkpoint(monkeypatch):
    monkeypatch.setenv('FUNES_SERVICE_ROLE', 'query')
    factory = Mock(side_effect=AssertionError('persistent cache opened'))
    monkeypatch.setattr(bridge.HubCache, 'from_env', factory)
    bridge.start_hub_cache()
    factory.assert_not_called()


def test_checkpoint_override_keeps_base_signature():
    assert inspect.signature(PostgresStore.native_index_checkpoint) == inspect.signature(Store.native_index_checkpoint)
    assert inspect.signature(PostgresStore.translation_put) == inspect.signature(Store.translation_put)
    store = object.__new__(PostgresStore)
    store.read_only = True
    store.status_snapshot = Mock(return_value={'canonical_index': {'complete': False}})
    assert store.native_index_checkpoint({'fingerprint': 'existing'}) == {'complete': False}
    store.status_snapshot.assert_called_once_with({'fingerprint': 'existing'}, '')


def test_query_translation_cache_is_bounded_and_ephemeral():
    store = object.__new__(PostgresStore)
    store.read_only = True
    store.lock = threading.RLock()
    store._query_translation_cache = {}
    store.conn = Mock(side_effect=AssertionError('SQL write'))
    for i in range(1025):
        store.translation_put(str(i), 'translated', 'ok', 'hash', 'version')
    assert len(store._query_translation_cache) == 1024
    assert store.translation_get('1024') == 'translated'
    store.conn.execute.assert_not_called()


@pytest.mark.parametrize('role,route', [('query', '/ingest'), ('query', '/sync'),
                                     ('query', '/reindex'), ('writer', '/search'),
                                     ('writer', '/warm')])
def test_bridge_rejects_wrong_role_before_body(monkeypatch, role, route):
    monkeypatch.setenv('FUNES_SERVICE_ROLE', role)
    monkeypatch.setattr(bridge, 'auth_ok', lambda _: True)
    handler = SimpleNamespace(path=route, send_json=Mock(),
                              body=Mock(side_effect=AssertionError('body read')))
    bridge.Handler.do_POST(handler)
    assert handler.send_json.call_args.args[0] == 403
    handler.body.assert_not_called()


@pytest.mark.parametrize('failure', ['translator', 'env', 'lease_loss'])
def test_writer_startup_failure_closes_store_before_lease_and_starts_no_work(monkeypatch, failure):
    import service.roles as roles
    import service.postgres as pg
    import service.server as source
    order = []
    class Lease:
        ready = True
        def __init__(self, *args, on_loss, **kwargs):
            self.on_loss = on_loss
        def acquire(self, **kwargs):
            assert kwargs == {'monitor': False}
        def require(self):
            if not self.ready:
                raise RuntimeError('lost lease')
        def start_monitor(self):
            app = self.on_loss.__self__
            assert app.reconcile_stop is not None
            assert app.reindex_stop is not None
            if failure == 'lease_loss':
                self.ready = False
                self.on_loss()
        def start_thread(self, thread):
            raise AssertionError('background started')
        def close(self):
            order.append('lease')
    store = SimpleNamespace(conn=SimpleNamespace(write_guard=None),
                            close=lambda: order.append('store'))
    monkeypatch.setenv('FUNES_SERVICE_ROLE', 'writer')
    monkeypatch.setenv('FUNES_POSTGRES_DSN', 'offline-fake')
    monkeypatch.setenv('FUNES_MEMORY', 'owner/name')
    monkeypatch.setattr(roles, 'WriterLease', Lease)
    monkeypatch.setattr(pg, 'PostgresStore', Mock(return_value=store))
    if failure == 'translator':
        monkeypatch.setattr(source, 'Translator', Mock(side_effect=ValueError('provider env')))
    elif failure == 'env':
        monkeypatch.setenv('FUNES_REINDEX_BATCH_SIZE', 'invalid')
    with pytest.raises((ValueError, RuntimeError)):
        App()
    assert order == ['store', 'lease']


def test_stopped_writer_readiness_cannot_recreate_app(monkeypatch):
    monkeypatch.setenv('FUNES_SERVICE_ROLE', 'writer')
    monkeypatch.setenv('FUNES_POSTGRES_DSN', 'offline-fake')
    stopped = threading.Event()
    stopped.set()
    monkeypatch.setattr(bridge, 'SERVICE_STOP', stopped)
    monkeypatch.setattr(bridge, 'SOURCE_APP', None)
    factory = Mock(side_effect=AssertionError('writer resurrected'))
    monkeypatch.setattr(bridge, 'SourceApp', factory)
    assert bridge.source_app() is None
    factory.assert_not_called()
