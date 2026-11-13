from unittest.mock import Mock

import pytest

from service.roles import WriterLease, writer_lock_key


class Session:
    closed = False

    def __init__(self, granted=True):
        self.granted = granted
        self.sql = []
        self.fail = False

    def execute(self, sql, params=None):
        self.sql.append((sql, params))
        if self.fail:
            raise OSError('offline')
        return Mock(fetchone=lambda: (self.granted,), close=lambda: None)

    def close(self):
        self.closed = True


def test_lock_namespace_and_normalization():
    assert writer_lock_key(' hf://datasets/owner/name/ ') == writer_lock_key('owner/name')
    assert writer_lock_key('owner/other') != writer_lock_key('owner/name')


def test_competitor_fails_without_retry():
    session = Session(False)
    lease = WriterLease('offline', 'owner/name', connect=lambda: session)
    with pytest.raises(RuntimeError, match='lease'):
        lease.acquire()
    assert session.closed
    assert not lease.ready


def test_loss_revokes_and_never_reacquires():
    session = Session()
    lost = Mock()
    lease = WriterLease('offline', 'owner/name', connect=lambda: session, on_loss=lost)
    lease.acquire(monitor=False)
    assert lease.ready
    session.fail = True
    assert not lease.check()
    assert not lease.ready
    lost.assert_called_once()
    with pytest.raises(RuntimeError):
        lease.require()
    with pytest.raises(RuntimeError):
        lease.acquire(monitor=False)
    lease.close()


def test_release_after_work_stops():
    session = Session()
    lease = WriterLease('offline', 'owner/name', connect=lambda: session)
    lease.acquire(monitor=False)
    lease.close()
    assert session.closed
    assert any('pg_advisory_unlock' in sql for sql, _ in session.sql)
