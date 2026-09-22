from types import SimpleNamespace

from scripts.monitor_source_postgres_rehearsal import main, section, snapshot


class Connection:
    def __init__(self, broken=False):
        self.queries = []
        self.broken = broken

    def cursor(self):
        return self

    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass

    def execute(self, query, parameters):
        self.queries.append((query, parameters))
        if self.broken:
            raise RuntimeError("postgresql://secret@host/private raw source text")
        self.description = [SimpleNamespace(name="sample")]

    def fetchall(self):
        return [(1,)]


def test_observations_do_not_scan_or_write_source_rows():
    conn = Connection()
    result = snapshot(conn)
    assert result["checkpoint"] == [{"sample": 1}]
    for query, _ in conn.queries:
        assert query.lstrip().upper().startswith("SELECT")
        assert "FROM memories" not in query
        assert "raw_text" not in query
        assert "COUNT(" not in query.upper()


def test_lock_errors_do_not_leak_credentials_or_payload():
    assert section(Connection(broken=True), "SELECT anything") == {
        "unavailable": True, "sqlstate": None}


def test_cli_sanitizes_connect_errors(monkeypatch, capsys):
    from scripts import monitor_source_postgres_rehearsal as monitor
    def fail(_):
        raise RuntimeError("postgresql://secret@host private raw text")
    monkeypatch.setenv("FUNES_POSTGRES_DSN", "private-secret")
    monkeypatch.setattr(monitor, "connect", fail)
    assert main([]) == 0
    output = capsys.readouterr().out
    assert "monitor_connection_unavailable" in output
    assert "secret" not in output
    assert "raw text" not in output


def test_read_only_is_set_after_read_write_target_selection(monkeypatch):
    import psycopg
    from scripts import monitor_source_postgres_rehearsal as monitor

    events = []

    class FakeConnection:
        def cursor(self):
            return self

        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

        def execute(self, query):
            events.append(("execute", query))

        def close(self):
            events.append(("close",))

    connection = FakeConnection()

    def select_target(dsn, **kwargs):
        events.append(("connect", dsn, kwargs))
        # Simulate libpq target selection rejecting a read-only startup session.
        if "default_transaction_read_only" in kwargs.get("options", ""):
            raise psycopg.OperationalError("target_session_attrs=read-write rejected read-only session")
        return connection

    monkeypatch.setattr(psycopg, "connect", select_target)
    dsn = "postgresql://user:password@localhost/db?target_session_attrs=read-write"
    assert monitor.connect(dsn) is connection
    assert events[0][0:2] == ("connect", dsn)
    options = events[0][2]["options"]
    assert "statement_timeout=5000" in options
    assert "lock_timeout=500" in options
    assert events[0][2]["autocommit"] is True
    assert events[0][2]["connect_timeout"] == 10
    assert events[1:] == [("execute", "SET default_transaction_read_only=on")]


def test_replica_connection_failure_preserves_primary_snapshot(monkeypatch, capsys):
    import json
    from scripts import monitor_source_postgres_rehearsal as monitor

    class ReplicaConnectionError(RuntimeError):
        sqlstate = "08001"

    primary = Connection()
    calls = []

    def connect(dsn):
        calls.append(dsn)
        if dsn == "replica-secret-dsn":
            raise ReplicaConnectionError("postgresql://secret:credential@host private raw text")
        return primary

    monkeypatch.setenv("FUNES_POSTGRES_DSN", "primary-secret-dsn")
    monkeypatch.setenv("FUNES_POSTGRES_READ_DSN", "replica-secret-dsn")
    monkeypatch.setattr(monitor, "connect", connect)
    assert main([]) == 0
    output = capsys.readouterr().out
    result = json.loads(output)
    assert calls == ["primary-secret-dsn", "replica-secret-dsn"]
    assert result["checkpoint"] == [{"sample": 1}]
    assert result["database"] == [{"sample": 1}]
    assert result["replica"] == {
        "unavailable": True, "error_class": "ReplicaConnectionError", "sqlstate": "08001",
    }
    assert "error" not in result
    assert "secret" not in output
    assert "credential" not in output
    assert "raw text" not in output
    assert "replica_lag" not in result
    assert len(primary.queries) == 10


def test_primary_connection_error_has_only_safe_diagnostics(monkeypatch, capsys):
    import json
    from scripts import monitor_source_postgres_rehearsal as monitor

    class PrimaryConnectionError(RuntimeError):
        sqlstate = "08006"

    def fail(_dsn):
        raise PrimaryConnectionError("postgresql://password@privatehost private raw text")

    monkeypatch.setenv("FUNES_POSTGRES_DSN", "private-secret")
    monkeypatch.delenv("FUNES_POSTGRES_READ_DSN", raising=False)
    monkeypatch.setattr(monitor, "connect", fail)
    assert main([]) == 0
    output = capsys.readouterr().out
    result = json.loads(output)
    assert result["error"] == "monitor_connection_unavailable"
    assert result["primary"] == {
        "unavailable": True, "error_class": "PrimaryConnectionError", "sqlstate": "08006",
    }
    assert "password" not in output and "privatehost" not in output and "raw text" not in output


def test_read_only_setup_failure_closes_connection(monkeypatch):
    import psycopg
    import pytest
    from scripts import monitor_source_postgres_rehearsal as monitor

    class FakeConnection:
        closed = False

        def cursor(self):
            return self

        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

        def execute(self, _query):
            raise RuntimeError("read-only setup failed")

        def close(self):
            self.closed = True

    connection = FakeConnection()
    monkeypatch.setattr(psycopg, "connect", lambda *_args, **_kwargs: connection)
    with pytest.raises(RuntimeError, match="read-only setup failed"):
        monitor.connect("unused-fake-dsn")
    assert connection.closed


def test_primary_only_snapshot_succeeds_and_closes_connection(monkeypatch, capsys):
    import json
    from scripts import monitor_source_postgres_rehearsal as monitor

    class ManagedConnection(Connection):
        closed = False

        def __exit__(self, *_):
            self.closed = True

    primary = ManagedConnection()
    monkeypatch.setenv("FUNES_POSTGRES_DSN", "primary-secret-dsn")
    monkeypatch.delenv("FUNES_POSTGRES_READ_DSN", raising=False)
    monkeypatch.setattr(monitor, "connect", lambda _dsn: primary)
    assert main([]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["checkpoint"] == [{"sample": 1}]
    assert "replica" not in result and "error" not in result
    assert primary.closed


def test_copy_activity_remains_visible_when_importer_locks_metadata():
    class LockUnavailable(RuntimeError):
        sqlstate = "55P03"

    class LockedMetadataConnection(Connection):
        def execute(self, query, parameters):
            if "FROM funes_source_migration" in query or "FROM funes_schema_state" in query:
                raise LockUnavailable("private COPY source row must not leak")
            super().execute(query, parameters)

    primary = LockedMetadataConnection()
    result = snapshot(primary)
    assert result["checkpoint"] == {"unavailable": True, "sqlstate": "55P03"}
    assert result["readiness"] == {"unavailable": True, "sqlstate": "55P03"}
    assert result["copy_progress"] == [{"sample": 1}]
    assert result["migration_activity"] == [{"sample": 1}]
    copy_sql = next(query for query, _ in primary.queries if "pg_stat_progress_copy" in query)
    assert "SELECT pid,command,type,bytes_processed,bytes_total,tuples_processed" in copy_sql
    assert "WHERE datname=current_database()" in copy_sql
    activity_sql = next(query for query, _ in primary.queries if "pg_stat_activity" in query)
    assert "SELECT pid,state,wait_event_type,wait_event" in activity_sql
    assert "WHERE datname=current_database()" in activity_sql
    assert "application_name='funes-source-migration'" in activity_sql
    assert "query" not in activity_sql.lower()
