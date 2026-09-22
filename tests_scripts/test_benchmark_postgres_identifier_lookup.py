import ast
import contextlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import benchmark_postgres_identifier_lookup as benchmark


def index_plan(*, execution_ms=0.75, count=2):
    return [{
        "Plan": {
            "Node Type": "Bitmap Heap Scan",
            "Actual Rows": count,
            "Actual Loops": 1,
            "Shared Hit Blocks": 9,
            "Shared Read Blocks": 2,
            "Rows Removed by Filter": 3,
            "Filter": "raw_text = 'RAW_MUST_NOT_ESCAPE'",
            "Recheck Cond": "source_identity = 'DO_NOT_COPY_PLAN_EXPRESSION'",
            "Plans": [{
                "Node Type": "Bitmap Index Scan",
                "Index Name": "memories_source_identity_key",
                "Actual Rows": count,
                "Actual Loops": 1,
                "Shared Hit Blocks": 4,
                "Shared Read Blocks": 1,
            }],
        },
        "Planning Time": 0.25,
        "Execution Time": execution_ms,
        "Query Text": "SELECT 'RAW_MUST_NOT_ESCAPE'",
    }]


def test_plan_summary_keeps_evidence_but_not_raw_expressions():
    report = benchmark.summarize_plan(index_plan())
    assert report["node_types"] == ["Bitmap Heap Scan", "Bitmap Index Scan"]
    assert report["index_names"] == ["memories_source_identity_key"]
    assert report["canonical_index_used"] is True
    assert report["sequential_scan"] is False
    # Parent buffer totals include child work; summing nodes would double-count.
    assert report["buffers"]["shared_hit_blocks"] == 9
    assert report["buffers"]["shared_read_blocks"] == 2
    assert report["rows_removed_total"] == 3
    assert report["server_execution_ms"] == 0.75
    assert report["nodes"][1]["buffers"]["shared_hit_blocks"] == 4
    encoded = json.dumps(report)
    assert "RAW_MUST_NOT_ESCAPE" not in encoded
    assert "DO_NOT_COPY_PLAN_EXPRESSION" not in encoded
    assert "Filter" not in encoded


def test_nearest_rank_percentiles_are_explicit_and_handle_one_sample():
    summary = benchmark.latency_summary(list(range(1, 21)))
    assert summary == {"samples": 20, "p50_ms": 10.0, "p95_ms": 19.0, "max_ms": 20.0}
    assert benchmark.latency_summary([3.5])["p95_ms"] == 3.5
    with pytest.raises(benchmark.BenchmarkError, match="empty_latency_samples"):
        benchmark.latency_summary([])


def case_payload(identities=None):
    return [{"label": "content_identifier", "query": "previous_response_id",
             "identities": ["canonical-a", "canonical-b"] if identities is None else identities,
             "identity_source": "frozen_sqlite_sample"}]


def test_cases_are_independent_inputs_and_batch_limits_do_not_change_sql_semantics():
    cases = benchmark.parse_cases({"cases": case_payload(["canonical-b", "canonical-a", "canonical-b"])})
    assert cases[0]["identities"] == ["canonical-b", "canonical-a"]
    lookups = benchmark.build_lookups(cases[0], [10, 100])
    assert lookups[0]["sql"] == "SELECT * FROM memories WHERE source_identity=%s ORDER BY id LIMIT 1"
    assert lookups[0]["identities"] == ["canonical-b"]
    assert lookups[1]["sql"] == "SELECT * FROM memories WHERE source_identity IN (%s,%s)"
    assert lookups[1]["identities"] == ["canonical-b", "canonical-a"]
    assert lookups[1]["requested_limit"] == 10
    assert lookups[2]["requested_limit"] == 100
    assert lookups[2]["requested_limit_filled"] is False
    assert all("previous_response_id" not in lookup["sql"] for lookup in lookups)


@pytest.mark.parametrize("identities", [[], [""], [" "], [None], [1], ["id\x00bad"], "canonical-a"])
def test_empty_or_noncanonical_identifiers_are_rejected(identities):
    with pytest.raises(benchmark.BenchmarkError, match="invalid_case_identities"):
        benchmark.parse_cases(case_payload(identities))


def test_native_reference_is_not_misrepresented_as_canonical_identity():
    with pytest.raises(benchmark.BenchmarkError, match="canonical_identities_required"):
        benchmark.parse_cases(case_payload(["funes-doc:encoded-id"]))


@pytest.mark.parametrize("repeats", [0, -1, 21, 200, True, 1.5])
def test_repeats_cannot_exceed_the_cost_bound(repeats):
    with pytest.raises(benchmark.BenchmarkError, match="invalid_repeats"):
        benchmark.validate_config(repeats, [10, 100])


@pytest.mark.parametrize("limits", [[], [0], [101], [True], [10, 10], [1, 2, 3, 4, 5, 6]])
def test_batch_limits_are_bounded(limits):
    with pytest.raises(benchmark.BenchmarkError, match="invalid_batch_limits"):
        benchmark.validate_config(20, limits)


class FakeConnection:
    """A DB boundary that records SQL and carries deliberately sensitive rows."""

    def __init__(self, *, plan_factory=None, missing=False, seqscan_setting="on", error=None, set_error=None):
        self.plan_factory = plan_factory or (lambda index, count: index_plan(execution_ms=index + 1, count=count))
        self.missing = missing
        self.seqscan_setting = seqscan_setting
        self.error = error
        self.set_error = set_error
        self.calls = []
        self.plan_count = 0
        self.preflight_count = 0
        self.closed = False
        self.cursors = []

    def cursor(self):
        cursor = FakeCursor(self)
        self.cursors.append(cursor)
        return cursor

    def close(self):
        self.closed = True


class FakeCursor:
    def __init__(self, connection):
        self.connection = connection
        self.description = [SimpleNamespace(name="source_identity"), SimpleNamespace(name="raw_text")]
        self.closed = False
        self.rows = []

    def execute(self, sql, parameters=None):
        self.connection.calls.append((sql, parameters))
        if sql == "SET default_transaction_read_only=on":
            if self.connection.set_error:
                raise self.connection.set_error
            self.rows = []
        elif sql.startswith("SELECT current_setting("):
            self.rows = [("on", self.connection.seqscan_setting, "on", "on", "on")]
        elif sql.startswith("EXPLAIN (FORMAT JSON) "):
            plan = self.connection.plan_factory(self.connection.plan_count, len(parameters))
            self.connection.preflight_count += 1
            def estimates_only(node):
                node["Plan Rows"] = node.pop("Actual Rows")
                node.pop("Actual Loops")
                for child in node.get("Plans", []):
                    estimates_only(child)
            estimates_only(plan[0]["Plan"])
            plan[0].pop("Execution Time")
            plan[0].pop("Planning Time")
            self.rows = [(plan,)]
        elif sql.startswith("EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) "):
            if self.connection.error:
                raise self.connection.error
            plan = self.connection.plan_factory(self.connection.plan_count, len(parameters))
            self.connection.plan_count += 1
            self.rows = [(plan,)]
        elif sql.startswith("SELECT * FROM memories WHERE source_identity"):
            selected = parameters[:-1] if self.connection.missing else parameters
            self.rows = [(identity, "SOURCE_RAW_MUST_NEVER_BE_REPORTED") for identity in selected]
        else:
            raise AssertionError("unexpected SQL at database boundary")

    def fetchone(self):
        return self.rows.pop(0) if self.rows else None

    def fetchall(self):
        rows, self.rows = self.rows, []
        return rows

    def close(self):
        self.closed = True


def run_fake(connection, **overrides):
    config = {"dsn": "postgresql://private:DSN_SECRET@localhost/db", "repeats": 2,
              "batch_limits": [10], "connect_factory": lambda dsn: connection,
              "clock": iter([value / 10 for value in range(1, 1000)]).__next__}
    config.update(overrides)
    return benchmark.run_benchmark(case_payload(), **config)


def test_repeats_measure_both_server_and_real_full_fetch_without_raw_output():
    connection = FakeConnection()
    report = run_fake(connection)
    first, batch = report["cases"][0]["lookups"]
    assert report["ok"] is True
    assert report["postgres_content_search"] is False
    assert report["identity_discovery_is_external"] is True
    assert first["server_execution"] == {"samples": 2, "p50_ms": 1.0, "p95_ms": 2.0, "max_ms": 2.0}
    assert batch["server_execution"]["samples"] == 2
    assert batch["client_rtt_fetch"]["samples"] == 2
    assert batch["client_rtt_fetch"]["p95_ms"] == pytest.approx(100)
    assert connection.plan_count == 4
    assert len([sql for sql, _ in connection.calls if sql.startswith("SELECT *")]) == 4
    assert connection.closed and all(cursor.closed for cursor in connection.cursors)
    serialized = json.dumps(report)
    assert "SOURCE_RAW_MUST_NEVER_BE_REPORTED" not in serialized
    assert "RAW_MUST_NOT_ESCAPE" not in serialized
    assert "DSN_SECRET" not in serialized
    assert all("previous_response_id" not in str(params) for _, params in connection.calls)
    assert not any("enable_seqscan=off" in sql for sql, _ in connection.calls)


def test_nested_seqscan_fails_before_fetch_or_additional_repeats():
    def with_seqscan(index, count):
        plan = index_plan(count=count)
        plan[0]["Plan"]["Plans"].append({
            "Node Type": "Gather", "Actual Rows": count, "Actual Loops": 1,
            "Plans": [{"Node Type": "Seq Scan", "Actual Rows": count,
                       "Actual Loops": 2, "Rows Removed by Filter": 100}],
        })
        return plan
    connection = FakeConnection(plan_factory=with_seqscan)
    report = run_fake(connection)
    assert report["ok"] is False
    first = report["cases"][0]["lookups"][0]
    assert first["failure"] == "sequential_scan"
    assert first["preflight_plans"][0]["sequential_scan"] is True
    assert first["preflight_plans"][0]["analyzed"] is False
    assert "server_execution_ms" not in first["preflight_plans"][0]
    assert benchmark.summarize_plan(with_seqscan(0, 1))["rows_removed_total"] == 203
    assert first["server_execution"] is None
    assert connection.preflight_count == 1
    assert connection.plan_count == 0
    assert not any(sql.startswith("SELECT *") for sql, _ in connection.calls)


def test_an_unrelated_index_cannot_satisfy_canonical_hydration_acceptance():
    def wrong_index(index, count):
        plan = index_plan(count=count)
        plan[0]["Plan"]["Plans"][0]["Index Name"] = "memories_pkey"
        return plan
    report = run_fake(FakeConnection(plan_factory=wrong_index))
    assert report["ok"] is False
    assert report["cases"][0]["lookups"][0]["failure"] == "canonical_index_not_used"


def test_fast_empty_result_is_not_a_successful_hydration():
    report = run_fake(FakeConnection(missing=True))
    assert report["ok"] is False
    assert report["cases"][0]["lookups"][0]["failure"] == "hydration_identity_mismatch"


def test_p95_at_the_100ms_goal_is_not_reported_as_passing():
    connection = FakeConnection(plan_factory=lambda index, count: index_plan(execution_ms=100, count=count))
    report = run_fake(connection)
    assert report["ok"] is False
    assert report["cases"][0]["lookups"][0]["failure"] == "server_p95_target_missed"


def test_preexisting_planner_suppression_is_rejected_not_silently_used():
    connection = FakeConnection(seqscan_setting="off")
    with pytest.raises(benchmark.BenchmarkError, match="planner_settings_not_default"):
        run_fake(connection)
    assert connection.plan_count == 0
    assert connection.closed


def test_connect_selects_read_write_target_before_setting_read_only(monkeypatch):
    connection = FakeConnection()
    dsn = "postgresql://owner:SECRET@host1,host2/db?target_session_attrs=read-write"
    def connect(actual_dsn, **kwargs):
        assert actual_dsn == dsn
        assert "default_transaction_read_only" not in kwargs["options"]
        assert kwargs["autocommit"] is True
        assert "statement_timeout=10000" in kwargs["options"]
        assert "lock_timeout=1000" in kwargs["options"]
        assert connection.calls == []
        return connection
    monkeypatch.setitem(sys.modules, "psycopg", SimpleNamespace(connect=connect))
    assert benchmark._connect_read_only(dsn) is connection
    assert connection.calls == [("SET default_transaction_read_only=on", None)]
    assert all(cursor.closed for cursor in connection.cursors)
    assert not connection.closed


def test_failed_read_only_setup_closes_connection_before_return(monkeypatch):
    connection = FakeConnection(set_error=RuntimeError("SECRET"))
    monkeypatch.setitem(sys.modules, "psycopg", SimpleNamespace(connect=lambda *args, **kwargs: connection))
    with pytest.raises(benchmark.BenchmarkError, match="^database_connect_failed$"):
        benchmark._connect_read_only("postgresql://owner:SECRET@host1,host2/db?target_session_attrs=read-write")
    assert connection.closed
    assert all(cursor.closed for cursor in connection.cursors)
    assert connection.calls == [("SET default_transaction_read_only=on", None)]


def test_cli_defaults_to_twenty_samples_and_private_output(tmp_path, monkeypatch, capsys):
    connection = FakeConnection(plan_factory=lambda index, count: index_plan(count=count))
    connect_arguments = []
    def connect(dsn, **kwargs):
        connect_arguments.append((dsn, kwargs))
        return connection
    monkeypatch.setitem(sys.modules, "psycopg", SimpleNamespace(connect=connect))
    secret = "postgresql://private:CLI_DSN_SECRET@host/db"
    monkeypatch.setenv("FUNES_POSTGRES_DSN", secret)
    cases_file = tmp_path / "cases.json"
    cases_file.write_text(json.dumps(case_payload()), encoding="utf-8")
    report_file = tmp_path / "report.json"
    assert benchmark.main(["--cases", str(cases_file), "--output", str(report_file)]) == 0
    report = json.loads(report_file.read_text())
    assert report["repeats"] == 20
    assert report["complete"] is True
    assert report["completed_lookups"] == report["planned_lookups"] == 3
    assert connection.plan_count == 60
    assert connection.preflight_count == 60
    assert all(lookup["server_execution"]["samples"] == 20 for lookup in report["cases"][0]["lookups"])
    kwargs = connect_arguments[0][1]
    assert kwargs["autocommit"] is True
    assert kwargs["prepare_threshold"] is None
    assert "default_transaction_read_only" not in kwargs["options"]
    assert connection.calls[0] == ("SET default_transaction_read_only=on", None)
    assert "statement_timeout=10000" in kwargs["options"]
    assert "lock_timeout=1000" in kwargs["options"]
    assert kwargs["connect_timeout"] == 10
    assert report_file.stat().st_mode & 0o777 == 0o600
    captured = capsys.readouterr()
    assert secret not in captured.out + captured.err + report_file.read_text()


@pytest.mark.parametrize("failure_stage", ["connect", "set", "explain"])
def test_cli_failures_overwrite_stale_success_without_exposing_dsn_or_raw(
    tmp_path, monkeypatch, capsys, failure_stage,
):
    secret = "postgresql://private:FAILED_DSN_SECRET@host/db"
    error = RuntimeError(secret + " source raw: FAILED_RAW_SECRET")
    connection = FakeConnection(error=error, set_error=error if failure_stage == "set" else None)
    def connect(*args, **kwargs):
        if failure_stage == "connect":
            raise error
        return connection
    monkeypatch.setitem(sys.modules, "psycopg", SimpleNamespace(connect=connect))
    monkeypatch.setenv("FUNES_POSTGRES_DSN", secret)
    cases_file = tmp_path / "cases.json"
    cases_file.write_text(json.dumps(case_payload()), encoding="utf-8")
    report_file = tmp_path / "report.json"
    report_file.write_text('{"ok": true}')
    assert benchmark.main(["--cases", str(cases_file), "--output", str(report_file)]) == 2
    assert json.loads(report_file.read_text())["ok"] is False
    captured = capsys.readouterr()
    combined = captured.out + captured.err + report_file.read_text()
    assert "FAILED_DSN_SECRET" not in combined
    assert "FAILED_RAW_SECRET" not in combined
    assert "Traceback" not in combined
    if failure_stage in {"set", "explain"}:
        assert connection.closed


def test_cli_rejects_dsn_argument_without_echoing_it(capsys):
    secret = "postgresql://user:ARGUMENT_SECRET@host/db"
    assert benchmark.main(["--dsn", secret]) == 2
    captured = capsys.readouterr()
    assert secret not in captured.out + captured.err
    assert "invalid_arguments" in captured.err


def test_unknown_error_codes_cannot_inject_sensitive_error_text():
    assert benchmark._safe_failure(benchmark.BenchmarkError("UNTRUSTED_SECRET")) == "benchmark_failed"


def test_cli_output_failure_is_sanitized_without_traceback(tmp_path, monkeypatch, capsys):
    cases_file = tmp_path / "cases.json"
    cases_file.write_text(json.dumps(case_payload()), encoding="utf-8")
    monkeypatch.setenv("FUNES_POSTGRES_DSN", "DSN_SECRET")
    connection = FakeConnection(plan_factory=lambda index, count: index_plan(count=count))
    monkeypatch.setitem(sys.modules, "psycopg", SimpleNamespace(connect=lambda *args, **kwargs: connection))
    assert benchmark.main(["--cases", str(cases_file), "--output", str(tmp_path), "--repeats", "1"]) == 2
    captured = capsys.readouterr()
    assert "output_write_failed" in captured.err
    assert "DSN_SECRET" not in captured.err
    assert connection.closed


def test_benchmark_sql_matches_the_real_store_methods_without_constructing_a_store():
    """Execute only the existing methods at a fake DB seam; no Store startup DDL."""
    root = Path(__file__).resolve().parents[1]
    statements = []
    class CaptureConnection:
        def execute(self, sql, parameters):
            statements.append((sql, list(parameters)))
            rows = [{"source_identity": identity} for identity in parameters]
            return SimpleNamespace(fetchone=lambda: rows[0], fetchall=lambda: rows)
        def close(self):
            pass
    conn = CaptureConnection()
    store = SimpleNamespace(conn=conn, lock=contextlib.nullcontext(),
                            _read_connection=lambda: conn, _row=lambda row: row)
    for filename, class_name, method_name, arguments in (
        ("service/postgres.py", "PostgresStore", "get", "canonical-a"),
        ("service/server.py", "Store", "get_many", ["canonical-a", "canonical-b"]),
    ):
        tree = ast.parse((root / filename).read_text())
        store_class = next(item for item in tree.body if isinstance(item, ast.ClassDef) and item.name == class_name)
        method = next(item for item in store_class.body if isinstance(item, ast.FunctionDef) and item.name == method_name)
        namespace = {"Any": object, "closing": contextlib.closing}
        exec(compile(ast.Module(body=[method], type_ignores=[]), filename, "exec"), namespace)
        namespace[method_name](store, arguments)
    lookups = benchmark.build_lookups(benchmark.parse_cases(case_payload())[0], [10])
    assert [(sql.replace("?", "%s"), ids) for sql, ids in statements] == [
        (lookup["sql"], lookup["identities"]) for lookup in lookups
    ]


def test_a_later_repeat_cannot_hide_a_planner_regression():
    def changing_plan(index, count):
        plan = index_plan(count=count)
        if index:
            plan[0]["Plan"]["Node Type"] = "Seq Scan"
        return plan
    connection = FakeConnection(plan_factory=changing_plan)
    report = run_fake(connection)
    result = report["cases"][0]["lookups"][0]
    assert report["ok"] is False
    assert result["failure"] == "sequential_scan"
    assert result["client_rtt_fetch"]["samples"] == 1
    assert result["server_execution"]["samples"] == 1
    assert connection.plan_count == 1
    assert connection.preflight_count == 2


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1, True, "0.1"])
def test_invalid_timing_values_never_produce_an_acceptance_result(value):
    with pytest.raises(benchmark.BenchmarkError, match="invalid_latency_sample"):
        benchmark.latency_summary([value])
