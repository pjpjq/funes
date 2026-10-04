#!/usr/bin/env python3
"""Read-only benchmark of PostgreSQL canonical source-identity hydration.

Input is a JSON array (or {"cases": [...]}) of objects with label, query and
identities. Obtain identities independently with native recall or safe sampling
of a frozen SQLite source. They must be decoded canonical source_identity values,
not native funes-doc references. Optional identity_source documents that origin.
The query is context ONLY: it is never sent to PostgreSQL as content search.
Cases are operator-approved report-safe inputs (max 20 cases, 100 identities each).
This measures successful canonical get, not numeric row-ID fallback after a miss.

Example input: [{"label": "technical_identifier", "query": "previous_response_id",
                 "identities": ["canonical-source-id"],
                 "identity_source": "native_recall"}]

Run with FUNES_POSTGRES_DSN in the environment, then:
  python scripts/benchmark_postgres_identifier_lookup.py --cases cases.json \
      --output benchmark.json

Uses the real SELECT * get/get_many SQL, not an identifier-only projection. It
does not create a Store, migrate, ANALYZE the table, modify data/schema, or force
planner choices. A nonexecuting EXPLAIN first rejects unsafe plans. Each accepted
EXPLAIN ANALYZE executes the SELECT, followed by a separate timed SELECT and full
fetch. This is not a cold-cache, native retrieval, or HTTP end-to-end benchmark.
Source payloads and unsanitized plans never enter reports.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


CANONICAL_INDEX = "memories_source_identity_key"
BUFFER_KEYS = {
    "Shared Hit Blocks": "shared_hit_blocks",
    "Shared Read Blocks": "shared_read_blocks",
    "Shared Dirtied Blocks": "shared_dirtied_blocks",
    "Shared Written Blocks": "shared_written_blocks",
    "Local Hit Blocks": "local_hit_blocks",
    "Local Read Blocks": "local_read_blocks",
    "Local Dirtied Blocks": "local_dirtied_blocks",
    "Local Written Blocks": "local_written_blocks",
    "Temp Read Blocks": "temp_read_blocks",
    "Temp Written Blocks": "temp_written_blocks",
}
REMOVED_KEYS = {
    "Rows Removed by Filter": "filter",
    "Rows Removed by Index Recheck": "index_recheck",
    "Rows Removed by Join Filter": "join_filter",
}
NODE_TYPES = frozenset({
    "Seq Scan", "Index Scan", "Index Only Scan", "Bitmap Index Scan",
    "Bitmap Heap Scan", "BitmapAnd", "BitmapOr", "Limit", "Sort",
    "Incremental Sort", "Gather", "Gather Merge", "Result", "Append",
    "Merge Append", "Materialize", "Memoize", "Nested Loop", "Hash Join",
    "Merge Join", "Hash", "Aggregate", "Group", "Unique", "Subquery Scan",
    "CTE Scan", "Function Scan", "Table Function Scan", "Values Scan",
    "WorkTable Scan", "Recursive Union", "LockRows", "ModifyTable", "SetOp",
    "WindowAgg", "ProjectSet", "Tid Scan", "Tid Range Scan", "Sample Scan",
    "Foreign Scan", "Custom Scan",
})
INDEX_NODES = frozenset({"Index Scan", "Index Only Scan", "Bitmap Index Scan"})


class BenchmarkError(RuntimeError):
    """Internal errors use fixed codes; never embed DB exceptions or payloads."""

    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


def number(value, *, code="invalid_plan") -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BenchmarkError(code)
    if not math.isfinite(value) or value < 0:
        raise BenchmarkError(code)
    return float(value)


def latency_summary(values: list[float]) -> dict:
    """Nearest-rank percentiles (ceil(p*n)); never synthesize empty samples."""
    if not values:
        raise BenchmarkError("empty_latency_samples")
    ordered = sorted(number(value, code="invalid_latency_sample") for value in values)
    return {
        "samples": len(ordered),
        "p50_ms": ordered[math.ceil(0.50 * len(ordered)) - 1],
        "p95_ms": ordered[math.ceil(0.95 * len(ordered)) - 1],
        "max_ms": ordered[-1],
    }


def summarize_plan(document, *, analyzed: bool = True) -> dict:
    """Allowlist plan structure/numbers; never copy Filter/Index Cond/Output."""
    if (not isinstance(document, list) or len(document) != 1
            or not isinstance(document[0], dict)):
        raise BenchmarkError("invalid_plan")
    top = document[0]
    nodes = []

    def visit(node, path):
        if not isinstance(node, dict):
            raise BenchmarkError("invalid_plan")
        node_type = node.get("Node Type")
        if not isinstance(node_type, str) or node_type not in NODE_TYPES:
            raise BenchmarkError("invalid_plan")
        # Bounded traversal also fails closed on malformed/cyclic fake input.
        if len(nodes) >= 100 or len(path) > 64:
            raise BenchmarkError("invalid_plan")
        entry = {
            "path": path,
            "node_type": node_type,
        }
        if analyzed:
            entry.update({
                "actual_rows": number(node.get("Actual Rows")),
                "actual_loops": number(node.get("Actual Loops")),
                "rows_removed_per_loop": {
                    key: number(node.get(source, 0)) for source, key in REMOVED_KEYS.items()
                },
                "buffers": {
                    key: number(node.get(source, 0)) for source, key in BUFFER_KEYS.items()
                },
            })
        elif "Plan Rows" in node:
            entry["estimated_rows"] = number(node["Plan Rows"])
        index = node.get("Index Name")
        if index is not None:
            if (not isinstance(index, str)
                    or not re.fullmatch(r"[A-Za-z_][A-Za-z_0-9.]{0,127}", index)):
                raise BenchmarkError("invalid_plan")
            entry["index_name"] = index
        nodes.append(entry)
        children = node.get("Plans", [])
        if not isinstance(children, list):
            raise BenchmarkError("invalid_plan")
        for position, child in enumerate(children):
            visit(child, f"{path}.{position}")

    visit(top.get("Plan"), "0")
    summary = {
        "analyzed": analyzed,
        "node_types": list(dict.fromkeys(node["node_type"] for node in nodes)),
        "index_names": list(dict.fromkeys(
            node["index_name"] for node in nodes if "index_name" in node
        )),
        "canonical_index_planned": any(
            node["node_type"] in INDEX_NODES
            and node.get("index_name") == CANONICAL_INDEX
            for node in nodes
        ),
        "sequential_scan": any(node["node_type"] == "Seq Scan" for node in nodes),
        "nodes": nodes,
    }
    if not analyzed:
        return summary
    summary.update({
        "server_execution_ms": number(top.get("Execution Time")),
        "server_planning_ms": number(top.get("Planning Time")),
        "canonical_index_used": any(
            node["node_type"] in INDEX_NODES
            and node.get("index_name") == CANONICAL_INDEX
            and node["actual_loops"] > 0 for node in nodes
        ),
        "rows_removed_total": sum(
            sum(node["rows_removed_per_loop"].values()) * node["actual_loops"]
            for node in nodes
        ),
        # EXPLAIN buffer counters include descendants; do not add them again.
        "buffers": nodes[0]["buffers"],
        "buffer_hits": nodes[0]["buffers"]["shared_hit_blocks"],
        "buffer_reads": nodes[0]["buffers"]["shared_read_blocks"],
        "rows_removed": [
            {"path": node["path"], "node_type": node["node_type"],
             **node["rows_removed_per_loop"]}
            for node in nodes
            if any(node["rows_removed_per_loop"].values())
        ],
    })
    return summary


MAX_CASES = 20
MAX_IDENTITIES_PER_CASE = 100
MAX_LABEL_LENGTH = 128
MAX_QUERY_LENGTH = 4_096
MAX_IDENTITY_LENGTH = 1_024
MAX_REPEATS = 20
MAX_BATCH_LIMIT = 100
SERVER_P95_TARGET_MS = 100.0
MAX_CASE_FILE_BYTES = 1_048_576
DEFAULT_BATCH_LIMITS = (10, 100)
GET_SQL = "SELECT * FROM memories WHERE source_identity=%s ORDER BY id LIMIT 1"
GET_MANY_PREFIX = "SELECT * FROM memories WHERE source_identity IN ("
SESSION_KEYS = ("transaction_read_only", "enable_seqscan", "enable_indexscan",
                "enable_bitmapscan", "enable_indexonlyscan")
SESSION_CHECK_SQL = "SELECT " + ",".join(f"current_setting('{key}')" for key in SESSION_KEYS)
ERROR_CODES = frozenset({
    "invalid_plan", "empty_latency_samples", "invalid_latency_sample", "invalid_cases",
    "invalid_case", "invalid_case_label", "invalid_case_query", "invalid_case_identities",
    "canonical_identities_required", "invalid_identity_source", "invalid_repeats",
    "invalid_batch_limits", "database_cursor_failed", "database_explain_failed",
    "database_query_failed", "database_connect_failed", "psycopg_unavailable",
    "missing_dsn", "invalid_session_settings", "read_only_required",
    "planner_settings_not_default", "sequential_scan", "canonical_index_not_used",
    "explain_row_count_mismatch", "hydration_identity_mismatch", "missing_identity_column",
    "server_p95_target_missed", "invalid_arguments", "cases_read_failed",
    "cases_file_too_large", "cases_invalid_json", "output_conflicts_with_cases",
    "interrupted",
})


def _safe_text(value: Any, *, max_length: int, code: str) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > max_length or "\x00" in value:
        raise BenchmarkError(code)
    return value


def parse_cases(document: Any) -> list[dict[str, Any]]:
    """Validate independent discovery input and deduplicate identities in order."""
    if isinstance(document, dict):
        document = document.get("cases")
    if not isinstance(document, list) or not document or len(document) > MAX_CASES:
        raise BenchmarkError("invalid_cases")
    parsed = []
    for case in document:
        if not isinstance(case, dict):
            raise BenchmarkError("invalid_case")
        label = _safe_text(case.get("label"), max_length=MAX_LABEL_LENGTH,
                           code="invalid_case_label")
        query = _safe_text(case.get("query"), max_length=MAX_QUERY_LENGTH,
                           code="invalid_case_query")
        identities = case.get("identities")
        if (not isinstance(identities, list) or not identities
                or len(identities) > MAX_IDENTITIES_PER_CASE):
            raise BenchmarkError("invalid_case_identities")
        clean = []
        seen = set()
        for identity in identities:
            if not isinstance(identity, str) or not identity.strip():
                raise BenchmarkError("invalid_case_identities")
            if len(identity) > MAX_IDENTITY_LENGTH or "\x00" in identity:
                raise BenchmarkError("invalid_case_identities")
            if identity.startswith("funes-doc:"):
                raise BenchmarkError("canonical_identities_required")
            if identity not in seen:
                seen.add(identity)
                clean.append(identity)
        identity_source = case.get("identity_source", "external_case_input")
        identity_source = _safe_text(identity_source, max_length=64,
                                     code="invalid_identity_source")
        parsed.append({"label": label, "query": query, "identities": clean,
                       "identity_source": identity_source})
    return parsed


def validate_config(repeats: int, batch_limits: list[int] | tuple[int, ...]) -> tuple[int, list[int]]:
    if isinstance(repeats, bool) or not isinstance(repeats, int) or not 1 <= repeats <= MAX_REPEATS:
        raise BenchmarkError("invalid_repeats")
    if (not isinstance(batch_limits, (list, tuple)) or not batch_limits
            or len(batch_limits) > 5):
        raise BenchmarkError("invalid_batch_limits")
    normalized = []
    for limit in batch_limits:
        if (isinstance(limit, bool) or not isinstance(limit, int)
                or not 1 <= limit <= MAX_BATCH_LIMIT or limit in normalized):
            raise BenchmarkError("invalid_batch_limits")
        normalized.append(limit)
    return repeats, normalized


def build_lookups(case: dict[str, Any], batch_limits: list[int] | tuple[int, ...]) -> list[dict[str, Any]]:
    """Build the exact SQL shapes used by PostgresStore.get/get_many.

    ``query`` is deliberately absent: it describes the independent native
    retrieval that discovered identities and is never used as PostgreSQL text
    search. ``requested_limit`` controls candidate overhydration only; the SQL
    intentionally has no LIMIT because get_many() does not have one.
    """
    identities = list(case["identities"])
    lookups = [{
        "kind": "get",
        "sql": GET_SQL,
        "identities": identities[:1],
        "requested_limit": 1,
        "requested_limit_filled": True,
    }]
    for limit in batch_limits:
        selected = identities[:limit]
        placeholders = ",".join("%s" for _ in selected)
        lookups.append({
            "kind": "get_many",
            "sql": GET_MANY_PREFIX + placeholders + ")",
            "identities": selected,
            "requested_limit": limit,
            "requested_limit_filled": len(selected) >= limit,
        })
    return lookups


def _plan_from_row(row: Any) -> Any:
    if isinstance(row, dict):
        if "QUERY PLAN" in row:
            row = row["QUERY PLAN"]
        elif len(row) == 1:
            row = next(iter(row.values()))
    elif isinstance(row, (tuple, list)) and len(row) == 1:
        row = row[0]
    if isinstance(row, str):
        try:
            row = json.loads(row)
        except (TypeError, ValueError) as error:
            raise BenchmarkError("invalid_plan") from error
    return row


def _cursor(conn):
    try:
        return conn.cursor()
    except Exception as error:
        raise BenchmarkError("database_cursor_failed") from error


def _execute_plan(conn, sql: str, identities: list[str], *, analyze: bool = True) -> dict:
    cursor = _cursor(conn)
    try:
        prefix = "EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) " if analyze else "EXPLAIN (FORMAT JSON) "
        cursor.execute(prefix + sql, tuple(identities))
        row = cursor.fetchone()
    except Exception as error:
        raise BenchmarkError("database_explain_failed") from error
    finally:
        try:
            cursor.close()
        except Exception:
            pass
    return summarize_plan(_plan_from_row(row), analyzed=analyze)


def _timed_fetch(conn, sql: str, identities: list[str], clock: Callable[[], float]) -> tuple[float, int, bool]:
    cursor = _cursor(conn)
    started = clock()
    try:
        cursor.execute(sql, tuple(identities))
        rows = cursor.fetchall()
        elapsed = number((clock() - started) * 1000.0, code="invalid_latency_sample")
        count = len(rows)
        columns = [column.name for column in cursor.description or ()]
        if "source_identity" not in columns:
            raise BenchmarkError("missing_identity_column")
        position = columns.index("source_identity")
        # Only compare identities; do not stringify/hash/serialize whole rows.
        actual = [row[position] for row in rows]
        matched = count == len(identities) and set(actual) == set(identities)
    except BenchmarkError:
        raise
    except Exception as error:
        raise BenchmarkError("database_query_failed") from error
    finally:
        try:
            cursor.close()
        except Exception:
            pass
    return elapsed, count, matched


def _check_session(conn) -> dict[str, str]:
    cursor = _cursor(conn)
    try:
        cursor.execute(SESSION_CHECK_SQL)
        row = cursor.fetchone()
        if not row or len(row) != len(SESSION_KEYS) or any(value not in ("on", "off") for value in row):
            raise BenchmarkError("invalid_session_settings")
        settings = dict(zip(SESSION_KEYS, row))
        if settings["transaction_read_only"] != "on":
            raise BenchmarkError("read_only_required")
        if any(settings[key] != "on" for key in SESSION_KEYS[1:]):
            raise BenchmarkError("planner_settings_not_default")
        return settings
    finally:
        cursor.close()


def _connect_read_only(dsn: str):
    try:
        import psycopg
    except ImportError as error:
        raise BenchmarkError("psycopg_unavailable") from error
    conn = None
    try:
        conn = psycopg.connect(
            dsn,
            autocommit=True,
            connect_timeout=10,
            application_name="funes-identifier-benchmark",
            # Both measured SQL and EXPLAIN remain unprepared, not different
            # prepared/generic-plan lifecycles. Report this SQL-only boundary.
            prepare_threshold=None,
        )
        # Let libpq finish multi-host target_session_attrs=read-write selection
        # before making this session read-only; startup read-only rejects it.
        # Apply session settings with SQL: managed proxies may reject startup
        # options. Autocommit keeps these SETs outside a transaction.
        cursor = conn.cursor()
        try:
            cursor.execute("SET statement_timeout='10s'")
            cursor.execute("SET lock_timeout='1s'")
            cursor.execute("SET default_transaction_read_only=on")
        finally:
            cursor.close()
        return conn
    except BaseException as error:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass
        if not isinstance(error, Exception):
            raise
        # Do not include the exception: psycopg can echo the complete DSN.
        raise BenchmarkError("database_connect_failed") from error


def _safe_failure(error: BaseException) -> str:
    if isinstance(error, KeyboardInterrupt):
        return "interrupted"
    if isinstance(error, BenchmarkError) and error.code in ERROR_CODES:
        return error.code
    return "benchmark_failed"


def run_benchmark(
    cases: list[dict[str, Any]],
    *,
    dsn: str,
    repeats: int = MAX_REPEATS,
    batch_limits: list[int] | tuple[int, ...] = DEFAULT_BATCH_LIMITS,
    connect_factory: Callable[[str], Any] | None = None,
    clock: Callable[[], float] = time.perf_counter,
) -> dict[str, Any]:
    """Preflight without execution, then repeat ANALYZE and full fetch; fail fast."""
    if not isinstance(dsn, str) or not dsn:
        raise BenchmarkError("missing_dsn")
    repeats, batch_limits = validate_config(repeats, batch_limits)
    cases = parse_cases(cases)
    connect = connect_factory or _connect_read_only
    conn = connect(dsn)
    report: dict[str, Any] = {
        "ok": True,
        "status": "ok",
        "benchmark": "postgres_identifier_hydration",
        "mode": "read_only",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "postgres_content_search": False,
        "identity_discovery_is_external": True,
        "repeats": repeats,
        "batch_limits": batch_limits,
        "server_p95_target_ms": SERVER_P95_TARGET_MS,
        "server_p95_target_comparison": "strictly_less_than",
        "percentile_method": "nearest_rank",
        "planned_lookups": len(cases) * (len(batch_limits) + 1),
        "completed_lookups": 0,
        "complete": False,
        "auto_prepare": False,
        "statement_timeout_ms": 10000,
        "lock_timeout_ms": 1000,
        "connect_timeout_seconds": 10,
        "native_content_search_measured": False,
        "identity_provenance_verified": False,
        "server_execution_is_from_explain": True,
        "client_latency_includes_fetch": True,
        "measurement_notes": [
            "query is input context only; no PostgreSQL content or search_identifiers lookup",
            "each round: nonexecuting EXPLAIN preflight, ANALYZE, full SELECT; uncontrolled warm cache",
            "server execution excludes planning and client payload transfer; client metric is execute+fetch",
            "separate executions: do not subtract server time from client time to infer network RTT",
            "SQL-only unprepared baseline; excludes connection setup, Store conversion/locks and HTTP/native latency",
        ],
        "cases": [],
        "failures": [],
    }
    try:
        report["session_settings"] = _check_session(conn)
        for case in cases:
            case_report = {
                "label": case["label"],
                "query": case["query"],
                "identity_source": case["identity_source"],
                "identity_count": len(case["identities"]),
                "lookups": [],
            }
            report["cases"].append(case_report)
            for lookup in build_lookups(case, batch_limits):
                plans = []
                preflight_plans = []
                client_samples = []
                returned_rows = []
                lookup_failure = None
                for _ in range(repeats):
                    try:
                        preflight = _execute_plan(conn, lookup["sql"], lookup["identities"], analyze=False)
                        preflight_plans.append(preflight)
                        if preflight["sequential_scan"]:
                            lookup_failure = "sequential_scan"
                        elif not preflight["canonical_index_planned"]:
                            lookup_failure = "canonical_index_not_used"
                        if lookup_failure:
                            break
                        plan = _execute_plan(conn, lookup["sql"], lookup["identities"])
                        plans.append(plan)
                        if plan["sequential_scan"]:
                            lookup_failure = "sequential_scan"
                        elif not plan["canonical_index_used"]:
                            lookup_failure = "canonical_index_not_used"
                        elif plan["nodes"][0]["actual_rows"] != len(lookup["identities"]):
                            lookup_failure = "explain_row_count_mismatch"
                        if lookup_failure:
                            break
                        elapsed, count, matched = _timed_fetch(
                            conn, lookup["sql"], lookup["identities"], clock,
                        )
                        client_samples.append(elapsed)
                        returned_rows.append(count)
                        if not matched:
                            lookup_failure = "hydration_identity_mismatch"
                            break
                    except Exception as error:
                        lookup_failure = _safe_failure(error)
                        break
                server_summary = latency_summary([plan["server_execution_ms"] for plan in plans]) if plans else None
                if (not lookup_failure and server_summary
                        and server_summary["p95_ms"] >= SERVER_P95_TARGET_MS):
                    lookup_failure = "server_p95_target_missed"
                result = {
                    "kind": lookup["kind"],
                    "sql": lookup["sql"],
                    "parameters": {"source_identity": lookup["identities"]},
                    "identity_count": len(lookup["identities"]),
                    "requested_limit": lookup["requested_limit"],
                    "requested_limit_filled": lookup["requested_limit_filled"],
                    "plan": plans[0] if plans else None,
                    "plans": plans,
                    "preflight_plans": preflight_plans,
                    "server_execution": server_summary,
                    "server_planning": latency_summary([plan["server_planning_ms"] for plan in plans]) if plans else None,
                    "client_rtt_fetch": latency_summary(client_samples) if client_samples else None,
                    "returned_rows": sorted(set(returned_rows)),
                    "server_p95_target_met": bool(
                        server_summary and len(plans) == repeats
                        and server_summary["p95_ms"] < SERVER_P95_TARGET_MS
                    ),
                    "failure": lookup_failure,
                }
                case_report["lookups"].append(result)
                if lookup_failure:
                    report["ok"] = False
                    report["status"] = "failed"
                    report["failures"].append({
                        "label": case["label"], "kind": lookup["kind"], "code": lookup_failure,
                    })
                    return report
                report["completed_lookups"] += 1
        report["complete"] = True
    finally:
        try:
            conn.close()
        except Exception:
            pass
    return report


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path = path.absolute()
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, sort_keys=True, indent=2)
            stream.write("\n")
        os.replace(temp_name, path)
    finally:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass


class SafeArgumentParser(argparse.ArgumentParser):
    def error(self, message):
        # argparse's ordinary diagnostics repeat invalid argument values.
        raise BenchmarkError("invalid_arguments")


def main(argv=None) -> int:
    parser = SafeArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cases", required=True, help="JSON file with independent label/query/identities cases")
    parser.add_argument("--output", required=True, help="JSON report path")
    parser.add_argument("--repeats", type=int, default=MAX_REPEATS,
                        help="EXPLAIN and full-fetch pairs per lookup (1-20, default 20)")
    parser.add_argument("--batch-limits", default=",".join(map(str, DEFAULT_BATCH_LIMITS)),
                        help="comma-separated independent candidate budgets (default 10,100; max 100)")
    output = None
    try:
        args = parser.parse_args(argv)
        output = Path(args.output)
        cases_path = Path(args.cases)
        if output.resolve() == cases_path.resolve():
            output = None
            raise BenchmarkError("output_conflicts_with_cases")
        try:
            limits = [int(value) for value in args.batch_limits.split(",")]
        except ValueError:
            raise BenchmarkError("invalid_batch_limits") from None
        validate_config(args.repeats, limits)
        try:
            with cases_path.open("rb") as stream:
                content = stream.read(MAX_CASE_FILE_BYTES + 1)
        except OSError:
            raise BenchmarkError("cases_read_failed") from None
        if len(content) > MAX_CASE_FILE_BYTES:
            raise BenchmarkError("cases_file_too_large")
        try:
            cases_document = json.loads(content)
        except (ValueError, UnicodeError):
            raise BenchmarkError("cases_invalid_json") from None
        cases = parse_cases(cases_document)
        dsn = os.environ.get("FUNES_POSTGRES_DSN")
        if not dsn:
            raise BenchmarkError("missing_dsn")
        report = run_benchmark(cases, dsn=dsn, repeats=args.repeats, batch_limits=limits)
        report["case_file_sha256"] = hashlib.sha256(content).hexdigest()
    except (Exception, KeyboardInterrupt) as error:
        report = {
            "ok": False,
            "status": "failed",
            "benchmark": "postgres_identifier_hydration",
            "error_code": _safe_failure(error),
            "dsn_source": "FUNES_POSTGRES_DSN",
        }
    if output is not None:
        try:
            _write_json(output, report)
        except Exception:
            print(json.dumps({"ok": False, "error_code": "output_write_failed"}), file=sys.stderr)
            return 2
    summary = {key: report[key] for key in (
        "ok", "status", "error_code", "completed_lookups", "planned_lookups", "failures",
    ) if key in report}
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True),
          file=sys.stdout if report["ok"] else sys.stderr)
    return 0 if report["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
