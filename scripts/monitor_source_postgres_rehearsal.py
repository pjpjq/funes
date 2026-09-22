#!/usr/bin/env python3
"""Read-only, bounded observations of an offline COPY rehearsal. No row payloads."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import time
from pathlib import Path


def section(connection, query, parameters=()):
    """An importer may hold ACCESS EXCLUSIVE: skip, never wait indefinitely."""
    try:
        with connection.cursor() as cursor:
            cursor.execute(query, parameters)
            names = [column.name for column in cursor.description]
            return [dict(zip(names, row)) for row in cursor.fetchall()]
    except Exception as error:
        # Database errors may include credentials or COPY source values.
        return {"unavailable": True, "sqlstate": getattr(error, "sqlstate", None)}


def snapshot(connection, replica=None):
    result = {
        "sampled_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "database": section(connection, "SELECT current_database() AS name, "
                            "pg_database_size(current_database()) AS bytes"),
        "checkpoint": section(connection, "SELECT phase, progress, updated_at "
                              "FROM funes_source_migration WHERE id=1"),
        "readiness": section(connection, "SELECT schema_version, migration_ready, document_count "
                             "FROM funes_schema_state WHERE id=1"),
        "copy_progress": section(connection, "SELECT pid,command,type,bytes_processed,bytes_total,tuples_processed "
                                 "FROM pg_stat_progress_copy WHERE datname=current_database()"),
        "migration_activity": section(connection, "SELECT pid,state,wait_event_type,wait_event "
                                      "FROM pg_stat_activity WHERE datname=current_database() "
                                      "AND application_name='funes-source-migration'"),
        "memories": section(connection, """SELECT
            pg_relation_size(c.oid) AS heap_main_bytes,
            pg_table_size(c.oid) AS table_including_toast_bytes,
            pg_total_relation_size(c.reltoastrelid) AS toast_total_bytes,
            pg_indexes_size(c.oid) AS btree_bytes,
            pg_total_relation_size(c.oid) AS total_bytes
            FROM pg_class c JOIN pg_namespace n ON c.relnamespace=n.oid
            WHERE n.nspname=current_schema() AND c.relname='memories'"""),
        "indexes": section(connection, """SELECT c.relname AS name, a.amname AS method,
            i.indisvalid AS valid, pg_relation_size(c.oid) AS bytes
            FROM pg_index i JOIN pg_class c ON c.oid=i.indexrelid
            JOIN pg_am a ON a.oid=c.relam WHERE i.indrelid=to_regclass('memories')
            ORDER BY c.relname"""),
        "statistics": section(connection, """SELECT n_live_tup AS estimated_rows,
            n_tup_ins AS cumulative_inserts, n_dead_tup AS estimated_dead_rows,
            last_analyze, last_autoanalyze FROM pg_stat_user_tables
            WHERE schemaname=current_schema() AND relname='memories'"""),
        "wal": section(connection, "SELECT wal_records,wal_fpi,wal_bytes,wal_buffers_full,stats_reset "
                       "FROM pg_stat_wal"),
        "primary_lsn": section(connection, "SELECT pg_current_wal_lsn()::text AS lsn"),
    }
    if replica is not None:
        result["replica"] = section(replica, "SELECT pg_is_in_recovery() AS recovering, "
                                    "pg_last_wal_replay_lsn()::text AS replay_lsn")
        primary, read = result["primary_lsn"], result["replica"]
        if isinstance(primary, list) and primary and isinstance(read, list) and read:
            lsn, replay = primary[0]["lsn"], read[0]["replay_lsn"]
            if lsn and replay:
                result["replica_lag"] = section(connection, "SELECT "
                    "GREATEST(pg_wal_lsn_diff(%s::pg_lsn,%s::pg_lsn),0) AS bytes", (lsn, replay))
                result["replica_lag_note"] = "primary sampled first; replica may advance; bytes are clamped at zero"
    return result


def connect(dsn):
    import psycopg
    connection = psycopg.connect(dsn, autocommit=True, connect_timeout=10,
                                application_name="funes-rehearsal-monitor",
                                options="-c statement_timeout=5000 -c lock_timeout=500")
    try:
        # Let libpq finish target_session_attrs selection before making this
        # monitoring session read-only. Startup read-only conflicts with a
        # caller's read-write target requirement, even on a healthy primary.
        with connection.cursor() as cursor:
            cursor.execute("SET default_transaction_read_only=on")
    except Exception:
        connection.close()
        raise
    return connection


def connection_unavailable(error):
    """Retain only diagnostic categories, never exception text or connection info."""
    return {"unavailable": True, "error_class": type(error).__name__,
            "sqlstate": getattr(error, "sqlstate", None)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Append sanitized JSONL observations (0600)")
    parser.add_argument("--interval", type=float, default=60)
    parser.add_argument("--samples", type=int, default=1)
    args = parser.parse_args(argv)
    if args.interval < 10 or not 1 <= args.samples <= 1440:
        parser.error("interval must be >=10 seconds and samples must be 1..1440")
    for index in range(args.samples):
        started = time.monotonic()
        try:
            from contextlib import ExitStack
            with ExitStack() as stack:
                try:
                    primary = stack.enter_context(connect(os.environ["FUNES_POSTGRES_DSN"]))
                except Exception as error:
                    result = {"sampled_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                              "error": "monitor_connection_unavailable",
                              "primary": connection_unavailable(error)}
                else:
                    replica_dsn = os.getenv("FUNES_POSTGRES_READ_DSN")
                    replica = None
                    replica_error = None
                    if replica_dsn:
                        try:
                            replica = stack.enter_context(connect(replica_dsn))
                        except Exception as error:
                            replica_error = connection_unavailable(error)
                    result = snapshot(primary, replica)
                    if replica_error is not None:
                        result["replica"] = replica_error
        except Exception as error:
            result = {"sampled_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                      "error": "monitor_connection_unavailable",
                      "diagnostic": connection_unavailable(error)}
        result["observation_seconds"] = round(time.monotonic() - started, 3)
        line = json.dumps(result, default=str, ensure_ascii=True) + "\n"
        if args.output:
            fd = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
            with os.fdopen(fd, "w") as stream:
                stream.write(line)
        else:
            print(line, end="", flush=True)
        if index + 1 < args.samples:
            time.sleep(max(0, args.interval - (time.monotonic() - started)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
