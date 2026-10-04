#!/usr/bin/env python3
"""Apply a frozen SQLite baseline -> authoritative SQLite tail to unready PG.

This is NOT a Hub replay, a writer-freeze attestation, or finalization. The
baseline must be the exact, fully verified import recorded by the migration
ledger. Both SQLite inputs must be closed immutable backups with identical
schemas. All four source tables, including updates/deletes at old IDs and every
checkpoint field, participate.

Only changed rows cross the network, using streaming COPY into temporary
before/after tables. Each affected target row must equal its entire baseline
row or its entire desired row (including absence). Unknown divergence aborts.
One transaction locks, applies, reads back every final row, and saves a receipt;
an interrupted attempt rolls back, and retrying the same inputs is idempotent.
There are no per-row SQL round trips and no unbounded Python row collections.
Temp tables may require disk proportional to changed payload size. A retry
restarts staging rather than resuming within the single atomic transaction.

The baseline migration ledger stays intact for the separate --finalize
--tail-source operation. Readiness stays false; sequence advancement and
derived-state reconstruction remain the existing finalizer's responsibility.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.migrate_source_postgres import (
    KEYS,
    TABLES,
    TRIGGERS,
    Migration,
    MigrationError,
    Snapshot,
    quoted,
)
from service.postgres import encode_pg_text
from service.server import utc_now

TAIL_LEDGER = "funes_source_migration_tail"


def check_snapshots(baseline: Snapshot, desired: Snapshot) -> None:
    if baseline.schema != desired.schema:
        raise MigrationError("tail source schema differs from the pinned baseline")
    baseline.assert_unchanged()
    desired.assert_unchanged()


def _key(table: str, value):
    if table == "translation_cache":
        if not isinstance(value, str):
            raise MigrationError(f"invalid source primary key: {table}")
        return encode_pg_text(value).encode("utf-8")
    if type(value) is not int:
        raise MigrationError(f"invalid source primary key: {table}")
    return value


def iter_changes(baseline: Snapshot, desired: Snapshot, table: str, *, progress=None):
    """Merge two PK-ordered cursors, retaining at most two source rows.

    Translation-cache order is the importer's encoded UTF-8 byte order, not
    locale or original Unicode order. All other keys are integer IDs.
    """
    position = baseline.columns[table].index(KEYS[table])
    old_rows, new_rows = baseline.rows(table), desired.rows(table)
    previous = {"old": None, "new": None}

    def advance(rows, side):
        row = next(rows, None)
        if row is None:
            return None, None
        key = _key(table, row[position])
        if previous[side] is not None and key <= previous[side]:
            raise MigrationError(f"source primary keys are not unique and ordered: {table}")
        previous[side] = key
        return row, key

    scanned = changed = 0
    try:
        old, old_key = advance(old_rows, "old")
        new, new_key = advance(new_rows, "new")
        while old is not None or new is not None:
            scanned += 1
            if new is None or (old is not None and old_key < new_key):
                changed += 1
                yield old, None
                old, old_key = advance(old_rows, "old")
            elif old is None or new_key < old_key:
                changed += 1
                yield None, new
                new, new_key = advance(new_rows, "new")
            else:
                if old != new:
                    changed += 1
                    yield old, new
                old, old_key = advance(old_rows, "old")
                new, new_key = advance(new_rows, "new")
            if progress and scanned % 50000 == 0:
                progress({"phase": "tail_staging", "table": table,
                          "scanned": scanned, "changed": changed})
    finally:
        old_rows.close()
        new_rows.close()


def _count_change(counts, old, new):
    counts["inserted" if old is None else "deleted" if new is None else "updated"] += 1


def summarize_tail(baseline: Snapshot, desired: Snapshot, *, progress=None):
    """Offline dry-run; never instantiate Migration or open PostgreSQL."""
    check_snapshots(baseline, desired)
    changes = {}
    for table in TABLES:
        counts = {"inserted": 0, "updated": 0, "deleted": 0}
        for old, new in iter_changes(baseline, desired, table, progress=progress):
            _count_change(counts, old, new)
        changes[table] = counts
    baseline.assert_unchanged(rehash=True)
    desired.assert_unchanged(rehash=True)
    return {"phase": "tail_dry_run", "writes": False,
            "baseline": baseline.identity, "source": desired.identity,
            "changes": changes, "final_write_boundary": False}


def _stored_row(snapshot: Snapshot, table: str, row):
    columns = snapshot.columns[table]
    if row is None:
        return [None] * len(columns)
    values = list(row)
    if table == "memories":
        raw, shadow = columns.index("raw_text"), columns.index("retrieval_text")
        if values[raw] == values[shadow]:
            values[shadow] = None
    for index, value in enumerate(values):
        if isinstance(value, str):
            values[index] = encode_pg_text(value)
        elif value is not None and type(value) is not int:
            raise MigrationError(f"unsupported SQLite value type: {table}")
    return values


def _stage(migration: Migration, desired: Snapshot, table: str, progress):
    source, pg = migration.source, migration.pg
    stage = "funes_tail_" + table
    key = KEYS[table]
    columns = source.columns[table]
    # One COPY stream holds both images; two concurrent COPYs on one connection
    # are not legal. Nullable images represent rows absent on either side.
    fields = ['"old_present" BOOLEAN NOT NULL', '"new_present" BOOLEAN NOT NULL',
              '"row_key" ' + ("TEXT COLLATE \"C\"" if table == "translation_cache" else "BIGINT")
              + " PRIMARY KEY"]
    for prefix in ("old_", "new_"):
        fields.extend(quoted(prefix + info[1]) + " "
                      + ("TEXT COLLATE \"C\"" if info[2].upper() == "TEXT" else "BIGINT")
                      for info in source.schema[table])
    pg.execute(f"CREATE TEMP TABLE {quoted(stage)} ({','.join(fields)}) ON COMMIT DROP")
    qualified = "pg_temp." + quoted(stage)
    counts = {"inserted": 0, "updated": 0, "deleted": 0}
    if progress:
        progress({"phase": "tail_staging", "table": table, "state": "started"})
    with pg.cursor() as cursor, cursor.copy(f"COPY {qualified} FROM STDIN") as copier:
        for old, new in iter_changes(source, desired, table, progress=progress):
            old_values, new_values = (_stored_row(source, table, old),
                                      _stored_row(desired, table, new))
            values = new_values if new is not None else old_values
            copier.write_row([old is not None, new is not None, values[columns.index(key)],
                              *old_values, *new_values])
            _count_change(counts, old, new)
    pg.execute(f"ANALYZE {qualified}")
    if progress:
        progress({"phase": "tail_staging", "table": table, "state": "staged", **counts})
    return qualified, counts


def _matches(snapshot: Snapshot, table: str, prefix: str) -> str:
    """Compare complete logical rows with byte-exact, null-safe equality."""
    fields = []
    types = {info[1]: info[2].upper() for info in snapshot.schema[table]}
    for name in snapshot.columns[table]:
        target, staged = "t." + quoted(name), "s." + quoted(prefix + name)
        if table == "memories" and name == "retrieval_text":
            target = f"COALESCE({target},t.raw_text)"
            staged = f"COALESCE({staged},s.{quoted(prefix + 'raw_text')})"
        if types[name] == "TEXT":
            target, staged = target + ' COLLATE "C"', staged + ' COLLATE "C"'
        fields.append(f"({target} IS NOT DISTINCT FROM {staged})")
    return " AND ".join(fields)


def apply_tail(migration: Migration, desired: Snapshot, *, progress=None, before_commit=None):
    """All-source-table atomic application; never makes the runtime ready."""
    source, pg = migration.source, migration.pg
    check_snapshots(source, desired)
    changes, applied, stages = {}, {}, {}
    with pg.transaction():
        pg.execute("SET LOCAL lock_timeout='30s'")
        pg.execute("SET LOCAL work_mem='8MB'")
        pg.execute("SET LOCAL maintenance_work_mem='16MB'")
        migration.load()
        migration._lock()
        # Re-read after acquiring table locks: an uncoordinated writer cannot
        # race the phase/identity checks with staging and final verification.
        migration.load()
        if (migration.phase != "verified" or not isinstance(migration.verification, dict)
                or migration.verification.get("source") != source.identity
                or not all(state["complete"] for state in migration.progress.values())):
            raise MigrationError("baseline import must be fully verified before applying a tail")
        for table in TABLES:
            stages[table], changes[table] = _stage(migration, desired, table, progress)

        # Preflight every changed table before mutating any source data.
        for table in TABLES:
            key, stage = quoted(KEYS[table]), stages[table]
            acceptable = []
            for prefix in ("old_", "new_"):
                present = "s." + quoted(prefix + "present")
                acceptable.append(f"(NOT {present} AND t.{key} IS NULL) OR "
                                  f"({present} AND t.{key} IS NOT NULL AND "
                                  f"({_matches(source, table, prefix)}))")
            conflict = pg.execute(
                f"SELECT EXISTS(SELECT 1 FROM {stage} s LEFT JOIN {quoted(table)} t "
                f"ON t.{key}=s.row_key WHERE NOT ({' OR '.join(acceptable)}))"
            ).fetchone()[0]
            if conflict:
                raise MigrationError(f"target row diverged from baseline and desired tail: {table}")

        trigger_states = pg.execute(
            "SELECT tgname,tgenabled FROM pg_trigger WHERE tgrelid='memories'::regclass "
            "AND NOT tgisinternal").fetchall()
        for trigger in TRIGGERS:
            pg.execute(f"ALTER TABLE memories DISABLE TRIGGER {quoted(trigger)}")
        for table in TABLES:
            key, stage = quoted(KEYS[table]), stages[table]
            columns = source.columns[table]
            # Delete before insert supports unique source_identity transfers and
            # swaps between IDs. Already-desired rows are neither touched nor
            # reinserted. Triggers stay disabled to preserve opaque checkpoints.
            removed = pg.execute(
                f"DELETE FROM {quoted(table)} t USING {stage} s WHERE t.{key}=s.row_key "
                f"AND NOT (s.new_present AND ({_matches(source, table, 'new_')}))"
            ).rowcount
            written = pg.execute(
                f"INSERT INTO {quoted(table)} ({','.join(map(quoted, columns))}) "
                f"SELECT {','.join('s.' + quoted('new_' + name) for name in columns)} "
                f"FROM {stage} s LEFT JOIN {quoted(table)} t ON t.{key}=s.row_key "
                f"WHERE s.new_present AND t.{key} IS NULL"
            ).rowcount
            applied[table] = {"removed": removed, "written": written}
            pg.execute(f"DROP TABLE {stage}")
            if progress:
                progress({"phase": "tail_applying", "table": table, **applied[table]})
        for trigger, original in trigger_states:
            action = "ENABLE" if original == "O" else "DISABLE"
            pg.execute(f"ALTER TABLE memories {action} TRIGGER {quoted(trigger)}")

        # This also catches unknown extra rows or corruption at unchanged IDs.
        # Full decoded typed digests and the desired source rehash are the same
        # routines used by the baseline importer, not a weaker tail checksum.
        verification = migration._verify_tables(desired)
        source.assert_unchanged(rehash=True)
        pg.execute("UPDATE funes_schema_state SET document_count=%s WHERE id=1",
                   (desired.counts["memories"],))
        pg.execute(f"""CREATE TABLE IF NOT EXISTS {TAIL_LEDGER} (
            id INTEGER PRIMARY KEY CHECK(id=1), version INTEGER NOT NULL,
            baseline_identity JSONB NOT NULL, source_identity JSONB NOT NULL,
            changes JSONB NOT NULL, verification JSONB NOT NULL, applied_at TEXT NOT NULL
        )""")
        pg.execute(f"""INSERT INTO {TAIL_LEDGER}
            (id,version,baseline_identity,source_identity,changes,verification,applied_at)
            VALUES(1,1,%s::jsonb,%s::jsonb,%s::jsonb,%s::jsonb,%s)
            ON CONFLICT(id) DO UPDATE SET version=EXCLUDED.version,
                baseline_identity=EXCLUDED.baseline_identity,source_identity=EXCLUDED.source_identity,
                changes=EXCLUDED.changes,verification=EXCLUDED.verification,applied_at=EXCLUDED.applied_at
        """, (json.dumps(source.identity), json.dumps(desired.identity), json.dumps(changes),
              json.dumps(verification), utc_now()))
        if before_commit:
            before_commit()
        source.assert_unchanged()
        desired.assert_unchanged()
        # No setval(): sequence changes are not transactionally rolled back.
        # The existing explicit finalizer advances it before enabling runtime.
    return {"phase": "tail_applied", "ready": False, "final_write_boundary": False,
            "baseline": source.identity, "changes": changes, "applied": applied, **verification}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="Original verified immutable baseline SQLite backup")
    parser.add_argument("--tail-source", required=True, help="Frozen authoritative final SQLite backup")
    parser.add_argument("--expected-sha256", help="Optional externally recorded baseline SHA-256")
    parser.add_argument("--tail-expected-sha256", help="Optional externally recorded final SHA-256")
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--dry-run", action="store_true", help="Compare local snapshots only; no PostgreSQL connection")
    action.add_argument("--apply", action="store_true", help="Apply the whole tail atomically to still-unready PostgreSQL")
    parser.add_argument("--defer-ready", action="store_true", help="Required for --apply; never finalizes")
    args = parser.parse_args(argv)
    if args.apply and not args.defer_ready:
        parser.error("--apply requires --defer-ready; finalization remains a separate operation")
    os.umask(0o077)

    def progress(state):
        print(json.dumps(state, sort_keys=True), flush=True)

    try:
        progress({"phase": "hashing_baseline_and_tail"})
        with Snapshot(args.source, args.expected_sha256) as source, \
                Snapshot(args.tail_source, args.tail_expected_sha256) as final:
            check_snapshots(source, final)
            if args.dry_run:
                result = summarize_tail(source, final, progress=progress)
            else:
                with Migration(source, event_callback=progress) as migration:
                    result = apply_tail(migration, final, progress=progress)
        progress(result)
        return 0
    except KeyboardInterrupt:
        print(json.dumps({"error": "tail application interrupted; rerun the same immutable inputs; "
                                   "inspect the tail receipt if commit acknowledgement was interrupted"}), file=sys.stderr)
        return 130
    except MigrationError as error:
        print(json.dumps({"error": str(error)}), file=sys.stderr)
        return 1
    except Exception:  # noqa: BLE001 -- never expose driver credentials/payloads
        # Driver errors can include credentials, raw payloads or COPY row data.
        print(json.dumps({"error": "tail application failed; inspect source/schema/network and the "
                                   "durable tail receipt before retrying; readiness not confirmed"}), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
