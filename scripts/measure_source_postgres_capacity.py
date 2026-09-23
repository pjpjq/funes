#!/usr/bin/env python3
"""Measure at most 10k frozen rows in an owned LOCAL disposable PG schema.

Never accepts a production DSN: only Unix-socket databases ending in _test.
No Hub calls, paid embeddings, full import, or edits of the frozen source.
The historical capacity digest is retained to prove identical sample selection.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import sqlite3
import statistics
import struct
import sys
import tempfile
import time
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
TABLES = ("memories", "translation_cache", "sync_state", "reindex_controls")
SUMMARY = {}


def progress(stage, **safe):
    print(json.dumps({"stage": stage, **safe}), flush=True)


class CheckFailed(RuntimeError):
    pass


def require(condition, code):
    if not condition:
        raise CheckFailed(code)


def signature(path):
    st = path.stat()
    return {'device': st.st_dev, 'inode': st.st_ino, 'size': st.st_size, 'mtime_ns': st.st_mtime_ns}


def q(name):
    return '"' + name.replace('"', '""') + '"'


def select_source_rows(db, table, columns):
    keys = {'memories': 'id', 'sync_state': 'id', 'reindex_controls': 'generation'}
    statement = 'SELECT ' + ','.join(q(c) for c in columns) + ' FROM ' + q(table)
    if table == 'translation_cache':
        from service.postgres import encode_pg_text
        rows = db.execute(statement).fetchall()
        key_index = columns.index('query')
        yield from sorted(rows, key=lambda r: encode_pg_text(r[key_index]).encode('utf-8'))
    else:
        yield from db.execute(statement + ' ORDER BY ' + q(keys[table]))


def typed(value):
    if value is None:
        return b'N', b''
    if isinstance(value, str):
        return b'S', value.encode('utf-8')
    if type(value) is int:
        return b'I', str(value).encode('ascii')
    if type(value) is float:
        return b'F', struct.pack('!d', value)
    if isinstance(value, bytes):
        return b'B', value
    raise CheckFailed()


def digest_rows(table, columns, rows):
    header = json.dumps([table, columns], ensure_ascii=False, separators=(',', ':')).encode()
    digest = hashlib.sha256(b'funes-capacity-source-v1\0' + header)
    col_digests = {c: hashlib.sha256(c.encode() + b'\0') for c in columns}
    col_bytes = dict.fromkeys(columns, 0)
    nul_fields, escaped_fields = {}, {}
    counts, payload = 0, 0
    row_bytes = []
    for row in rows:
        require(len(row) == len(columns), 'row_shape')
        digest.update(b'R')
        row_size = 0
        for col, value in zip(columns, row):
            tag, data = typed(value)
            frame = tag + struct.pack('!Q', len(data))
            digest.update(frame)
            digest.update(data)
            col_digests[col].update(frame)
            col_digests[col].update(data)
            col_bytes[col] += len(data)
            row_size += len(data)
            if isinstance(value, str):
                if '\0' in value:
                    nul_fields[col] = nul_fields.get(col, 0) + 1
                if '\ue000' in value:
                    escaped_fields[col] = escaped_fields.get(col, 0) + 1
        row_bytes.append(row_size)
        payload += row_size
        counts += 1
    row_bytes.sort()
    def quantile(p):
        return row_bytes[min(int(p * max(counts - 1, 0)), counts - 1)] if counts else 0
    return {'rows': counts, 'sha256': digest.hexdigest(),
            'column_sha256': {c: d.hexdigest() for c, d in col_digests.items()},
            'payload_bytes': payload, 'column_payload_bytes': col_bytes,
            'nul_fields': nul_fields, 'literal_escape_fields': escaped_fields,
            'row_payload_bytes': {'min': quantile(0), 'p50': quantile(.50),
                                  'p90': quantile(.90), 'p99': quantile(.99),
                                  'max': quantile(1), 'mean': payload / max(counts, 1)}}


def summary_equal(left, right):
    return left['rows'] == right['rows'] and left['sha256'] == right['sha256']


def build_sample():
    progress('sample_start')
    start = time.monotonic()
    WORK.mkdir(mode=0o700)
    src = sqlite3.connect('file:' + str(SOURCE) + '?mode=ro&immutable=1', uri=True)
    src.execute('PRAGMA query_only=ON')
    src.execute('PRAGMA cache_size=-16384')
    src.execute('PRAGMA mmap_size=0')
    initial = signature(SOURCE)
    count = src.execute('SELECT count(*) FROM memories').fetchone()[0]
    lo = src.execute('SELECT min(id) FROM memories').fetchone()[0]
    hi = src.execute('SELECT max(id) FROM memories').fetchone()[0]
    require(count == EXPECTED_ROWS, 'source_count_changed')
    require(hi - lo + 1 == count, 'source_ids_not_contiguous')
    ids = [lo + i * (hi - lo) // (SAMPLE_ROWS - 1) for i in range(SAMPLE_ROWS)]
    require(len(set(ids)) == SAMPLE_ROWS, 'sample_ids_not_unique')
    dst = sqlite3.connect(str(WORK / 'sample.sqlite3'))
    dst.execute('PRAGMA cache_size=-16384')
    dst.execute('PRAGMA journal_mode=DELETE')
    columns = {}
    counts = {}
    source_schema_hashes = {}
    for table in TABLES:
        create = src.execute('SELECT sql FROM sqlite_master WHERE type=\'table\' AND name=?', (table,)).fetchone()[0]
        dst.execute(create)
        source_schema_hashes[table] = hashlib.sha256(create.encode()).hexdigest()
        columns[table] = [r[1] for r in src.execute('PRAGMA table_info(' + q(table) + ')')]
        fields = ','.join(q(c) for c in columns[table])
        insert = 'INSERT INTO ' + q(table) + '(' + fields + ') VALUES (' + ','.join('?' for _ in columns[table]) + ')'
        if table == 'memories':
            row_query = 'SELECT ' + fields + ' FROM memories WHERE id=?'
            for offset in range(0, len(ids), 100):
                rows = [src.execute(row_query, (ident,)).fetchone() for ident in ids[offset:offset + 100]]
                require(all(row is not None for row in rows), 'source_sample_row_missing')
                dst.executemany(insert, rows)
                if (offset + 100) % 2000 == 0:
                    print(json.dumps({'stage': 'sampling', 'rows': offset + 100}), flush=True)
        else:
            cursor = src.execute('SELECT ' + fields + ' FROM ' + q(table))
            while True:
                rows = cursor.fetchmany(100)
                if not rows:
                    break
                dst.executemany(insert, rows)
        counts[table] = dst.execute('SELECT count(*) FROM ' + q(table)).fetchone()[0]
        dst.commit()
        copied_schema = dst.execute('SELECT sql FROM sqlite_master WHERE type=\'table\' AND name=?', (table,)).fetchone()[0]
        require(copied_schema == create, 'sample_table_schema_changed')
    dst.close()
    src.close()
    require(signature(SOURCE) == initial, 'source_stat_changed_during_sampling')
    SUMMARY['source_stat_before'] = initial
    SUMMARY['source_rows'] = count
    SUMMARY['sample_table_counts'] = counts
    SUMMARY['sample_selection'] = 'deterministic equally spaced source IDs, endpoints included; verified dense ID interval'
    SUMMARY['source_schema_sha256'] = source_schema_hashes
    SUMMARY['sample_file_bytes'] = (WORK / 'sample.sqlite3').stat().st_size
    SUMMARY['sampling_seconds'] = time.monotonic() - start
    sample = sqlite3.connect('file:' + str(WORK / 'sample.sqlite3') + '?mode=ro&immutable=1', uri=True)
    sample.execute('PRAGMA query_only=ON')
    sample.execute('PRAGMA cache_size=-16384')
    expected = {table: digest_rows(table, columns[table], select_source_rows(sample, table, columns[table])) for table in TABLES}
    SUMMARY['source_sample'] = expected
    SUMMARY['source_sample_total_payload_bytes'] = sum(d['payload_bytes'] for d in expected.values())
    progress('sample_complete', rows=count, sample_rows=SAMPLE_ROWS,
             sample_payload_bytes=SUMMARY['source_sample_total_payload_bytes'],
             sampling_seconds=SUMMARY['sampling_seconds'])
    return sample, columns, expected


def relation_sizes(raw):
    query = '''SELECT c.relname, c.relkind,
       pg_relation_size(c.oid), pg_table_size(c.oid), pg_indexes_size(c.oid),
       CASE WHEN c.reltoastrelid=0 THEN 0 ELSE pg_total_relation_size(c.reltoastrelid) END,
       pg_total_relation_size(c.oid)
       FROM pg_class c JOIN pg_namespace n ON n.oid=c.relnamespace
       WHERE n.nspname=%s AND c.relkind IN ('r','m','S') ORDER BY c.relname'''
    result = {}
    with raw.cursor() as cursor:
        cursor.execute(query, (SCHEMA,))
        for row in cursor:
            name, kind, heap, table, indexes, toast, total = tuple(row)
            result[name] = {'kind': kind, 'heap_main_bytes': heap, 'table_including_toast_bytes': table,
                            'base_index_bytes': indexes, 'toast_including_own_indexes_bytes': toast,
                            'other_forks_bytes': total - heap - indexes - toast, 'total_bytes': total}
        cursor.execute('''SELECT ci.relname, pg_total_relation_size(ci.oid)
          FROM pg_index i JOIN pg_class ct ON ct.oid=i.indrelid
          JOIN pg_namespace n ON n.oid=ct.relnamespace JOIN pg_class ci ON ci.oid=i.indexrelid
          WHERE n.nspname=%s ORDER BY ci.relname''', (SCHEMA,))
        index_sizes = {row[0]: row[1] for row in cursor}
    return {'relations': result, 'indexes': index_sizes,
            'total_bytes': sum(v['total_bytes'] for v in result.values())}



def main(argv=None):
    import psycopg
    from psycopg import sql
    from psycopg.conninfo import conninfo_to_dict, make_conninfo
    from scripts.migrate_source_postgres import Snapshot, Migration
    from service.postgres import PostgresStore, POSTGRES_SCHEMA_VERSION

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--baseline", required=True, help="Sanitized historical 10k sample digest JSON")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    # Fixed cap: this command cannot accidentally become a full importer.
    global SOURCE, WORK, SAMPLE_ROWS, EXPECTED_ROWS, SCHEMA
    SAMPLE_ROWS = 10000
    baseline = json.loads(Path(args.baseline).read_text())
    EXPECTED_ROWS = int(baseline["source_rows"])
    SOURCE = Path(args.source).resolve(strict=True)
    dsn = os.environ.get("FUNES_TEST_POSTGRES_DSN", "")
    options = conninfo_to_dict(dsn)
    require(str(options.get("host", "")).startswith("/") and
            str(options.get("dbname", "")).endswith("_test") and
            not options.get("hostaddr"), "local_disposable_database_required")
    os.umask(0o077)
    started = time.monotonic()
    before = signature(SOURCE)
    require(before["size"] == baseline["source_file_bytes"], "source_size_changed")
    for suffix in ("-wal", "-journal"):
        sidecar = Path(str(SOURCE) + suffix)
        require(not sidecar.exists() or sidecar.stat().st_size == 0, "source_not_frozen")
    progress("hash_frozen_source")
    with SOURCE.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    require(digest == baseline["source_sha256"], "source_hash_changed")
    SUMMARY.update(source_sha256=digest, schema_version=POSTGRES_SCHEMA_VERSION,
                   source_writes=False, production_requests=False, status="running")
    admin = sample = None
    owned_oid = None
    SCHEMA = "funes_capacity_v2_" + uuid.uuid4().hex
    with tempfile.TemporaryDirectory(prefix="funes-capacity-v2-") as directory:
        WORK = Path(directory) / "sample"
        try:
            sample, columns, expected = build_sample()
            require(all(expected[t]["sha256"] == baseline["source_sample"][t]["sha256"]
                        for t in TABLES), "historical_sample_digest_mismatch")
            SUMMARY["same_sample_as_original"] = True
            lengths = sorted(row[0] for row in sample.execute("SELECT length(CAST(raw_text AS BLOB)) FROM memories"))
            SUMMARY["raw_text_bytes"] = dict(total=sum(lengths), mean=statistics.mean(lengths),
                median=statistics.median(lengths), p95_nearest_rank=lengths[math.ceil(.95*len(lengths))-1],
                min=lengths[0], max=lengths[-1])
            sample.close(); sample = None
            admin = psycopg.connect(dsn, autocommit=True, connect_timeout=10)
            require(admin.info.dbname.endswith("_test"), "unexpected_database")
            require(admin.info.server_version // 10000 == 16, "comparison_requires_pg16")
            SUMMARY["server_version_num"] = admin.info.server_version
            admin.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(SCHEMA)))
            owned_oid = admin.execute("SELECT oid FROM pg_namespace WHERE nspname=%s", (SCHEMA,)).fetchone()[0]
            local_dsn = make_conninfo(dsn, options="-csearch_path=" + SCHEMA +
                " -cclient_min_messages=error -cstatement_timeout=240000 -clock_timeout=10000"
                " -cmaintenance_work_mem=32MB -cwork_mem=16MB -cmax_parallel_maintenance_workers=0"
                " -cmax_parallel_workers_per_gather=0 -cdefault_toast_compression=pglz")
            progress("migrate_sample", rows=SAMPLE_ROWS)
            with Snapshot(WORK / "sample.sqlite3") as frozen, Migration(frozen, local_dsn) as migration:
                migration.initialize()
                copy = migration.copy(batch_rows=500, batch_bytes=8*1024*1024)
                require(copy["phase"] == "verified" and not copy["ready"], "sample_copy_unverified")
                final = migration.finalize(tail_confirmed=True)
                require(final["ready"], "sample_not_ready")
                SUMMARY["migration_logical_readback"] = {
                    "baseline_all_columns_equal": True,
                    "before_finalize_all_columns_equal": final["before_finalize_tables"] == copy["tables"],
                    "final_all_columns_equal_except_declared_derived_state": True,
                    "rebuilt_fields": final["rebuilt_fields"], "final_table_digests": final["tables"],
                }
                pg = migration.pg
                pg.execute("ANALYZE memories")
                SUMMARY["sizes"] = relation_sizes(pg)
                row = pg.execute("SELECT count(*),count(*) FILTER (WHERE retrieval_text IS NULL),"
                    "COALESCE(sum(octet_length(raw_text)) FILTER (WHERE retrieval_text IS NULL),0),"
                    "COALESCE(sum(octet_length(retrieval_text)),0) FROM memories").fetchone()
                SUMMARY["physical_shadow"] = dict(rows=row[0], null_rows=row[1],
                    omitted_equal_text_bytes=row[2], stored_nonnull_text_bytes=row[3])
                SUMMARY["index_definitions"] = dict(pg.execute(
                    "SELECT indexname,indexdef FROM pg_indexes WHERE schemaname=current_schema() ORDER BY indexname").fetchall())
                SUMMARY["memory_columns"] = [dict(name=r[0], type=r[1], nullable=r[2], generated=r[3]) for r in pg.execute(
                    "SELECT column_name,data_type,is_nullable,is_generated FROM information_schema.columns "
                    "WHERE table_schema=current_schema() AND table_name='memories' ORDER BY ordinal_position")]
                SUMMARY["index_methods"] = sorted({r[0] for r in pg.execute(
                    "SELECT am.amname FROM pg_index i JOIN pg_class c ON c.oid=i.indexrelid JOIN pg_am am ON am.oid=c.relam "
                    "JOIN pg_namespace n ON n.oid=c.relnamespace WHERE n.nspname=current_schema()")})
                require(SUMMARY["index_methods"] == ["btree"], "unexpected_index_method")
                toast = pg.execute("SELECT pg_relation_size(c.reltoastrelid),pg_indexes_size(c.reltoastrelid),"
                                   "pg_table_size(c.reltoastrelid) FROM pg_class c WHERE c.oid='memories'::regclass").fetchone()
                SUMMARY["toast_detail"] = dict(main_bytes=toast[0], index_bytes=toast[1], table_including_forks_bytes=toast[2])
            store = PostgresStore(str(WORK / "runtime"), local_dsn)
            try:
                require(store.count() == SAMPLE_ROWS and not store.fts_ready(), "runtime_contract")
                store.verify_schema()
                SUMMARY["runtime_ready_without_pg_fts"] = True
            finally:
                store.close()
            sizes = SUMMARY["sizes"]
            mem = sizes["relations"]["memories"]
            factor = EXPECTED_ROWS / SAMPLE_ROWS
            fixed = sizes["total_bytes"] - mem["total_bytes"]
            SUMMARY["projection"] = dict(full_rows=EXPECTED_ROWS, sample_rows=SAMPLE_ROWS, factor=factor,
                components_bytes={k:v*factor for k,v in mem.items() if k.endswith("_bytes")},
                fixed_auxiliary_bytes=fixed, single_replica_total_bytes=mem["total_bytes"]*factor+fixed,
                single_replica_total_gib=(mem["total_bytes"]*factor+fixed)/1024**3,
                replicated=False, budget_gib=20,
                excludes=["WAL", "backups", "temporary maintenance space", "future growth", "other schemas"])
            SUMMARY["source_stat_unchanged"] = signature(SOURCE) == before
            require(SUMMARY["source_stat_unchanged"], "source_stat_changed")
            SUMMARY["status"] = "passed"
        finally:
            if sample is not None:
                sample.close()
            if admin is not None:
                try:
                    if owned_oid is not None:
                        found = admin.execute("SELECT oid FROM pg_namespace WHERE nspname=%s", (SCHEMA,)).fetchone()
                        require(found and found[0] == owned_oid and SCHEMA.startswith("funes_capacity_v2_"), "cleanup_ownership_guard")
                        admin.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(SCHEMA)))
                        SUMMARY["owned_schema_removed"] = True
                finally:
                    admin.close()
    SUMMARY["temporary_sample_removed"] = not WORK.exists()
    SUMMARY["elapsed_seconds"] = time.monotonic() - started
    SUMMARY["client_peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024)
    output = Path(args.output)
    output.write_text(json.dumps(SUMMARY, indent=2) + "\n")
    progress("complete", single_replica_gib=SUMMARY["projection"]["single_replica_total_gib"],
             elapsed_seconds=SUMMARY["elapsed_seconds"])


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        # Database failures may contain source values: never print exception text.
        print(json.dumps({"status": "failed", "error_type": type(error).__name__,
                          "check": str(error) if isinstance(error, CheckFailed) else None}), file=sys.stderr)
        raise SystemExit(1)
