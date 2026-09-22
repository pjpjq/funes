# One-time source-store migration: frozen SQLite → PostgreSQL

Run `/path/to/funes/scripts/migrate_source_postgres.py` on an operator host, **not
inside the live application startup**. This copies only `memories`,
`translation_cache`, `reindex_controls`, and `sync_state`. It does not reingest
history, call Hugging Face, build embeddings, rebuild native indexes, translate,
or alter the Lance memory. Run against a **new, dedicated PostgreSQL database or
schema** with enough storage for table/index/WAL growth. Keep the old source and
its backups until a separately authorized cutover/rollback window closes.

## Review status and v2 storage boundary

**September 22, 2026: code and isolated 10k-sample review only. No production
PostgreSQL/HF cutover or full import has been performed by this change.**
The same-sample capacity result is **17.0879 GiB per replica**, excluding WAL,
backups, temporary maintenance space and future growth. See
[source-postgres-capacity.md](source-postgres-capacity.md) for the complete
measurement, formula, schema and verification boundaries.

Schema version 2 is **B plus nonduplicate retrieval shadow**: raw/source data,
metadata, `search_identifiers`, small B-tree indexes, and only genuinely different
`retrieval_text`. It creates **no stored/generated tsvector, GIN, GiST, pgvector,
trigram or content-search index**. `search_identifiers` remains source payload,
not a PostgreSQL full-text index. Voyage and Lance retain semantic/BM25 retrieval;
PostgreSQL serves durable source state, exact identity lookup and raw hydration.

The importer also has a version-2 ledger. Runtime rejects v1, incomplete or
incompatible schemas without DDL, fallback or an automatic rebuild. The commands
below describe a future explicitly approved migration into a fresh v2 target;
they are **not** an instruction to alter an existing production schema. Existing
v1 data must be retained until a separately reviewed conversion/cutover.

## 1. Pin the immutable source (read-only, no database credentials)

Create a SQLite backup from the authoritative source, stop modifying that backup,
and retain the original checksum. Do not point at an active WAL database. The
importer opens SQLite with `mode=ro&immutable=1`, rejects nonempty WAL/journal
sidecars, hashes the complete file on every invocation, pins size/schema/counts,
and checks inode/size/mtime/ctime around batches. A changed source cannot resume.

The inspected Alice backup on September 22, 2026 is:

```text
/root/funes-source-snapshot/funes.sqlite3
bytes: 25935802368
sha256: 8f72d49271e326f30acb489b921c686ac3b4c4186cd991d251475205c4467003
memories: 3590437 (IDs 1 through 3590437)
translation_cache: 63
sync_state: 1
reindex_controls: 0
```

These are a pinned historical snapshot, **not a claim about the current Hub
tail**. Dry-run reads source data only and prints table counts, digests, and the
number of fields containing NUL; no source values are printed:

```sh
python /path/to/funes/scripts/migrate_source_postgres.py \
  --source /root/funes-source-snapshot/funes.sqlite3 \
  --expected-sha256 8f72d49271e326f30acb489b921c686ac3b4c4186cd991d251475205c4467003 \
  --dry-run
```

The whole-file hash and full streaming digest intentionally read the snapshot;
on a large snapshot this is not a quick metadata-only probe. Do not start
multiple verification jobs against the same disk.

## 2. Initialize and copy into the new destination, never mark ready

Install the service requirements, including Psycopg 3. Inject
`FUNES_POSTGRES_DSN` through the host's secret environment or a mode-0600 secret
file; never paste it into command arguments, shell history, stdout, or a report.
Do not enable shell tracing. This script intentionally has no `--dsn` option.
The database role needs schema creation/ownership, COPY, sequence, table locking,
and permission to disable its own user triggers; superuser/replication privileges
are not required.

```sh
python /path/to/funes/scripts/migrate_source_postgres.py \
  --source /root/funes-source-snapshot/funes.sqlite3 \
  --expected-sha256 8f72d49271e326f30acb489b921c686ac3b4c4186cd991d251475205c4467003 \
  --initialize --defer-ready --batch-rows 500 --batch-bytes 8388608
```

The source's exact column set and types must match the destination. Initialization
calls `PostgresStore.prepare_bulk_migration()`: primary/unique constraints stay
present; the seven nonunique B-tree indexes are deferred and both native triggers
stay disabled until explicit finalization. No lexical vector is created, stored
or rebuilt during COPY or finalization. Offline schema checks
accept exactly both native triggers enabled or both disabled, never a mixed or
missing trigger set. Resume uses the existing schema without rebuilding indexes.
Existing source rows, a different schema/version, an already-ready destination, or an
existing unrelated checkpoint are refused. The only restart exception is the
exact empty bootstrap schema/row left if a process dies before creating its
migration ledger. No source data or unrelated destination data is deleted.

Each bounded COPY batch commits **with** its cursor in
`funes_source_migration`; after a crash it is either fully present or absent.
Rows stream without materializing the database. The defaults cap batches at 500
rows or approximately 8 MiB of encoded text (one oversized row is unavoidable).
The application marker remains `migration_ready=false` during COPY and after
successful baseline verification. Application restart must fail closed.

`memories` COPY disables only the two native counter/pending triggers under an
`ACCESS EXCLUSIVE` lock in the same transaction, then restores their prior state
before commit. For a bulk-prepared import they remain disabled between batches.
An interruption rolls back that batch's data, cursor, and trigger changes. This
preserves each baseline `native_index_pending` value and avoids double-counting.
The entire original `sync_state` row is copied through staging and an upsert,
preserving every revision, counter, optimize marker, and cursor for exact
baseline verification. Explicit finalization later rebuilds only the derived
fields listed in section 4.

All text columns use the shared reversible PostgreSQL codec exactly once.
PostgreSQL TEXT cannot store NUL: `U+E000` becomes `U+E000 e`, and NUL becomes
`U+E000 0`. Escape-marker collisions are escaped too. Readback decodes exactly
once. Raw/retrieval strings, metadata JSON whitespace, Unicode normalization,
newlines and empty values are never silently normalized or stripped.

The sole physical compaction rule is exact equality before text encoding:
`retrieval_text == raw_text` is stored as `retrieval_text=NULL`. Live writes enforce
this in the native BEFORE trigger; COPY applies the same rule explicitly because
triggers are disabled. Logical reads, native handoff, snapshots and digest
verification expand it as `COALESCE(retrieval_text, raw_text)`. An explicit empty
shadow with a nonempty raw value stays `''`, not NULL. Empty raw plus empty shadow
may use NULL because its logical value is still the empty string. Compaction
alone does not advance the native revision; real raw/shadow changes still do.

## 3. Resume and independently verify PostgreSQL's actual rows

After an interruption, use the **same immutable source** and destination:

```sh
python /path/to/funes/scripts/migrate_source_postgres.py \
  --source /root/funes-source-snapshot/funes.sqlite3 \
  --expected-sha256 8f72d49271e326f30acb489b921c686ac3b4c4186cd991d251475205c4467003 \
  --resume --defer-ready
```

`--max-batches 1` can be added to initialization/resume for a small controlled
probe. It leaves a durable resumable checkpoint and never marks ready. Resume
re-reads all already-committed prefixes from both SQLite and PostgreSQL; corrupted
rows or extra target rows cause refusal before more COPY work.

After all four tables are copied, the importer uses a server-side cursor to read
actual PostgreSQL source columns, decodes text, and compares **row count plus a
deterministic SHA-256 per table** against streaming SQLite rows. These are not
inserter-maintained counters or hashes. The digest length-frames type/value bytes
and distinguishes NULL, empty string, integer, Unicode normalization, NUL, and
field boundaries. Integer primary keys sort numerically; translation-cache keys
sort by encoded UTF-8 bytes on both sides, independent of database collation.
Version 2 refuses generated columns and non-B-tree indexes. The physical compact
NULL is expanded before text decoding and hashing, preserving the original
SQLite logical retrieval value in every equality check.

Only after full equality does the importer advance the ID sequence and store
`phase=verified`; readiness is still false. An explicit repeat is available:

```sh
python /path/to/funes/scripts/migrate_source_postgres.py \
  --source /root/funes-source-snapshot/funes.sqlite3 --verify
```

Counts alone are insufficient. Preserve the JSON digest report with the
operator's change record, but not credentials or raw source data. Validation
failures cannot grant readiness. A disconnect or interruption at commit can
make the result ambiguous: inspect `funes_schema_state.migration_ready` and
`funes_source_migration.phase` before retrying, rather than assuming an error
means nothing committed. The CLI suppresses raw database exceptions because
they can contain SQL/source values.

## 4. Stop old writers, catch up the incremental tail, then explicitly finalize

The importer **does not automatically replay HF history or fetch a tail**. The
cutover coordinator must stop the old writers, identify the exact boundary of
the pinned backup, apply only subsequent records/controls/checkpoints through
an explicitly offline migration-capable adapter, and verify that no old writer
can advance the source after that boundary. Include changed existing rows,
deletions, controls, translation-cache changes, and checkpoints, not just IDs
above the baseline maximum. Opening a source with `fts_ready=0` in SQLite's
runtime Store can itself update existing `search_identifiers`; final verification
will reject a tail that omits those updates. No live application may use a
partial target. Use this script's finalization gate rather than calling the
store's `finalize_migration()` directly: the script wraps that method with
independent pre/post readback and atomic verification records.

If no tail changed source state, finalization rechecks the baseline itself:

```sh
python /path/to/funes/scripts/migrate_source_postgres.py \
  --source /root/funes-source-snapshot/funes.sqlite3 \
  --finalize --tail-confirmed
```

If incremental catchup changed any row/checkpoint, create an **independent frozen
SQLite backup of the final authoritative state** and compare PostgreSQL against
that, retaining the original `--source` identity for the migration ledger:

```sh
python /path/to/funes/scripts/migrate_source_postgres.py \
  --source /root/funes-source-snapshot/funes.sqlite3 \
  --expected-sha256 8f72d49271e326f30acb489b921c686ac3b4c4186cd991d251475205c4467003 \
  --finalize --tail-confirmed \
  --tail-source /secure/final-frozen-source.sqlite3 \
  --tail-expected-sha256 FINAL_FROZEN_SNAPSHOT_SHA256
```

`--tail-confirmed` is an explicit operator attestation, not an automatic Hub
freshness check. Finalization restores deferred indexes/triggers while readiness
is still false, then takes exclusive source/marker/ledger locks. Within one
transaction it first checks all four tables **exactly** against the authoritative
frozen source, calls `PostgresStore.finalize_migration(expected_count=..., source=...)`,
and independently reads every PostgreSQL source row again. The second comparison
reconstructs the permitted derived changes directly from SQLite, not from the
store's reported counters. Only these source columns may change:

- `memories.native_index_pending` is recomputed from eligibility, native status,
  and the imported checkpoint profile/memory.
- `sync_state.fts_ready`, `fts_schema_version`,
  `native_checkpoint_state_version`, `native_eligible_count`,
  `native_indexed_count`, `native_held_count`, and `native_invalid_count` are
  rebuilt to the supported runtime versions and independently counted values.

PostgreSQL `fts_ready` and `fts_schema_version` are deliberately finalized to
zero, independent of source readiness. An imported SQLite FTS-ready marker is not
carried forward as a false claim that PostgreSQL supplies content search.
Raw/logical retrieval text, metadata,
search identifiers, source/native generations, native/index/optimize revisions,
profile/memory identifiers, paid-index cursors, control state, and every other
source field must remain exactly equal. This only rebuilds native pending/count
checkpoints and disables PostgreSQL FTS markers; it is **not** lexical index
reconstruction, embedding, paid-provider work, or Lance reindexing. The report records `before_finalize_tables`, final `tables`,
`rebuilt_fields`, and `expected_derived_sync_state` so the distinction is explicit.

The final runtime-schema check, correct `document_count`, ready marker, and
final verification record are committed together. Any validation mismatch rolls
back the derived data and readiness changes; already-built indexes may remain,
which is safe while the target stays unready. A retry must still pass both full
comparisons. Only after success may the coordinator configure PostgreSQL and run
fresh health/readiness/get/recall/ingest checks. Keep Lance/HF embedding
configuration unchanged. A ready target cannot be reimported with this script.

## 5. Validate the importer safely

The no-database tests run without PostgreSQL. For full tests, inject
`FUNES_TEST_POSTGRES_DSN` for a **disposable** database; the fixture creates unique
`migration_test_*` schemas and deletes only its own schemas on completion:

```sh
cd /path/to/funes
PYTHONPATH=.:service python -m pytest -q service/tests/test_postgres_migration.py
```

Tests cover transaction interruption/resume, NUL/escape-marker collisions,
raw/metadata/checkpoint/cursor preservation, independent corruption detection,
source/schema identity refusal, nonempty-target refusal, startup marker guards,
sequence advancement, concurrent migrator refusal, and independently verified
incremental-tail finalization. Fault-injection tests alter raw bytes, protected
revisions, derived pending flags, or a required B-tree index after finalization and
verify that the outer transaction restores source state and keeps readiness
false; a clean retry must succeed. Tests also check deferred indexes/triggers
and recovery after an interrupted pristine bulk preparation. They never connect
to production unless an operator incorrectly supplies a production test DSN.
