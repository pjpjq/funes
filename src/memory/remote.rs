//! The remote side of a memory: how its Lance dataset is read from and written to a Hub repo.
//!
//! [`append`] adds rows; [`reindex`] folds the unindexed backlog into the FTS/IVF indexes. Each
//! runs a native Lance op and lands the result in one or more bounded `create_commit` calls on the
//! branch, guarded by a chained `parent_commit` against the head it read. Activation metadata is
//! always uploaded last, so readers keep seeing the previous manifest until the final chunk. Each
//! operation is a single attempt: if the head moved first it reports a conflict
//! ([`Appended::Conflict`] / [`Reindexed::Conflict`]) and the caller retries against the new head.
//!
//! The result goes up as bounded `create_commit` calls because the Hub rejects a commit with more
//! than 1000 files. Lance, left to write straight to `hf://`, would commit each file on its own:
//! that store is OpenDAL's HuggingFace service, where every `put` is its own git commit.
//!
//! [`compact`] folds the unindexed backlog into the FTS/IVF indexes, building any the dataset
//! lacks, compacts the fragments and deletes the old versions, in several commits.
//!
//!
//! ```text
//!   Lance Dataset → object_store → OpenDAL hf service → HF Hub
//!       put = XET upload + one git commit, per file
//! ```
//!
//! So each op runs through a [`CaptureStore`](super::capture_store::CaptureStore) installed via
//! Lance's [`WrappingObjectStore`] seam: Lance's writes are captured in temporary files instead of hitting
//! the Hub, and we choose the commits that ship them.
//!
//! # Why this shape
//!
//! **Intercept at the object-store layer.** Every file an append or optimize produces — data
//! fragment, manifest, transaction, index — is written through `object_store`, so it is the one
//! hook that captures the *whole* write set with no knowledge of Lance's on-disk layout. A
//! narrower seam can't do it: a custom `CommitHandler` only governs the final manifest commit and
//! never sees the data fragments, which are written earlier.
//!
//! **Decorate Lance's object store rather than inject our own.** Lance does support dependency injection
//! (`DatasetBuilder::with_object_store`, now deprecated, or an `ObjectStoreProvider`), but both
//! make *us* construct the HF object store — reproducing Lance's OpenDAL-hf setup, XET wiring, and
//! token/revision plumbing, and keeping it in lockstep. [`WrappingObjectStore`] instead hands us
//! the object store Lance already built (`wrap`'s `original`), so we decorate it and never reconstruct
//! anything. It is also the non-deprecated seam.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::future::Future;
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{ensure, Context, Result};
use arrow_array::{new_null_array, RecordBatch, RecordBatchIterator};
use arrow_schema::{Schema, SchemaRef};
use async_trait::async_trait;
use bytes::Bytes;
use futures::FutureExt;
use hf_hub::progress::{Progress, ProgressEvent, ProgressHandler, UploadEvent};
use hf_hub::repository::{CommitInfo, CommitOperation, RepoTreeEntry};
use hf_hub::{HFError, HFRepository, RepoTypeDataset};
use lance::dataset::transaction::{Operation, Transaction};
use lance::dataset::{ColumnAlteration, CommitBuilder, Dataset, NewColumnTransform, WriteParams};
use lance::Error as LanceError;
use lance_io::object_store::{ChainedWrappingObjectStore, WrappingObjectStore};
use object_store::list::PaginatedListStore;
use object_store::ObjectStore as OSObjectStore;
use serde::Serialize;

use super::capture_store::{CaptureStore, Captured, CapturedFile};
use super::dataset::{self, IndexBuildEvent};
use super::fetch_store::{FetchStore, FileFetcher};
use super::shard_store::{physical_path, ShardStore};
use crate::hub;

#[derive(Clone, Debug, Serialize)]
struct IngestPhaseMetric {
    stage: &'static str,
    duration_ms: f64,
}

fn ingest_metrics_enabled_from(value: Option<&str>) -> bool {
    matches!(
        value.map(|val| val.trim().to_ascii_lowercase()).as_deref(),
        Some("1" | "true" | "yes" | "on")
    )
}

fn ingest_metrics_enabled() -> bool {
    ingest_metrics_enabled_from(std::env::var("FUNES_INGEST_METRICS").ok().as_deref())
}

fn round_ms(duration: Duration) -> f64 {
    ((duration.as_secs_f64() * 1000.0) * 100.0).round() / 100.0
}

fn write_metric(writer: &mut impl std::io::Write, json: &str) -> std::io::Result<()> {
    // A failed upload leaves the progress bar unterminated: start a machine-readable line.
    writeln!(writer, "\nfunes_metric {json}")
}

fn emit_phase_metric(stage: &'static str, duration: Duration) {
    if !ingest_metrics_enabled() {
        return;
    }
    let metric = IngestPhaseMetric {
        stage,
        duration_ms: round_ms(duration),
    };
    if let Ok(json) = serde_json::to_string(&metric) {
        let _ = write_metric(&mut std::io::stderr().lock(), &json);
    }
}

/// Outcome of an [`append`] commit.
pub(crate) enum Appended {
    /// The data was committed; carries the new commit oid, the resulting unindexed-row backlog, and
    /// whether the dataset has a text index at all.
    Committed {
        oid: String,
        unindexed: u64,
        text_indexed: bool,
    },
    /// The branch head moved before our commit; the caller may retry against the new head.
    Conflict,
}

/// Outcome of a [`compact`].
pub(crate) enum Compacted {
    /// Committed, with the last commit oid.
    Committed(String),
    /// The new version landed, with its commit oid, but deleting the old ones failed. The next
    /// compaction deletes them.
    Uncleaned(String, anyhow::Error),
    /// Nothing to optimize or delete.
    AlreadyCompact,
    /// The head moved in a way the compaction can't build on. The caller may retry against the new
    /// head.
    Conflict,
}

pub(crate) enum Reindexed {
    /// The index delta was committed; carries the new commit oid.
    Committed(String),
    /// Nothing to optimize — the index was already current.
    AlreadyCurrent,
    /// The branch head moved before our commit; the caller may retry against the new head.
    Conflict,
}

/// Outcome of one canonical-document replace attempt.
pub(crate) enum Replaced {
    Committed(String),
    /// The Hub branch moved after the dataset version was read. The caller must reopen it and
    /// re-evaluate source revisions before retrying.
    Conflict,
}

/// Append `batches` to the remote Lance dataset at `dataset_uri` (an `hf://…/<table>.lance` URI)
/// and land them in one `create_commit` on branch `rev`, guarded by the current head. The append
/// writes only data and the language counts, and leaves the new rows
/// unindexed (refresh the index separately with [`compact`]). `extra_files` (repo path → bytes,
/// e.g. the dataset card) ride the same guarded commit; cloned per attempt, so a conflict retry
/// re-attaches them. Returns [`Appended::Committed`] with the new commit oid, the resulting
/// unindexed-row backlog (the largest across the dataset's indexes — what `push` thresholds on) and
/// whether a text index exists, or [`Appended::Conflict`] if the head moved first — a single
/// attempt against the head it read, so the caller drives the retry.
#[allow(clippy::too_many_arguments)] // internal orchestration, one call site (`push`)
pub(crate) async fn append(
    repo: &HFRepository<RepoTypeDataset>,
    dataset_uri: &str,
    storage_options: HashMap<String, String>,
    rev: &str,
    message: String,
    batches: Vec<RecordBatch>,
    schema: SchemaRef,
    extra_files: &BTreeMap<String, Bytes>,
) -> Result<Appended> {
    let parent = head_oid(repo, rev).await?;
    append_at(
        repo,
        repo,
        dataset_uri,
        storage_options,
        parent,
        rev,
        message,
        batches,
        schema,
        extra_files,
        true,
    )
    .await
}

#[allow(clippy::too_many_arguments)] // Shared CAS append boundary for push and canonical ingestion.
async fn append_at(
    _repo: &HFRepository<RepoTypeDataset>,
    commit_repo: &HFRepository<RepoTypeDataset>,
    dataset_uri: &str,
    storage_options: HashMap<String, String>,
    parent: String,
    rev: &str,
    message: String,
    batches: Vec<RecordBatch>,
    schema: SchemaRef,
    extra_files: &BTreeMap<String, Bytes>,
    measure_unindexed: bool,
) -> Result<Appended> {
    // Open the same immutable head guarded by the eventual Hub commit. Without
    // this pin a branch move between selection and open could duplicate rows.
    let (mut ds, wrapper) = open_pinned(dataset_uri, &storage_options, &parent).await?;
    same_model(&ds, &schema)?;
    if schema.column_with_name("source_identity").is_some() {
        dataset::ensure_canonical_columns(&mut ds).await?;
    }

    // Counts that miss rows stay as they are: recounting them would read the whole remote.
    let rows = dataset::count_rows(&ds, None).await? as u64;
    let mut languages = dataset::languages(&ds).filter(|languages| languages.rows == rows);
    if let Some(languages) = &mut languages {
        for batch in &batches {
            languages.add(batch);
        }
    }
    let target_schema = Arc::new(Schema::from(ds.schema()));
    let batches = batches
        .into_iter()
        .map(|batch| align_batch(batch, target_schema.clone()))
        .collect::<Result<Vec<_>>>()?;
    let reader = RecordBatchIterator::new(batches.into_iter().map(Ok), target_schema);
    let append_start = Instant::now();
    ds.append(reader, None)
        .await
        .context("appending to the remote dataset")?;
    emit_phase_metric("lance_append", append_start.elapsed());
    if let Some(languages) = &languages {
        dataset::stamp_languages(&mut ds, languages).await?;
    }

    // Snapshot the captured writes before optionally reading index stats: `index_statistics` can
    // write a stats migration through the same wrapper, and that must not leak into the data commit.
    let mut files = captured_files(&wrapper);
    // Canonical ingestion discards this result, so it skips the expensive statistics reads.
    let unindexed = if measure_unindexed {
        dataset::max_unindexed_rows(&ds).await
    } else {
        0
    };
    let text_indexed = measure_unindexed && dataset::sub_index_counts(&ds).await?.contains_key(dataset::FTS_INDEX);
    for (path, body) in extra_files {
        files.insert(path.clone(), CapturedFile::from_bytes(body)?);
    }

    let (ops, _dir) = write_ops(&files)?;
    match send_commit(commit_repo, ops, Some(parent), rev, message).await {
        Ok(info) => Ok(Appended::Committed {
            oid: info.commit_oid.unwrap_or_else(|| "?".to_string()),
            unindexed,
            text_indexed,
        }),
        Err(e) if head_moved(&e) => Ok(Appended::Conflict),
        Err(e) => Err(commit_error("data commit failed", &e)),
    }
}

/// Refuse rows embedded with another model than `ds`'s: Lance appends them whatever model their
/// schema records.
fn same_model(ds: &Dataset, rows: &Schema) -> Result<()> {
    let (there, here) = (dataset::embedding_profile(ds)?, dataset::profile_from_schema(rows)?);
    ensure!(
        here == there,
        "the remote memory is embedded with {}, and the rows to append with {}: a memory takes one embedding model",
        there.model,
        here.model
    );
    Ok(())
}

/// Append canonical rows known to be absent at `expected_parent` without a full merge scan.
#[allow(clippy::too_many_arguments)] // Keep the selected snapshot and CAS target explicit.
pub(crate) async fn append_documents(
    repo: &HFRepository<RepoTypeDataset>,
    commit_repo: &HFRepository<RepoTypeDataset>,
    dataset_uri: &str,
    storage_options: HashMap<String, String>,
    expected_parent: &str,
    rev: &str,
    message: String,
    batches: Vec<RecordBatch>,
    schema: SchemaRef,
) -> Result<Replaced> {
    let extra_files = BTreeMap::new();
    match append_at(
        repo,
        commit_repo,
        dataset_uri,
        storage_options,
        expected_parent.to_string(),
        rev,
        message,
        batches,
        schema,
        &extra_files,
        false,
    )
    .await?
    {
        Appended::Committed { oid, .. } => Ok(Replaced::Committed(oid)),
        Appended::Conflict => Ok(Replaced::Conflict),
    }
}

/// Align an append batch to the remote schema, filling additive nullable columns with null. This
/// lets old transcript memories and canonical-aware memories publish to each other safely.
fn align_batch(batch: RecordBatch, target: SchemaRef) -> Result<RecordBatch> {
    let columns = target
        .fields()
        .iter()
        .map(|field| {
            batch
                .column_by_name(field.name())
                .cloned()
                .unwrap_or_else(|| new_null_array(field.data_type(), batch.num_rows()))
        })
        .collect();
    Ok(RecordBatch::try_new(target, columns)?)
}

/// Build the whole dataset locally (data + indexes) and upload it in bounded chained commits —
/// unlike [`append`]/[`reindex`], the first commit has no head to guard against since the dataset
/// does not exist yet. `None` if the build produced no files.
#[allow(clippy::too_many_arguments)] // internal orchestration, one call site (`push`)
pub(crate) async fn first_publish(
    repo: &HFRepository<RepoTypeDataset>,
    prefix: &str,
    batches: Vec<RecordBatch>,
    schema: SchemaRef,
    rev: &str,
    message: String,
    extra_files: &BTreeMap<String, Bytes>,
    on_event: impl Fn(IndexBuildEvent),
) -> Result<Option<String>> {
    let staging = tempfile::tempdir()?;
    // Empty prefix = dataset at the repo root; joining "" would leave a stray trailing separator.
    let db_dir = if prefix.is_empty() {
        staging.path().to_path_buf()
    } else {
        staging.path().join(prefix)
    };
    std::fs::create_dir_all(&db_dir)?;
    let table_uri = dataset::table_uri(&db_dir);
    let mut languages = dataset::Languages::default();
    for batch in &batches {
        languages.add(batch);
    }
    let reader = RecordBatchIterator::new(batches.into_iter().map(Ok), schema);
    let mut ds = Dataset::write(reader, &table_uri, Some(WriteParams::default()))
        .await
        .context("building the dataset for first publish")?;
    dataset::stamp_languages(&mut ds, &languages).await?;
    dataset::build_indexes(&mut ds, on_event).await?;

    let mut ops = Vec::new();
    for entry in walkdir::WalkDir::new(&db_dir).into_iter().filter_map(|e| e.ok()) {
        if !entry.file_type().is_file() {
            continue;
        }
        let rel = entry.path().strip_prefix(staging.path()).unwrap_or(entry.path());
        ops.push(CommitOperation::add_file(
            rel.to_string_lossy().into_owned(),
            entry.path().to_path_buf(),
        ));
    }
    if ops.is_empty() {
        return Ok(None);
    }
    let captured = extra_files
        .iter()
        .map(|(path, body)| Ok((path.clone(), CapturedFile::from_bytes(body)?)))
        .collect::<Result<BTreeMap<_, _>>>()?;
    let (extra_ops, _extra_dir) = write_ops(&captured)?;
    ops.extend(extra_ops);

    let info = send_commit_chunks(repo, ops, None, rev, message)
        .await
        .map_err(|e| anyhow::Error::new(e).context("create_commit failed"))?;
    Ok(Some(info.commit_oid.unwrap_or_else(|| "?".to_string())))
}

/// Publish a first canonical dataset in one parent-guarded Hub commit. Unlike the general push
/// bootstrap, canonical ingestion must notice a concurrent creator and re-read its revisions.
pub(crate) async fn first_document_publish(
    repo: &HFRepository<RepoTypeDataset>,
    prefix: &str,
    batches: Vec<RecordBatch>,
    schema: SchemaRef,
    expected_parent: &str,
    rev: &str,
    message: String,
) -> Result<Replaced> {
    let staging = tempfile::tempdir()?;
    let db_dir = if prefix.is_empty() {
        staging.path().to_path_buf()
    } else {
        staging.path().join(prefix)
    };
    std::fs::create_dir_all(&db_dir)?;
    let table_uri = dataset::table_uri(&db_dir);
    let reader = RecordBatchIterator::new(batches.into_iter().map(Ok), schema);
    let mut ds = Dataset::write(reader, &table_uri, Some(WriteParams::default()))
        .await
        .context("building the first canonical dataset")?;
    dataset::build_indexes(&mut ds, |_| {}).await?;

    let mut ops = Vec::new();
    for entry in walkdir::WalkDir::new(&db_dir).into_iter().filter_map(|e| e.ok()) {
        if !entry.file_type().is_file() {
            continue;
        }
        let rel = entry.path().strip_prefix(staging.path()).unwrap_or(entry.path());
        ops.push(CommitOperation::add_file(
            rel.to_string_lossy().into_owned(),
            entry.path().to_path_buf(),
        ));
    }
    ensure!(!ops.is_empty(), "canonical dataset build produced no files");
    match send_commit(repo, ops, Some(expected_parent.to_string()), rev, message).await {
        Ok(info) => Ok(Replaced::Committed(info.commit_oid.unwrap_or_else(|| "?".to_string()))),
        Err(e) if head_moved(&e) => Ok(Replaced::Conflict),
        Err(e) => Err(commit_error("canonical data commit failed", &e)),
    }
}

/// Compact the remote dataset's fragments, refresh its indexes, building any it lacks, and delete
/// its old versions, on branch `rev`. The work can take minutes, so it doesn't hold the head it
/// read: it reserves the fragment ids its compaction takes, uploads the new files, replays its Lance
/// commits onto the head as it is then, and deletes the old files.
///
/// `repo` must not retry HTTP requests: a manifest commit retried after a lost response reports a
/// conflict, and the files of the version it landed would be discarded.
pub(crate) async fn compact(
    repo: &HFRepository<RepoTypeDataset>,
    dataset_uri: &str,
    storage_options: HashMap<String, String>,
    rev: &str,
    message: String,
    on_event: impl Fn(IndexBuildEvent),
) -> Result<Compacted> {
    let head = head_oid(repo, rev).await?;
    let (mut ds, wrapper) = open_pinned(dataset_uri, &storage_options, &head).await?;
    let read = ds.version().version;

    // A push landing meanwhile takes the fragment ids past the reserved ones.
    let compacted = dataset::fragments_to_compact(&ds).await?;
    if compacted > 0 {
        let (reserving, reserved) = open_pinned(dataset_uri, &storage_options, &head).await?;
        let reserve = Operation::ReserveFragments {
            num_fragments: compacted as u32,
        };
        CommitBuilder::new(Arc::new(reserving))
            .execute(Transaction::new(read, reserve, None))
            .await
            .context("reserving the fragment ids")?;
        let (ops, _dir) = write_ops(&captured_files(&reserved))?;
        match send_commit(repo, ops, Some(head), rev, message.clone()).await {
            Ok(_) => {}
            Err(e) if head_moved(&e) => return Ok(Compacted::Conflict),
            Err(e) => return Err(anyhow::Error::new(e).context("reservation commit failed")),
        }
    }

    let txns = refresh(&mut ds, read, on_event).await?;
    let written = rewritten_fragments(&txns);
    ensure!(
        written <= compacted,
        "the compaction wrote {written} fragments for {compacted} reserved ids"
    );
    if txns.is_empty() {
        // Nothing to replay, but a cleanup that failed may have left old versions.
        let oid = clean(repo, dataset_uri, &storage_options, rev, &message).await?;
        return Ok(oid.map_or(Compacted::AlreadyCompact, Compacted::Committed));
    }
    let mut files = captured_files(&wrapper);
    files.retain(|path, _| !is_version_file(path));
    let staged: Vec<String> = files.keys().cloned().collect();
    let (ops, _dir) = write_ops(&files)?;
    if let Err(e) = commit_in_parts(repo, rev, ops, &message, None).await {
        discard(repo, rev, staged, &message).await;
        return Err(e);
    }

    let oid = match replay(repo, dataset_uri, &storage_options, rev, read, &txns, &message).await {
        Ok(Replayed::Landed(oid)) => oid,
        Ok(Replayed::Refused) => {
            discard(repo, rev, staged, &message).await;
            return Ok(Compacted::Conflict);
        }
        Ok(Replayed::Uncertain(e)) => return Err(e),
        Err(e) => {
            discard(repo, rev, staged, &message).await;
            return Err(e);
        }
    };
    match clean(repo, dataset_uri, &storage_options, rev, &message).await {
        Ok(deleted) => Ok(Compacted::Committed(deleted.unwrap_or(oid))),
        Err(e) => Ok(Compacted::Uncleaned(oid, e)),
    }
}

/// Delete the old versions at the head of `rev`. The oid of the last delete commit, if any.
async fn clean(
    repo: &HFRepository<RepoTypeDataset>,
    dataset_uri: &str,
    storage_options: &HashMap<String, String>,
    rev: &str,
    message: &str,
) -> Result<Option<String>> {
    let head = head_oid(repo, rev).await?;
    let (ds, wrapper) = open_pinned(dataset_uri, storage_options, &head).await?;
    delete_old_versions_keeping_uploads(&ds).await?;
    let deletes = captured_deletes(&wrapper, repo, &head).await?;
    if !deletes.is_empty() {
        eprintln!("deleting {} old files…", deletes.len());
    }
    commit_in_parts(repo, rev, deletes, message, Some(head)).await
}

/// Compact a remote dataset and refresh its indexes, returning the commits made after version
/// `read`. Compacting first keeps the rewritten fragments outside every index.
async fn refresh(ds: &mut Dataset, read: u64, on_event: impl Fn(IndexBuildEvent)) -> Result<Vec<Transaction>> {
    dataset::compact_fragments(ds, &on_event).await?;
    dataset::build_indexes(ds, on_event)
        .await
        .context("refreshing the remote indexes")?;
    transactions_since(ds, read).await
}

/// Delete the old versions and the files only they reference. A file no version references is kept
/// for 7 days, as another host may have uploaded it ahead of its manifest.
async fn delete_old_versions_keeping_uploads(ds: &Dataset) -> Result<()> {
    ds.cleanup_old_versions(chrono::Duration::zero(), Some(false), None)
        .await
        .context("deleting the old versions")?;
    Ok(())
}

/// Open the remote dataset at the commit `sha`, so a session sees one head however long it runs,
/// with a [`CaptureStore`] installed. Reads are whole-file downloads: the Hub can cut short a range
/// covering a whole file.
async fn open_pinned(
    dataset_uri: &str,
    storage_options: &HashMap<String, String>,
    sha: &str,
) -> Result<(Dataset, Arc<CaptureWrapper>)> {
    let (owner, name, _) = hub::parse_hf(dataset_uri)?;
    // A client of its own: reads may retry, commits must not.
    let token = storage_options.get("hf_token").map(String::as_str);
    let repo = Arc::new(hub::client(token, true)?.dataset(owner, name));
    let fetch = Arc::new(FetchWrapper::new(repo, sha.to_string()));
    let mut options = storage_options.clone();
    options.insert("revision".to_string(), sha.to_string());
    options.insert("hf_revision".to_string(), sha.to_string());
    // Reads must be wrapped before load so manifest/index requests use the pinned cache.
    let wrapper = Arc::new(CaptureWrapper {
        captured: Captured::default(),
    });
    let ds = dataset::open_wrapped(dataset_uri, options, capturing_fetch_wrapper(fetch, wrapper.clone())).await?;
    Ok((ds, wrapper))
}

/// The commits made after version `read`, but for the fragment reservation: the compaction makes
/// its own on the Hub.
async fn transactions_since(ds: &Dataset, read: u64) -> Result<Vec<Transaction>> {
    let mut txns = Vec::new();
    for version in read + 1..=ds.version().version {
        let txn = ds
            .read_transaction_by_version(version)
            .await?
            .context("a version without its transaction")?;
        if !matches!(txn.operation, Operation::ReserveFragments { .. }) {
            txns.push(txn);
        }
    }
    Ok(txns)
}

/// The fragments `txns` write in place of others.
fn rewritten_fragments(txns: &[Transaction]) -> usize {
    txns.iter()
        .map(|txn| match &txn.operation {
            Operation::Rewrite { groups, .. } => groups.iter().map(|group| group.new_fragments.len()).sum(),
            _ => 0,
        })
        .sum()
}

/// A manifest or a transaction: what a replay writes anew.
fn is_version_file(path: &str) -> bool {
    path.contains("/_versions/") || path.contains("/_transactions/")
}

/// How a [`replay`] ended.
enum Replayed {
    /// The manifest commit landed, with its oid.
    Landed(String),
    /// [`replay_onto`] refused, or pushes kept landing past [`MAX_REPLAYS`].
    Refused,
    /// The manifest commit failed, and may have landed anyway.
    Uncertain(anyhow::Error),
}

/// Replays tried while pushes keep landing between a head read and the commit.
const MAX_REPLAYS: usize = 10;

/// Commit `txns` onto the head of `rev` as it is now.
async fn replay(
    repo: &HFRepository<RepoTypeDataset>,
    dataset_uri: &str,
    storage_options: &HashMap<String, String>,
    rev: &str,
    read: u64,
    txns: &[Transaction],
    message: &str,
) -> Result<Replayed> {
    for _ in 0..MAX_REPLAYS {
        let head = head_oid(repo, rev).await?;
        let (ds, wrapper) = open_pinned(dataset_uri, storage_options, &head).await?;
        if replay_onto(ds, read, txns).await?.is_none() {
            return Ok(Replayed::Refused);
        }
        let (ops, _dir) = write_ops(&captured_files(&wrapper))?;
        match send_commit(repo, ops, Some(head), rev, message.to_string()).await {
            Ok(info) => return Ok(Replayed::Landed(info.commit_oid.unwrap_or_else(|| "?".to_string()))),
            Err(e) if head_moved(&e) => continue,
            Err(e) => {
                return Ok(Replayed::Uncertain(
                    anyhow::Error::new(e).context("compaction commit failed"),
                ))
            }
        }
    }
    Ok(Replayed::Refused)
}

/// `txns` committed onto `ds`, or `None` if a commit other than a push or a fragment reservation
/// landed after version `read`. Lance takes a compaction or an index refresh after a push.
async fn replay_onto(mut ds: Dataset, read: u64, txns: &[Transaction]) -> Result<Option<Dataset>> {
    for version in read + 1..=ds.version().version {
        let landed = match ds.read_transaction_by_version(version).await {
            Ok(landed) => landed,
            // Only a compaction deletes versions, once its own landed.
            Err(e) if matches!(e, LanceError::DatasetNotFound { .. }) || e.is_not_found() => return Ok(None),
            Err(e) => return Err(e.into()),
        };
        if !matches!(
            landed.map(|txn| txn.operation),
            Some(Operation::Append { .. } | Operation::UpdateConfig { .. } | Operation::ReserveFragments { .. })
        ) {
            return Ok(None);
        }
    }
    for txn in txns {
        let mut txn = txn.clone();
        txn.read_version = ds.version().version;
        ds = CommitBuilder::new(Arc::new(ds))
            .execute(txn)
            .await
            .context("replaying the compaction")?;
    }
    Ok(Some(ds))
}

/// Delete those of the `staged` files that reached the branch, which no version references yet. Asks
/// the Hub which did, since a commit that failed may still have landed. Best-effort: the next
/// cleanup takes any left once they are 7 days old.
async fn discard(repo: &HFRepository<RepoTypeDataset>, rev: &str, staged: Vec<String>, message: &str) {
    let mut deletes = Vec::new();
    for part in staged.chunks(MAX_COMMIT_OPS) {
        let Ok(landed) = repo
            .get_paths_info()
            .paths(part.to_vec())
            .revision(rev.to_string())
            .send()
            .await
        else {
            return;
        };
        deletes.extend(landed.into_iter().filter_map(|entry| match entry {
            RepoTreeEntry::File { path, .. } => Some(CommitOperation::delete(path)),
            RepoTreeEntry::Directory { .. } => None,
        }));
    }
    let _ = commit_in_parts(repo, rev, deletes, message, None).await;
}

/// Operations per Hub commit: the Hub checks each within a 60 s request timeout and advises 50 to
/// 100.
const MAX_COMMIT_OPS: usize = 100;

/// Commit `ops` in parts of at most [`MAX_COMMIT_OPS`], unguarded: new files and deletes of
/// unreferenced ones hold whatever landed meanwhile. The oid of the last part.
/// Cleanup supplies its pinned parent instead: a changed head must invalidate the delete plan.
async fn commit_in_parts(
    repo: &HFRepository<RepoTypeDataset>,
    rev: &str,
    ops: Vec<CommitOperation>,
    message: &str,
    mut parent: Option<String>,
) -> Result<Option<String>> {
    let mut oid = None;
    let mut ops = ops.into_iter().peekable();
    while ops.peek().is_some() {
        let part = ops.by_ref().take(MAX_COMMIT_OPS).collect();
        let info = send_commit(repo, part, parent.clone(), rev, message.to_string())
            .await
            .map_err(|e| anyhow::Error::new(e).context("compaction commit failed"))?;
        if parent.is_some() {
            parent = Some(
                info.commit_oid
                    .as_ref()
                    .filter(|oid| !oid.is_empty())
                    .context("cleanup commit returned no oid; refusing to continue its delete plan")?
                    .clone(),
            );
        }
        oid = info.commit_oid;
    }
    Ok(oid)
}

/// Refresh indexes in a guarded commit; do not compact data fragments or delete versions.
pub(crate) async fn reindex(
    repo: &HFRepository<RepoTypeDataset>,
    dataset_uri: &str,
    storage_options: HashMap<String, String>,
    rev: &str,
    message: String,
) -> Result<Reindexed> {
    let parent = head_oid(repo, rev).await?;
    let (mut ds, wrapper) = open_pinned(dataset_uri, &storage_options, &parent).await?;
    dataset::build_indexes(&mut ds, |_| {}).await?;
    let files = captured_files(&wrapper);
    if files.is_empty() {
        return Ok(Reindexed::AlreadyCurrent);
    }
    let (ops, _dir) = write_ops(&files)?;
    match send_commit(repo, ops, Some(parent), rev, message).await {
        Ok(info) => Ok(Reindexed::Committed(info.commit_oid.unwrap_or_else(|| "?".into()))),
        Err(error) if head_moved(&error) => Ok(Reindexed::Conflict),
        Err(error) => Err(commit_error("reindex commit failed", &error)),
    }
}

/// Rename a column on the remote dataset in one head-guarded commit. `alter_columns` is
/// metadata-only (the captured writes are a new manifest and transaction, no data files), so the
/// commit is small whatever the memory's size. Returns the new commit oid. A moved head is an
/// error, not a retry: a rename is an exclusive-writer operation.
pub async fn rename_column(
    repo: &HFRepository<RepoTypeDataset>,
    dataset_uri: &str,
    storage_options: HashMap<String, String>,
    rev: &str,
    message: String,
    from: &str,
    to: &str,
) -> Result<String> {
    let parent = head_oid(repo, rev).await?;
    let (mut ds, wrapper) = open_pinned(dataset_uri, &storage_options, &parent).await?;
    ds.alter_columns(&[ColumnAlteration::new(from.into()).rename(to.into())])
        .await
        .context("renaming the remote column")?;
    let files = captured_files(&wrapper);
    ensure!(!files.is_empty(), "the rename produced no files to commit");
    let (ops, _dir) = write_ops(&files)?;
    let info = send_commit(repo, ops, Some(parent), rev, message)
        .await
        .map_err(|e| anyhow::Error::new(e).context("rename commit failed"))?;
    Ok(info.commit_oid.unwrap_or_else(|| "?".to_string()))
}

/// Add a column to the remote dataset via `add_columns`, landing the new column's files in one
/// head-guarded commit. `transform` produces the new column per batch (a UDF over `read_columns`).
/// Writes real per-fragment column data, shipped as one captured commit; data, vectors, and
/// indexes are untouched. Returns the new oid. A moved head is an error — a single guarded
/// attempt, not retried.
pub async fn add_column(
    repo: &HFRepository<RepoTypeDataset>,
    dataset_uri: &str,
    storage_options: HashMap<String, String>,
    rev: &str,
    message: String,
    transform: NewColumnTransform,
    read_columns: Vec<String>,
) -> Result<String> {
    let parent = head_oid(repo, rev).await?;
    let (mut ds, wrapper) = open_pinned(dataset_uri, &storage_options, &parent).await?;
    ds.add_columns(transform, Some(read_columns), None)
        .await
        .context("adding the remote column")?;
    let files = captured_files(&wrapper);
    ensure!(!files.is_empty(), "add_columns produced no files to commit");
    let (ops, _dir) = write_ops(&files)?;
    let info = send_commit(repo, ops, Some(parent), rev, message)
        .await
        .map_err(|e| commit_error("add_column commit failed", &e))?;
    Ok(info.commit_oid.unwrap_or_else(|| "?".to_string()))
}

/// Count the remote dataset's languages over all of its rows and record them in one head-guarded
/// commit, returning the counts and the new oid. A moved head is an error: a single guarded
/// attempt, not retried.
pub async fn count_languages(
    repo: &HFRepository<RepoTypeDataset>,
    dataset_uri: &str,
    storage_options: HashMap<String, String>,
    rev: &str,
    message: String,
) -> Result<(dataset::Languages, String)> {
    let parent = head_oid(repo, rev).await?;
    let (mut ds, wrapper) = open_pinned(dataset_uri, &storage_options, &parent).await?;
    let languages = dataset::scan_languages(&ds).await?;
    dataset::stamp_languages(&mut ds, &languages).await?;
    let (ops, _dir) = write_ops(&captured_files(&wrapper))?;
    let info = send_commit(repo, ops, Some(parent), rev, message)
        .await
        .map_err(|e| anyhow::Error::new(e).context("language count commit failed"))?;
    Ok((languages, info.commit_oid.unwrap_or_else(|| "?".to_string())))
}

/// Atomically replace every split belonging to selected canonical sources. The indexed delete and
/// append run through the same capture store and land in one guarded Hub commit.
#[allow(clippy::too_many_arguments)] // Keep the selected snapshot and CAS target explicit at this write boundary.
pub(crate) async fn replace_documents(
    _repo: &HFRepository<RepoTypeDataset>,
    commit_repo: &HFRepository<RepoTypeDataset>,
    dataset_uri: &str,
    storage_options: HashMap<String, String>,
    expected_parent: &str,
    rev: &str,
    message: String,
    batches: Vec<RecordBatch>,
    delete_filter: &str,
) -> Result<Replaced> {
    // Run Lance against the exact immutable head whose source revisions the caller selected.
    // Committing against anything newer would turn a selection/write race into a stale overwrite.
    let (mut ds, wrapper) = open_pinned(dataset_uri, &storage_options, expected_parent).await?;
    for batch in &batches {
        same_model(&ds, batch.schema().as_ref())?;
    }
    dataset::ensure_canonical_columns(&mut ds).await?;
    replace_dataset_rows(&mut ds, batches, delete_filter).await?;

    let files = captured_files(&wrapper);
    ensure!(!files.is_empty(), "canonical replace produced no files to commit");
    let (ops, _dir) = write_ops(&files)?;
    match send_commit(commit_repo, ops, Some(expected_parent.to_string()), rev, message).await {
        Ok(info) => Ok(Replaced::Committed(info.commit_oid.unwrap_or_else(|| "?".to_string()))),
        Err(e) if head_moved(&e) => Ok(Replaced::Conflict),
        Err(e) => Err(commit_error("canonical data commit failed", &e)),
    }
}

async fn replace_dataset_rows(ds: &mut Dataset, batches: Vec<RecordBatch>, delete_filter: &str) -> Result<()> {
    let delete_start = Instant::now();
    ds.delete(delete_filter)
        .await
        .context("deleting stale canonical document rows")?;
    emit_phase_metric("lance_delete", delete_start.elapsed());
    let target_schema = Arc::new(Schema::from(ds.schema()));
    let batches = batches
        .into_iter()
        .map(|batch| align_batch(batch, target_schema.clone()))
        .collect::<Result<Vec<_>>>()?;
    let reader = RecordBatchIterator::new(batches.into_iter().map(Ok), target_schema);
    let append_start = Instant::now();
    ds.append(reader, None)
        .await
        .context("appending replacement canonical document rows")?;
    emit_phase_metric("lance_append", append_start.elapsed());
    Ok(())
}

/// Compose the write path as `CaptureStore(FetchStore(remote))`: reads use the pinned local
/// cache, while writes stop at the capture and never reach the Hub object store directly.
fn capturing_fetch_wrapper(fetch: Arc<FetchWrapper>, capture: Arc<CaptureWrapper>) -> Arc<dyn WrappingObjectStore> {
    Arc::new(ChainedWrappingObjectStore::new(vec![fetch, capture]))
}

/// The captured writes as physical repo-path → temporary file, ready to commit. Existing flat files are
/// unchanged; new Lance files live in deterministic buckets so no directory hits Hub's file cap.
fn captured_files(wrapper: &CaptureWrapper) -> BTreeMap<String, CapturedFile> {
    let start = Instant::now();
    let files = wrapper
        .captured
        .lock()
        .unwrap()
        .iter()
        .filter_map(|(p, b)| {
            b.as_ref().map(|b| {
                let logical = p.to_string();
                (physical_path(&logical).unwrap_or(logical), b.clone())
            })
        })
        .collect();
    emit_phase_metric("captured_files", start.elapsed());
    files
}

/// The captured deletes as commit operations: the existing files Lance removed. The Hub lists
/// folders as entries too, but rejects a commit deleting one, so a delete with others under it is
/// left out: the folder goes with its last file.
/// Resolve both layouts at the read snapshot; guessing one path can leave a shadowed copy behind.
async fn captured_deletes(
    wrapper: &CaptureWrapper,
    repo: &HFRepository<RepoTypeDataset>,
    head: &str,
) -> Result<Vec<CommitOperation>> {
    let deleted: BTreeSet<String> = wrapper
        .captured
        .lock()
        .unwrap()
        .iter()
        .filter(|(_, b)| b.is_none())
        .map(|(p, _)| p.to_string())
        .collect();
    let candidates: BTreeSet<String> = deleted
        .iter()
        .filter(|path| {
            let folder = format!("{path}/");
            !deleted
                .range(folder.clone()..)
                .next()
                .is_some_and(|next| next.starts_with(&folder))
        })
        .flat_map(|path| std::iter::once(path.clone()).chain(physical_path(path)))
        .collect();
    let candidates: Vec<String> = candidates.into_iter().collect();
    let mut files = BTreeSet::new();
    for part in candidates.chunks(MAX_COMMIT_OPS) {
        let entries = repo
            .get_paths_info()
            .paths(part.to_vec())
            .revision(head.to_string())
            .send()
            .await
            .context("resolving cleanup files at the pinned revision")?;
        for entry in entries {
            if let RepoTreeEntry::File { path, .. } = entry {
                ensure!(part.contains(&path), "cleanup response included an unrequested file");
                files.insert(path);
            }
        }
    }
    Ok(files.into_iter().map(CommitOperation::delete).collect())
}

/// Cap on CAS-conflict retries (the data append, and a forced compaction) when the branch head
/// keeps moving under us, so a busy remote can't spin forever.
pub(crate) const MAX_COMMIT_RETRIES: u32 = 10;

/// A remote memory opened for a write: its repo handle, with no HTTP retries as [`compact`]
/// requires, the branch reads pin to, its table's URI and the storage options a Lance open of it
/// takes.
pub(crate) struct Writable {
    pub(crate) repo: HFRepository<RepoTypeDataset>,
    /// `<owner>/<name>`.
    pub(crate) repo_id: String,
    /// Where the memory sits inside the repo; empty at its root.
    pub(crate) prefix: String,
    pub(crate) rev: String,
    pub(crate) dataset_uri: String,
    pub(crate) opts: HashMap<String, String>,
}

/// The remote memory at `uri`, opened for a write. Writing needs a Hugging Face token.
pub(crate) fn writable(uri: &str) -> Result<Writable> {
    let (owner, name, prefix) = hub::parse_hf(uri)?;
    let token = hub::hf_token().with_context(|| format!("no HF token (set HF_TOKEN) — writing to {uri} needs one"))?;
    let repo_id = format!("{owner}/{name}");
    let repo = hub::client(Some(token.as_str()), false)?.dataset(owner, name);
    let rev = super::READ_BRANCH.to_string();
    let opts = HashMap::from([("hf_token".to_string(), token), ("revision".to_string(), rev.clone())]);
    Ok(Writable {
        repo,
        repo_id,
        prefix,
        rev,
        dataset_uri: dataset::table_uri(uri),
        opts,
    })
}

/// Read the commit at the tip of branch `rev` — the parent-commit guard for the next commit.
pub(crate) async fn head_oid(repo: &HFRepository<RepoTypeDataset>, rev: &str) -> Result<String> {
    let refs = repo.list_refs().send().await.context("listing remote refs")?;
    refs.branches
        .iter()
        .find(|b| b.name == rev)
        .map(|b| b.target_commit.clone())
        .context("target branch not found on the remote")
}

/// Turn captured files into add-file commit operations — hf-hub uploads directly from local paths.
/// The returned file guards must outlive all chunks, including retries after the capture is dropped.
fn write_ops(files: &BTreeMap<String, CapturedFile>) -> Result<(Vec<CommitOperation>, Vec<CapturedFile>)> {
    let start = Instant::now();
    let mut ops = Vec::with_capacity(files.len());
    for (repo_path, body) in files {
        ensure!(
            std::fs::metadata(body.path())?.len() == body.len(),
            "captured file changed before upload"
        );
        ops.push(CommitOperation::add_file(repo_path.clone(), body.path().to_path_buf()));
    }
    emit_phase_metric("write_ops", start.elapsed());
    Ok((ops, files.values().cloned().collect()))
}

/// The repo's `README.md` at `rev`, or `None` when it has none — fetched straight to bytes,
/// never the shared cache, so a push always classifies the dataset card against the branch
/// head it targets.
pub(crate) async fn fetch_readme(repo: &HFRepository<RepoTypeDataset>, rev: &str) -> Result<Option<String>> {
    let fetched = repo
        .download_file_to_bytes()
        .filename("README.md")
        .revision(rev.to_string())
        .send()
        .await;
    match fetched {
        Ok(bytes) => Ok(Some(String::from_utf8_lossy(&bytes).into_owned())),
        Err(HFError::EntryNotFound { .. }) => Ok(None),
        Err(e) => Err(anyhow::Error::new(e).context("reading the remote dataset card")),
    }
}

/// How many times a commit the Hub refused as busy is sent again, waiting 1, 2, 4… seconds.
const BUSY_RESENDS: u32 = 5;

/// One `create_commit` of `ops` on branch `rev`, guarded by `parent` when there is one. Returns the
/// raw hf-hub result so callers can tell a head-moved [`HFError::Conflict`] from other failures.
/// A commit the Hub refuses as busy did not land, so it is sent again after a wait.
async fn send_commit_once(
    repo: &HFRepository<RepoTypeDataset>,
    ops: Vec<CommitOperation>,
    parent: Option<String>,
    rev: &str,
    message: String,
) -> std::result::Result<CommitInfo, HFError> {
    let deletes = ops
        .iter()
        .filter(|op| matches!(op, CommitOperation::Delete { .. }))
        .count();
    let mut resends = 0;
    loop {
        let sent = repo
            .create_commit()
            .operations(ops.clone())
            .commit_message(message.clone())
            .maybe_parent_commit(parent.clone())
            .revision(rev.to_string())
            .progress(upload_progress(deletes))
            .send()
            .await;
        match sent {
            Err(e) if repo_busy(&e) && resends < BUSY_RESENDS => {
                tokio::time::sleep(Duration::from_secs(1 << resends)).await;
                resends += 1;
            }
            result => return result,
        }
    }
}

/// Whether the Hub refused a commit because another is in progress on the repo, in its own words.
fn repo_busy(e: &HFError) -> bool {
    matches!(e, HFError::Conflict { context } if context.body.contains("Another commit operation is in progress"))
}

/// Upload `ops` in bounded chained `create_commit` calls on branch `rev`, guarded by `parent`.
/// The returned result is the final Hub response so callers can tell a head-moved
/// [`HFError::Conflict`] from other failures. Activation metadata is forced into the last chunk.
async fn send_commit(
    repo: &HFRepository<RepoTypeDataset>,
    ops: Vec<CommitOperation>,
    parent: Option<String>,
    rev: &str,
    message: String,
) -> std::result::Result<CommitInfo, HFError> {
    send_commit_chunks(repo, ops, parent, rev, message).await
}

/// Hub currently rejects a single commit containing more than 1000 files. Keep headroom for
/// retries and future metadata by using a lower bound, and make activation metadata the final
/// chunk so an interrupted upload cannot expose a manifest that references missing files.
const MAX_COMMIT_OPERATIONS: usize = 900;

const MAX_COMMIT_ATTEMPTS: u32 = 6;
// Shared across chunks; bounds retry waits, not in-flight request time. Never shorten Retry-After.
const COMMIT_RETRY_BUDGET: Duration = Duration::from_secs(900);

#[derive(Debug, Serialize)]
struct HfCommitMetric {
    stage: &'static str,
    duration_ms: f64,
    attempt: u32,
    status_code: u16,
    phase: &'static str,
    retry_after_ms: Option<f64>,
    backoff_ms: f64,
    retrying: bool,
    limit_kind: &'static str,
}

fn hf_limit_kind(server_message: Option<&str>, body: &str) -> &'static str {
    let reason = format!("{} {}", server_message.unwrap_or(""), body).to_ascii_lowercase();
    if ["too many commits", "commit rate limit", "commits per"]
        .iter()
        .any(|hint| reason.contains(hint))
    {
        "commit_action"
    } else if reason.contains("api rate limit") {
        "api"
    } else if reason.contains("concurrent") && reason.contains("limit") {
        "concurrency"
    } else {
        "unknown"
    }
}

fn commit_retry_wait(
    error: &HFError,
    attempt: u32,
    elapsed: Duration,
    budget: Duration,
    guarded: bool,
) -> Option<Duration> {
    if !guarded || attempt >= MAX_COMMIT_ATTEMPTS {
        return None;
    }
    let wait = match error {
        HFError::RateLimited { retry_after, .. } => {
            retry_after.unwrap_or_else(|| Duration::from_secs((30u64 << attempt.saturating_sub(1).min(4)).min(300)))
        }
        // The canonical writer disables hf-hub's retries; retain bounded transient retries.
        HFError::Http { context } if matches!(context.status.as_u16(), 408 | 500 | 502 | 503 | 504) => {
            Duration::from_millis(200u64 << attempt.saturating_sub(1).min(4))
        }
        // Request-phase transport failures include resets/incomplete responses. Parent CAS makes
        // retries safe even when the server accepted a request before the connection disappeared.
        HFError::Request { source: error, .. } if error.is_connect() || error.is_timeout() || error.is_request() => {
            Duration::from_millis(200u64 << attempt.saturating_sub(1).min(4))
        }
        _ => return None,
    };
    (elapsed.checked_add(wait)? < budget).then_some(wait)
}

/// Keep captured files and vectors alive while a guarded commit is rate-limited. A 409/412
/// is never retried here: the caller must reopen the head and recheck every source revision.
async fn retry_commit<F, Fut>(
    guarded: bool,
    started: Instant,
    budget: Duration,
    mut send: F,
) -> std::result::Result<CommitInfo, HFError>
where
    F: FnMut() -> Fut,
    Fut: Future<Output = std::result::Result<CommitInfo, HFError>>,
{
    for attempt in 1..=MAX_COMMIT_ATTEMPTS {
        let attempt_started = Instant::now();
        let result = send().await;
        let wait = result
            .as_ref()
            .err()
            .and_then(|error| commit_retry_wait(error, attempt, started.elapsed(), budget, guarded));
        if ingest_metrics_enabled() {
            let (status_code, phase, retry_after_ms, limit_kind) = match &result {
                Ok(_) => (200, "commit", None, "unknown"),
                Err(HFError::RateLimited { retry_after, context }) => (
                    context.status.as_u16(),
                    hf_http_phase(&context.url),
                    retry_after.map(round_ms),
                    hf_limit_kind(context.server_message.as_deref(), &context.body),
                ),
                Err(
                    HFError::Http { context }
                    | HFError::Conflict { context }
                    | HFError::AuthRequired { context }
                    | HFError::Forbidden { context },
                ) => (context.status.as_u16(), hf_http_phase(&context.url), None, "unknown"),
                _ => (0, "unknown", None, "unknown"),
            };
            let metric = HfCommitMetric {
                stage: "hf_commit_attempt",
                duration_ms: round_ms(attempt_started.elapsed()),
                attempt,
                status_code,
                phase,
                retry_after_ms,
                backoff_ms: wait.map(round_ms).unwrap_or(0.0),
                retrying: wait.is_some(),
                limit_kind,
            };
            if let Ok(json) = serde_json::to_string(&metric) {
                let _ = write_metric(&mut std::io::stderr().lock(), &json);
            }
        }
        match wait {
            Some(wait) => tokio::time::sleep(wait).await,
            None => return result,
        }
    }
    unreachable!("last attempt never retries")
}

/// Paths that activate a new Lance snapshot. The final chunk may also contain `README.md`, which
/// is updated with the same dataset snapshot and should not precede the manifest activation.
fn is_activation_path(path: &str) -> bool {
    let path = path.trim_start_matches("./");
    path == "README.md"
        || path.starts_with("_versions/")
        || path.contains("/_versions/")
        || path.starts_with("_transactions/")
        || path.contains("/_transactions/")
        || path.ends_with(".manifest")
        || path.ends_with("latest_version_hint.json")
}

fn operation_path(operation: &CommitOperation) -> &str {
    match operation {
        CommitOperation::Add { path_in_repo, .. } | CommitOperation::Delete { path_in_repo } => path_in_repo,
    }
}

/// Split a commit before any network upload. Activation metadata is deliberately rejected when it
/// cannot fit in one final chunk; uploading part of it would make the ordering guarantee false.
fn split_commit_operations(ops: Vec<CommitOperation>) -> std::result::Result<Vec<Vec<CommitOperation>>, HFError> {
    let mut regular = Vec::with_capacity(ops.len());
    let mut activation = Vec::new();
    for operation in ops {
        if is_activation_path(operation_path(&operation)) {
            activation.push(operation);
        } else {
            regular.push(operation);
        }
    }
    if activation.len() > MAX_COMMIT_OPERATIONS {
        return Err(HFError::InvalidParameter(format!(
            "activation metadata exceeds one commit chunk: {} operations (limit {})",
            activation.len(),
            MAX_COMMIT_OPERATIONS
        )));
    }

    let mut chunks = regular
        .chunks(MAX_COMMIT_OPERATIONS)
        .map(|chunk| chunk.to_vec())
        .collect::<Vec<_>>();
    if !activation.is_empty() {
        if let Some(last) = chunks.last_mut() {
            if last.len() + activation.len() <= MAX_COMMIT_OPERATIONS {
                last.extend(activation);
            } else {
                chunks.push(activation);
            }
        } else {
            chunks.push(activation);
        }
    }
    Ok(chunks)
}

/// Upload chunks serially. The first chunk uses the caller's expected parent (if any); every
/// later chunk uses the successful preceding commit SHA. A missing SHA is handled by reading the
/// branch head rather than passing an empty parent to the next request.
async fn send_commit_chunks(
    repo: &HFRepository<RepoTypeDataset>,
    ops: Vec<CommitOperation>,
    parent: Option<String>,
    rev: &str,
    message: String,
) -> std::result::Result<CommitInfo, HFError> {
    let chunks = split_commit_operations(ops)?;
    let mut parent = parent;
    let mut final_info = None;
    let started = Instant::now();
    for (index, chunk) in chunks.into_iter().enumerate() {
        let chunk_message = if index == 0 {
            message.clone()
        } else {
            format!("{message} (chunk {index})")
        };
        let chunk_start = Instant::now();
        let info = retry_commit(parent.is_some(), started, COMMIT_RETRY_BUDGET, || {
            send_commit_once(repo, chunk.clone(), parent.clone(), rev, chunk_message.clone())
                // Erase the upload future's deep type so every caller stays within rustc's limit.
                .boxed()
        })
        .await?;
        emit_phase_metric("hf_commit_chunk", chunk_start.elapsed());
        parent = match info.commit_oid.as_deref().filter(|oid| !oid.is_empty()) {
            Some(oid) => Some(oid.to_string()),
            None => {
                let wait_start = Instant::now();
                let head = head_oid_for_commit(repo, rev).await?;
                emit_phase_metric("hf_commit_wait", wait_start.elapsed());
                Some(head)
            }
        };
        final_info = Some(info);
    }
    final_info.ok_or_else(|| HFError::InvalidParameter("cannot commit an empty operation list".to_string()))
}

/// Same branch-head lookup as [`head_oid`], retaining the HF error type used by chunk uploads and
/// never copying an arbitrary response body into a persisted diagnostic.
async fn head_oid_for_commit(repo: &HFRepository<RepoTypeDataset>, rev: &str) -> std::result::Result<String, HFError> {
    let refs = repo.list_refs().send().await?;
    refs.branches
        .iter()
        .find(|branch| branch.name == rev)
        .map(|branch| branch.target_commit.clone())
        .ok_or_else(|| HFError::InvalidParameter("target branch not found on the remote".to_string()))
}

/// Keep enough of a Hub rejection to identify its class without persisting URLs,
/// repository paths, credentials, or an unbounded response body.
fn safe_server_reason(value: &str) -> Option<String> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        return None;
    }
    let lower = trimmed.to_ascii_lowercase();
    let classified = if lower.contains("larger than 10 mib")
        || lower.contains("larger than 10 mb")
        || lower.contains("file size")
        || lower.contains("git-lfs")
        || lower.contains("git lfs")
    {
        Some("file_size_or_lfs")
    } else if lower.contains("more than 1000 files") || lower.contains("too many files") || lower.contains("file limit")
    {
        Some("file_count_limit")
    } else if lower.contains("lfs pointer") || lower.contains("lfs object") {
        Some("missing_lfs_object")
    } else if lower.contains("parent commit") || lower.contains("parentcommit") {
        Some("invalid_parent")
    } else if lower.contains("empty commit") || lower.contains("no operations") || lower.contains("commit content") {
        Some("empty_commit")
    } else if lower.contains("invalid path") || lower.contains("path is invalid") {
        Some("invalid_path")
    } else if lower.contains("rate limit") || lower.contains("too many requests") {
        Some("rate_limit")
    } else if lower.contains("unauthorized") || lower.contains("forbidden") || lower.contains("permission") {
        Some("auth_or_permission")
    } else if lower.contains("conflict") || lower.contains("head moved") {
        Some("stale_parent")
    } else {
        None
    };
    if let Some(reason) = classified {
        return Some(reason.to_string());
    }

    let sanitized = trimmed
        .split_whitespace()
        .map(|word| {
            let lower_word = word.to_ascii_lowercase();
            if word.contains("://") {
                "<url>".to_string()
            } else if word.starts_with('/') || word.contains('\\') {
                "<path>".to_string()
            } else if lower_word.contains("bearer")
                || lower_word.starts_with("token=")
                || lower_word.starts_with("api_key=")
            {
                "<redacted>".to_string()
            } else {
                word.chars()
                    .filter(|c| c.is_ascii_alphanumeric() || matches!(c, ' ' | '.' | ',' | ':' | ';' | '-' | '_'))
                    .collect::<String>()
            }
        })
        .filter(|word| !word.is_empty())
        .collect::<Vec<_>>()
        .join(" ");
    let bounded = sanitized.chars().take(160).collect::<String>();
    (!bounded.is_empty()).then_some(bounded)
}

/// Convert an HF commit failure into a bounded diagnostic that is safe to persist in the
/// reconciler state. `HFError`'s Display implementation intentionally includes response URLs and,
/// for conflicts, the full response body; neither belongs in logs or API responses.
fn commit_error(label: &str, error: &HFError) -> anyhow::Error {
    fn safe_atom(value: &str) -> Option<String> {
        let value = value.trim();
        if value.is_empty()
            || value.contains("://")
            || value.contains('/')
            || value.contains('\\')
            || value.to_ascii_lowercase().contains("bearer")
            || value.to_ascii_lowercase().contains("token")
        {
            return None;
        }
        let sanitized: String = value
            .chars()
            .filter(|c| c.is_ascii_alphanumeric() || matches!(c, ' ' | '.' | ',' | ':' | ';' | '-' | '_'))
            .take(160)
            .collect();
        let sanitized = sanitized.split_whitespace().collect::<Vec<_>>().join(" ");
        (!sanitized.is_empty()).then_some(sanitized)
    }

    macro_rules! http_detail {
        ($variant:expr, $context:expr) => {{
            let context = $context;
            let phase = hf_http_phase(&context.url);
            let mut detail = format!(
                "{label}: hf_error={} http_status={} hf_phase={phase}",
                $variant,
                context.status.as_u16()
            );
            if let Some(code) = context.error_code.as_deref().and_then(safe_atom) {
                detail.push_str(&format!(" error_code={code}"));
            }
            let server_reason = context
                .server_message
                .as_deref()
                .and_then(safe_server_reason)
                .or_else(|| safe_server_reason(&context.body));
            if let Some(message) = server_reason {
                detail.push_str(&format!(" server_reason={message}"));
            }
            anyhow::anyhow!(detail)
        }};
    }

    match error {
        HFError::Http { context } => http_detail!("http", context),
        HFError::AuthRequired { context } => http_detail!("auth_required", context),
        HFError::Forbidden { context } => http_detail!("forbidden", context),
        HFError::RateLimited { context, .. } => http_detail!("rate_limited", context),
        HFError::Conflict { context } => http_detail!("conflict", context),
        HFError::Xet { operation, .. } => anyhow::anyhow!("{label}: hf_error=xet operation={operation}"),
        HFError::Request { .. } => anyhow::anyhow!("{label}: hf_error=request_transport"),
        HFError::RepoNotFound { .. } => anyhow::anyhow!("{label}: hf_error=repo_not_found"),
        HFError::RevisionNotFound { .. } => anyhow::anyhow!("{label}: hf_error=revision_not_found"),
        HFError::EntryNotFound { .. } => anyhow::anyhow!("{label}: hf_error=entry_not_found"),
        HFError::BucketNotFound { .. } => anyhow::anyhow!("{label}: hf_error=bucket_not_found"),
        HFError::LocalEntryNotFound { .. } => {
            anyhow::anyhow!("{label}: hf_error=local_entry_not_found")
        }
        HFError::CacheNotEnabled => anyhow::anyhow!("{label}: hf_error=cache_not_enabled"),
        HFError::CacheLockTimeout { .. } => {
            anyhow::anyhow!("{label}: hf_error=cache_lock_timeout")
        }
        HFError::Io(_) => anyhow::anyhow!("{label}: hf_error=io"),
        HFError::Json(_) => anyhow::anyhow!("{label}: hf_error=json"),
        HFError::Url(_) => anyhow::anyhow!("{label}: hf_error=url"),
        HFError::InvalidParameter(_) => anyhow::anyhow!("{label}: hf_error=invalid_parameter"),
        HFError::DiffParse(_) => anyhow::anyhow!("{label}: hf_error=diff_parse"),
        HFError::MalformedResponse { .. } => {
            anyhow::anyhow!("{label}: hf_error=malformed_response")
        }
        HFError::Other(_) => anyhow::anyhow!("{label}: hf_error=other"),
        _ => anyhow::anyhow!("{label}: hf_error=unknown"),
    }
}

/// Bounded phase label derived from the Hub endpoint only; never persist the URL itself.
fn hf_http_phase(url: &str) -> &'static str {
    if url.contains("/preupload/") {
        "preupload"
    } else if url.contains("/commit/") {
        "commit"
    } else {
        "unknown"
    }
}

/// Whether a [`send_commit`] failure is the Hub rejecting a stale `parent_commit`: the commit API
/// answers a moved head with 412 Precondition Failed, which hf-hub leaves as a generic
/// [`HFError::Http`] (only 409 is typed as [`HFError::Conflict`]). A busy repo's 409 is not one:
/// [`send_commit`] has already waited it out.
fn head_moved(e: &HFError) -> bool {
    match e {
        HFError::Conflict { .. } => !repo_busy(e),
        HFError::Http { context } => context.status.as_u16() == 412,
        _ => false,
    }
}

/// A live stderr byte-bar for an upload `create_commit`, redrawn in place (`\r`) as xet streams the
/// data. Small commits skip the byte phase (no `Progress` events) — then nothing is drawn and the
/// caller's "uploading…" line is the only trace. A commit that deletes files says how many.
/// `Send + Sync`: hf-hub calls it off the main thread.
struct UploadBar {
    files: AtomicUsize,
    deletes: usize,
}

impl ProgressHandler for UploadBar {
    fn on_progress(&self, event: &ProgressEvent) {
        let ProgressEvent::Upload(e) = event else {
            return;
        };
        match e {
            UploadEvent::Progress {
                bytes_completed,
                total_bytes,
                bytes_per_sec,
                ..
            } => {
                let pct = if *total_bytes > 0 {
                    100.0 * *bytes_completed as f64 / *total_bytes as f64
                } else {
                    0.0
                };
                let rate = bytes_per_sec
                    .map(|r| format!(" ({}/s)", human_bytes(r as u64)))
                    .unwrap_or_default();
                eprint!(
                    "\r    uploaded {} / {}  {pct:.0}%{rate}   ",
                    human_bytes(*bytes_completed),
                    human_bytes(*total_bytes),
                );
                let _ = std::io::stderr().flush();
            }
            UploadEvent::Start { total_files, .. } => self.files.store(*total_files, Ordering::Relaxed),
            UploadEvent::Committing => {
                eprint!("\r    committing…                                        ");
                let _ = std::io::stderr().flush();
            }
            UploadEvent::Complete => {
                if self.files.load(Ordering::Relaxed) > 0 {
                    eprintln!("\r    upload complete                                     ");
                }
                if self.deletes > 0 {
                    eprintln!(
                        "\r    deleted {} files                                     ",
                        self.deletes
                    );
                }
            }
        }
    }
}

/// The upload progress handler for a `create_commit` deleting `deletes` files, shared by
/// [`send_commit`] and [`first_publish`]. See [`UploadBar`].
pub(crate) fn upload_progress(deletes: usize) -> Progress {
    Progress::new(UploadBar {
        files: AtomicUsize::new(0),
        deletes,
    })
}

/// Human-readable byte count (binary units), for the upload bar.
fn human_bytes(n: u64) -> String {
    const UNITS: [&str; 4] = ["B", "KiB", "MiB", "GiB"];
    let mut v = n as f64;
    let mut i = 0;
    while v >= 1024.0 && i < UNITS.len() - 1 {
        v /= 1024.0;
        i += 1;
    }
    if i == 0 {
        format!("{n} B")
    } else {
        format!("{v:.1} {}", UNITS[i])
    }
}

/// Installs a [`CaptureStore`] in front of the store Lance built for the dataset URI, and holds the
/// shared capture map so the operation that created it can read the files back once Lance is done.
#[derive(Debug)]
struct CaptureWrapper {
    captured: Captured,
}

impl WrappingObjectStore for CaptureWrapper {
    fn wrap(&self, _prefix: &str, original: Arc<dyn OSObjectStore>) -> Arc<dyn OSObjectStore> {
        Arc::new(CaptureStore::new(original, self.captured.clone()))
    }

    /// A pushed-down listing would walk past the capture: it would miss the files written here and
    /// still list the ones deleted here.
    fn wrap_paginated(
        &self,
        _prefix: &str,
        _original: Arc<dyn PaginatedListStore>,
    ) -> Option<Arc<dyn PaginatedListStore>> {
        None
    }
}

/// Fetches a repo file whole at the pinned revision, returning its path in hf-hub's local cache.
/// Pinning to a commit SHA (not a branch) is what makes a warm read zero-network — hf-hub serves the
/// cached blob without a request — and fixes every read at one immutable revision.
#[derive(Debug)]
struct HubFetcher {
    repo: Arc<HFRepository<RepoTypeDataset>>,
    revision: String,
}

#[async_trait]
impl FileFetcher for HubFetcher {
    async fn fetch(&self, filename: &str) -> Result<PathBuf> {
        let path = self
            .repo
            .download_file()
            .filename(filename)
            .revision(self.revision.clone())
            .send()
            .await
            .with_context(|| format!("caching {filename}@{}", self.revision))?;
        let mount = std::env::var_os("FUNES_HUB_CACHE_MOUNT").map(PathBuf::from);
        localize_restored_blob(path, mount.as_deref(), || async {
            self.repo
                .download_file()
                .filename(filename)
                .revision(self.revision.clone())
                .force_download(true)
                .send()
                .await
                .with_context(|| format!("materializing {filename}@{}", self.revision))
        })
        .await
    }

    /// A cache entry links to a blob named by the object's expected hash rather than by the bytes
    /// on disk, so dropping the entry alone would resolve to those same bytes again.
    async fn discard(&self, path: &Path) -> Result<()> {
        let blob = std::fs::read_link(path).ok().map(|target| match path.parent() {
            Some(dir) if target.is_relative() => dir.join(target),
            _ => target,
        });
        std::fs::remove_file(path).with_context(|| format!("discarding {}", path.display()))?;
        if let Some(blob) = blob {
            let _ = std::fs::remove_file(blob);
        }
        Ok(())
    }
}

/// Resolve only the two local links installed by the Bucket restore, without statting the mount.
fn restored_blob(path: &Path, mount: Option<&Path>) -> Option<PathBuf> {
    let mount = mount.filter(|root| root.is_absolute())?;
    let target = std::fs::read_link(path).ok()?;
    let blob = if target.is_relative() {
        path.parent()?.join(target)
    } else {
        target
    };
    let remote = std::fs::read_link(&blob).ok()?;
    (remote.is_absolute()
        && remote.starts_with(mount)
        && !remote.components().any(|part| part == std::path::Component::ParentDir))
    .then_some(blob)
}

async fn localize_restored_blob<F, Fut>(path: PathBuf, mount: Option<&Path>, download: F) -> Result<PathBuf>
where
    F: FnOnce() -> Fut,
    Fut: Future<Output = Result<PathBuf>>,
{
    let Some(blob) = restored_blob(&path, mount) else {
        return Ok(path);
    };
    // Bucket links are remote I/O, not a local cache. Coalesce across readers and maintenance;
    // hf-hub atomically replaces the link only after its whole-file download succeeds.
    let lock = std::fs::OpenOptions::new()
        .create(true)
        .truncate(false)
        .write(true)
        .open(blob.with_extension("funes-local.lock"))?;
    let waiting = super::fetch_store::track_read_io(super::fetch_store::ReadIoKind::CacheWait);
    loop {
        match lock.try_lock() {
            Ok(()) => break,
            Err(std::fs::TryLockError::WouldBlock) => tokio::time::sleep(Duration::from_millis(25)).await,
            Err(std::fs::TryLockError::Error(error)) => return Err(error.into()),
        }
    }
    drop(waiting);
    if restored_blob(&path, mount).is_none() {
        return Ok(path);
    }
    let _download = super::fetch_store::track_read_io(super::fetch_store::ReadIoKind::Materialize);
    download().await
}

/// Installs a [`FetchStore`] backed by a [`HubFetcher`] in front of the store Lance built for a
/// remote read. The read mirror of [`CaptureWrapper`]; built by the caller, where the repo handle
/// and head SHA are known, because `wrap` is handed only the built store and an opendal-internal
/// prefix.
#[derive(Debug)]
pub(crate) struct FetchWrapper {
    fetcher: Arc<dyn FileFetcher>,
}

impl FetchWrapper {
    /// A wrapper that serves reads from `repo` at the pinned `revision` (a commit SHA).
    pub(crate) fn new(repo: Arc<HFRepository<RepoTypeDataset>>, revision: String) -> Self {
        Self {
            fetcher: Arc::new(HubFetcher { repo, revision }),
        }
    }
}

impl WrappingObjectStore for FetchWrapper {
    fn wrap(&self, _prefix: &str, original: Arc<dyn OSObjectStore>) -> Arc<dyn OSObjectStore> {
        let cached = Arc::new(FetchStore::new(original, self.fetcher.clone()));
        Arc::new(ShardStore::new(cached))
    }

    /// [`FetchStore`] serves reads alone, but the surrounding [`ShardStore`] must translate listings.
    fn wrap_paginated(
        &self,
        _prefix: &str,
        _original: Arc<dyn PaginatedListStore>,
    ) -> Option<Arc<dyn PaginatedListStore>> {
        None
    }
}

/// Resolve the head commit of `branch` for `owner/name` and build a [`FetchWrapper`] pinned to it,
/// returning the wrapper and that SHA. The SHA is the read pin: the caller puts it in the dataset's
/// `hf_revision` so Lance reads the exact commit the wrapper serves, and a commit SHA is what makes
/// warm reads zero-network. A fresh repo handle is built from `token` and shared into the wrapper.
pub(crate) async fn fetch_wrapper(
    owner: &str,
    name: &str,
    token: Option<&str>,
    branch: &str,
) -> Result<(Arc<FetchWrapper>, String)> {
    let repo = Arc::new(hub::client(token, true)?.dataset(owner, name));
    let sha = head_oid(&repo, branch).await?;
    Ok((Arc::new(FetchWrapper::new(repo, sha.clone())), sha))
}

/// Build the read wrapper for one already-resolved immutable commit. Canonical ingestion uses this
/// so revision selection and the eventual `parent_commit` refer to exactly the same snapshot.
pub(crate) fn fetch_wrapper_at(
    owner: &str,
    name: &str,
    token: Option<&str>,
    revision: &str,
) -> Result<Arc<FetchWrapper>> {
    let repo = Arc::new(hub::client(token, true)?.dataset(owner, name));
    Ok(Arc::new(FetchWrapper::new(repo, revision.to_string())))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::EmbeddingModel;
    use arrow_array::StringArray;
    use arrow_schema::{DataType, Field, Schema};
    use futures::TryStreamExt;
    use lance::dataset::transaction::{Operation, Transaction};
    use lance::dataset::CommitBuilder;
    use lance::index::DatasetIndexExt;
    use lance_index::scalar::FullTextSearchQuery;
    use lance_index::scalar::ScalarIndexParams;
    use lance_index::IndexType;
    use lance_io::object_store::ObjectStore as LanceObjectStore;
    use object_store::memory::InMemory;
    use object_store::path::Path as OPath;
    use object_store::ObjectStoreExt;
    use object_store::{GetOptions, PutOptions, PutPayload};

    #[derive(Debug)]
    struct StaticFetcher {
        dir: tempfile::TempDir,
        body: Bytes,
    }

    #[async_trait]
    impl FileFetcher for StaticFetcher {
        async fn fetch(&self, _filename: &str) -> Result<PathBuf> {
            let path = self.dir.path().join("cached");
            std::fs::write(&path, &self.body)?;
            Ok(path)
        }

        async fn discard(&self, path: &Path) -> Result<()> {
            std::fs::remove_file(path)?;
            Ok(())
        }
    }

    #[cfg(unix)]
    mod restored_cache {
        use super::*;
        use std::os::unix::fs::symlink;
        use std::sync::atomic::{AtomicUsize, Ordering};

        struct Fixture {
            _dir: tempfile::TempDir,
            mount: PathBuf,
            origin: PathBuf,
            blob: PathBuf,
            snapshot: PathBuf,
        }

        impl Fixture {
            fn new() -> Self {
                let dir = tempfile::tempdir().unwrap();
                let mount = dir.path().join("bucket");
                let cache = dir.path().join("cache");
                std::fs::create_dir(&mount).unwrap();
                std::fs::create_dir_all(cache.join("blobs")).unwrap();
                std::fs::create_dir_all(cache.join("snapshots/revision")).unwrap();
                let origin = mount.join("blob");
                let blob = cache.join("blobs/hash");
                let snapshot = cache.join("snapshots/revision/index.lance");
                std::fs::write(&origin, b"immutable index bytes").unwrap();
                symlink(&origin, &blob).unwrap();
                symlink("../../blobs/hash", &snapshot).unwrap();
                Self {
                    _dir: dir,
                    mount,
                    origin,
                    blob,
                    snapshot,
                }
            }

            fn finish_download(&self) -> Result<PathBuf> {
                let incomplete = self.blob.with_extension("incomplete");
                std::fs::write(&incomplete, b"immutable index bytes")?;
                std::fs::rename(incomplete, &self.blob)?;
                Ok(self.snapshot.clone())
            }
        }

        #[test]
        fn detects_only_configured_bucket_links_without_reading_them() {
            let fixture = Fixture::new();
            assert!(restored_blob(&fixture.snapshot, Some(&fixture.mount)).is_some());
            assert!(restored_blob(&fixture.snapshot, None).is_none());
            assert!(restored_blob(&fixture.snapshot, Some(Path::new("bucket"))).is_none());
            assert!(restored_blob(&fixture.snapshot, Some(&fixture.mount.join("other"))).is_none());
            std::fs::remove_file(&fixture.origin).unwrap();
            assert!(restored_blob(&fixture.snapshot, Some(&fixture.mount)).is_some());
            std::fs::remove_file(&fixture.blob).unwrap();
            symlink(fixture.mount.join("../outside"), &fixture.blob).unwrap();
            assert!(restored_blob(&fixture.snapshot, Some(&fixture.mount)).is_none());
        }

        #[tokio::test]
        async fn local_blob_and_disabled_mount_do_not_download() {
            let fixture = Fixture::new();
            let path = localize_restored_blob(fixture.snapshot.clone(), None, || async {
                panic!("disabled cache must not download")
            })
            .await
            .unwrap();
            assert_eq!(path, fixture.snapshot);
            fixture.finish_download().unwrap();
            localize_restored_blob(fixture.snapshot.clone(), Some(&fixture.mount), || async {
                panic!("local cache hit must not download")
            })
            .await
            .unwrap();
        }

        #[tokio::test]
        async fn concurrent_ranges_materialize_once_and_preserve_origin() {
            let fixture = Fixture::new();
            let calls = AtomicUsize::new(0);
            let results = futures::future::join_all((0..8).map(|_| {
                localize_restored_blob(fixture.snapshot.clone(), Some(&fixture.mount), || async {
                    calls.fetch_add(1, Ordering::SeqCst);
                    tokio::time::sleep(Duration::from_millis(60)).await;
                    fixture.finish_download()
                })
            }))
            .await;
            assert!(results.iter().all(Result::is_ok));
            assert_eq!(calls.load(Ordering::SeqCst), 1);
            assert!(!fixture.blob.is_symlink());
            assert_eq!(std::fs::read(&fixture.snapshot).unwrap(), b"immutable index bytes");
            assert_eq!(std::fs::read(&fixture.origin).unwrap(), b"immutable index bytes");
        }

        #[tokio::test]
        async fn failed_download_preserves_link_and_allows_retry() {
            let fixture = Fixture::new();
            let result = localize_restored_blob(fixture.snapshot.clone(), Some(&fixture.mount), || async {
                anyhow::bail!("download failed")
            })
            .await;
            assert!(result.is_err());
            assert!(fixture.blob.is_symlink());
            assert_eq!(std::fs::read(&fixture.snapshot).unwrap(), b"immutable index bytes");
            localize_restored_blob(fixture.snapshot.clone(), Some(&fixture.mount), || async {
                fixture.finish_download()
            })
            .await
            .unwrap();
            assert!(!fixture.blob.is_symlink());
        }

        #[tokio::test]
        async fn cancelled_download_releases_lock_and_preserves_link() {
            let fixture = Fixture::new();
            let result = tokio::time::timeout(
                Duration::from_millis(30),
                localize_restored_blob(fixture.snapshot.clone(), Some(&fixture.mount), || {
                    std::future::pending()
                }),
            )
            .await;
            assert!(result.is_err());
            assert!(fixture.blob.is_symlink());
            tokio::time::timeout(
                Duration::from_secs(1),
                localize_restored_blob(fixture.snapshot.clone(), Some(&fixture.mount), || async {
                    fixture.finish_download()
                }),
            )
            .await
            .unwrap()
            .unwrap();
        }
    }

    // Real hf-hub errors, without network credentials or Hub/Voyage calls.
    fn mock_commit_repo<T: AsRef<str>>(
        responses: Vec<(u16, Option<u64>, T)>,
    ) -> (HFRepository<RepoTypeDataset>, std::thread::JoinHandle<Vec<String>>) {
        use std::io::Read;
        use std::net::TcpListener;

        let responses: Vec<_> = responses
            .into_iter()
            .map(|(status, retry_after, body)| (status, retry_after, body.as_ref().to_string()))
            .collect();
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let endpoint = format!("http://{}", listener.local_addr().unwrap());
        listener.set_nonblocking(true).unwrap();
        let handle = std::thread::spawn(move || {
            let mut requests = Vec::new();
            for (status, retry_after, body) in responses {
                let start = Instant::now();
                let mut stream = loop {
                    match listener.accept() {
                        Ok((stream, _)) => break stream,
                        Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                            assert!(start.elapsed() < Duration::from_secs(10), "missing mock request");
                            std::thread::sleep(Duration::from_millis(5));
                        }
                        Err(error) => panic!("mock accept: {error}"),
                    }
                };
                stream.set_nonblocking(false).unwrap();
                stream.set_read_timeout(Some(Duration::from_secs(5))).unwrap();
                let mut bytes = Vec::new();
                loop {
                    let mut buffer = [0; 4096];
                    let count = stream.read(&mut buffer).unwrap();
                    assert!(count > 0, "incomplete mock request");
                    bytes.extend_from_slice(&buffer[..count]);
                    if let Some(end) = bytes.windows(4).position(|part| part == b"\r\n\r\n") {
                        let header = String::from_utf8_lossy(&bytes[..end]);
                        let length = header
                            .lines()
                            .find_map(|line| {
                                let (name, value) = line.split_once(':')?;
                                name.eq_ignore_ascii_case("content-length")
                                    .then(|| value.trim().parse::<usize>().unwrap())
                            })
                            .unwrap_or(0);
                        if bytes.len() >= end + 4 + length {
                            break;
                        }
                    }
                }
                requests.push(String::from_utf8(bytes).unwrap());
                if status == 0 {
                    // Simulate a server closing the connection before any response headers.
                    continue;
                }
                let retry_header = retry_after
                    .map(|seconds| format!("Retry-After: {seconds}\r\n"))
                    .unwrap_or_default();
                write!(stream, "HTTP/1.1 {status} Test\r\nContent-Type: application/json\r\n{retry_header}Content-Length: {}\r\nConnection: close\r\n\r\n{body}", body.len()).unwrap();
            }
            requests
        });
        let repo = hf_hub::HFClient::builder()
            .endpoint(endpoint)
            .token("test-only")
            .retry_max_attempts(0)
            .build()
            .unwrap()
            .dataset("test", "memory");
        (repo, handle)
    }

    #[tokio::test]
    async fn commit_retry_preserves_parent_and_captured_bytes_after_rate_limit() {
        let (repo, requests) = mock_commit_repo(vec![
            (429, Some(0), r#"{"error":"too many commits"}"#),
            (
                200,
                None,
                r#"{"files":[{"path":"vectors","uploadMode":"regular","shouldIgnore":false}]}"#,
            ),
            (429, Some(0), r#"{"error":"commit rate limit"}"#),
            (
                200,
                None,
                r#"{"files":[{"path":"vectors","uploadMode":"regular","shouldIgnore":false}]}"#,
            ),
            (200, None, r#"{"commitOid":"new-head"}"#),
        ]);
        let info = send_commit_chunks(
            &repo,
            vec![CommitOperation::add_bytes("vectors", b"already-embedded".to_vec())],
            Some("original-head".to_string()),
            "main",
            "same vectors".to_string(),
        )
        .await
        .unwrap();
        assert_eq!(info.commit_oid.as_deref(), Some("new-head"));
        let requests = requests.join().unwrap();
        assert_eq!(requests.len(), 5);
        let bodies = requests
            .iter()
            .map(|request| request.split_once("\r\n\r\n").unwrap().1)
            .collect::<Vec<_>>();
        assert_eq!(bodies[0], bodies[1]);
        assert_eq!(bodies[1], bodies[3]);
        assert_eq!(bodies[2], bodies[4]);
        assert!(bodies[2].contains("original-head"));
    }

    #[tokio::test]
    async fn commit_retry_respects_budget_bootstrap_and_cas_conflicts() {
        for (status, guarded) in [(429, false), (409, true), (412, true), (401, true)] {
            let (repo, requests) = mock_commit_repo(vec![(status, Some(120), r#"{"error":"bounded"}"#)]);
            let error = send_commit_chunks(
                &repo,
                vec![CommitOperation::delete("old")],
                guarded.then_some("parent".to_string()),
                "main",
                "test".to_string(),
            )
            .await
            .unwrap_err();
            assert_eq!(requests.join().unwrap().len(), 1);
            assert!(commit_retry_wait(&error, 1, Duration::ZERO, Duration::from_secs(100), guarded).is_none());
            if status == 429 {
                assert_eq!(
                    commit_retry_wait(&error, 1, Duration::ZERO, Duration::from_secs(121), true),
                    Some(Duration::from_secs(120))
                );
                assert!(
                    commit_retry_wait(&error, MAX_COMMIT_ATTEMPTS, Duration::ZERO, COMMIT_RETRY_BUDGET, true).is_none()
                );
                assert!(commit_retry_wait(&error, 1, Duration::from_secs(1), Duration::from_secs(121), true).is_none());
            }
        }
    }

    #[tokio::test]
    async fn commit_retry_shares_deadline_across_chunks() {
        let started = Instant::now();
        let budget = Duration::from_millis(300);
        let (repo, first_requests) = mock_commit_repo(vec![(200, None, r#"{"commitOid":"first"}"#)]);
        retry_commit(true, started, budget, || async {
            repo.create_commit()
                .operations(vec![CommitOperation::delete("old")])
                .commit_message("first")
                .parent_commit("parent")
                .send()
                .await
        })
        .await
        .unwrap();
        assert_eq!(first_requests.join().unwrap().len(), 1);
        tokio::time::sleep(Duration::from_millis(150)).await;
        let (repo, second_requests) = mock_commit_repo(vec![(503, None, r#"{"error":"temporary"}"#)]);
        let error = retry_commit(true, started, budget, || async {
            repo.create_commit()
                .operations(vec![CommitOperation::delete("old")])
                .commit_message("second")
                .parent_commit("first")
                .send()
                .await
        })
        .await
        .unwrap_err();
        assert!(matches!(error, HFError::Http { .. }));
        assert_eq!(second_requests.join().unwrap().len(), 1);
    }

    #[tokio::test]
    async fn commit_retry_handles_connection_reset_without_reembedding() {
        let (repo, requests) = mock_commit_repo(vec![(0, None, ""), (200, None, r#"{"commitOid":"recovered"}"#)]);
        let info = send_commit_chunks(
            &repo,
            vec![CommitOperation::delete("old")],
            Some("parent".to_string()),
            "main",
            "same operation".to_string(),
        )
        .await
        .unwrap();
        assert_eq!(info.commit_oid.as_deref(), Some("recovered"));
        let requests = requests.join().unwrap();
        assert_eq!(requests.len(), 2);
        assert_eq!(
            requests[0].split_once("\r\n\r\n").unwrap().1,
            requests[1].split_once("\r\n\r\n").unwrap().1
        );
    }

    #[tokio::test]
    async fn commit_retry_keeps_transient_retry_and_safe_metrics() {
        let (repo, requests) = mock_commit_repo(vec![
            (503, None, r#"{"error":"temporary"}"#),
            (200, None, r#"{"commitOid":"recovered"}"#),
        ]);
        let info = send_commit_chunks(
            &repo,
            vec![CommitOperation::delete("old")],
            Some("parent".to_string()),
            "main",
            "test".to_string(),
        )
        .await
        .unwrap();
        assert_eq!(info.commit_oid.as_deref(), Some("recovered"));
        assert_eq!(requests.join().unwrap().len(), 2);
        let metric = HfCommitMetric {
            stage: "hf_commit_attempt",
            duration_ms: 1.0,
            attempt: 1,
            status_code: 429,
            phase: "preupload",
            retry_after_ms: Some(120000.0),
            backoff_ms: 120000.0,
            retrying: true,
            limit_kind: "commit_action",
        };
        let json = serde_json::to_value(metric).unwrap();
        assert_eq!(json.as_object().unwrap().len(), 9);
        assert_eq!(json["retry_after_ms"], 120000.0);
        for forbidden in ["http://", "https://", "token", "already-embedded", "original-head"] {
            assert!(!json.to_string().contains(forbidden));
        }
    }

    #[derive(Debug)]
    struct UnusedFetcher;

    #[async_trait]
    impl FileFetcher for UnusedFetcher {
        async fn fetch(&self, _filename: &str) -> Result<PathBuf> {
            anyhow::bail!("test reads use the in-memory backend")
        }

        async fn discard(&self, _path: &Path) -> Result<()> {
            anyhow::bail!("test reads use the in-memory backend")
        }
    }

    #[derive(Debug)]
    struct UnusedLister;

    #[async_trait]
    impl PaginatedListStore for UnusedLister {
        async fn list_paginated(
            &self,
            _prefix: Option<&str>,
            _options: object_store::list::PaginatedListOptions,
        ) -> object_store::Result<object_store::list::PaginatedListResult> {
            unreachable!("listing must pass through the sharded capture wrappers")
        }
    }

    #[test]
    fn paginated_listing_cannot_bypass_shards_or_captured_writes() {
        let original: Arc<dyn PaginatedListStore> = Arc::new(UnusedLister);
        let fetch = Arc::new(FetchWrapper { fetcher: Arc::new(UnusedFetcher) });
        let capture = Arc::new(CaptureWrapper { captured: Captured::default() });
        assert!(fetch.wrap_paginated("fixture", original.clone()).is_none());
        assert!(capture.wrap_paginated("fixture", original.clone()).is_none());
        assert!(capturing_fetch_wrapper(fetch, capture).wrap_paginated("fixture", original).is_none());
    }

    /// Use the real production decorators over a deterministic backend, without Hub or Voyage.
    #[derive(Debug)]
    struct TestStoreWrapper {
        store: Arc<dyn OSObjectStore>,
        sharded: bool,
        captured: Option<Captured>,
    }

    impl WrappingObjectStore for TestStoreWrapper {
        fn wrap(&self, prefix: &str, _original: Arc<dyn OSObjectStore>) -> Arc<dyn OSObjectStore> {
            if let Some(captured) = &self.captured {
                capturing_fetch_wrapper(
                    Arc::new(FetchWrapper {
                        fetcher: Arc::new(UnusedFetcher),
                    }),
                    Arc::new(CaptureWrapper {
                        captured: captured.clone(),
                    }),
                )
                .wrap(prefix, self.store.clone())
            } else if self.sharded {
                FetchWrapper {
                    fetcher: Arc::new(UnusedFetcher),
                }
                .wrap(prefix, self.store.clone())
            } else {
                self.store.clone()
            }
        }

        fn wrap_paginated(
            &self,
            _prefix: &str,
            _original: Arc<dyn PaginatedListStore>,
        ) -> Option<Arc<dyn PaginatedListStore>> {
            None
        }
    }

    #[test]
    fn metric_lines_survive_unterminated_upload_progress() {
        let mut output = "\r    committing…       ".as_bytes().to_vec();
        for (attempt, status_code) in [(1, 429), (2, 200)] {
            let metric = HfCommitMetric {
                stage: "hf_commit_attempt",
                duration_ms: 1.0,
                attempt,
                status_code,
                phase: "commit",
                retry_after_ms: None,
                backoff_ms: 0.0,
                retrying: attempt == 1,
                limit_kind: "unknown",
            };
            write_metric(&mut output, &serde_json::to_string(&metric).unwrap()).unwrap();
        }
        let text = String::from_utf8(output).unwrap();
        assert!(text.starts_with("\r    committing…       \n"));
        let metrics: Vec<serde_json::Value> = text
            .lines()
            .filter_map(|line| line.strip_prefix("funes_metric "))
            .map(|json| serde_json::from_str(json).unwrap())
            .collect();
        assert_eq!(metrics.len(), 2);
        assert_eq!(metrics[0]["status_code"], 429);
        assert_eq!(metrics[1]["status_code"], 200);
        assert_eq!(metrics[1]["attempt"], 2);
    }

    #[test]
    fn remote_phase_metrics_env_and_schema_are_strict_and_leak_free() {
        assert!(ingest_metrics_enabled_from(Some("1")));
        assert!(ingest_metrics_enabled_from(Some("true")));
        assert!(ingest_metrics_enabled_from(Some("yes")));
        assert!(ingest_metrics_enabled_from(Some("on")));
        assert!(ingest_metrics_enabled_from(Some(" TRUE ")));
        assert!(!ingest_metrics_enabled_from(Some("0")));
        assert!(!ingest_metrics_enabled_from(Some("false")));
        assert!(!ingest_metrics_enabled_from(Some("off")));
        assert!(!ingest_metrics_enabled_from(None));

        assert_eq!(round_ms(Duration::from_millis(0)), 0.0);
        assert_eq!(round_ms(Duration::from_micros(12345)), 12.35);
        assert_eq!(round_ms(Duration::from_millis(50)), 50.0);

        for stage in [
            "lance_append",
            "lance_delete",
            "captured_files",
            "write_ops",
            "hf_commit_chunk",
            "hf_commit_wait",
        ] {
            let metric = IngestPhaseMetric {
                stage,
                duration_ms: 12.34,
            };
            let serialized = serde_json::to_string(&metric).unwrap();
            let value: serde_json::Value = serde_json::from_str(&serialized).unwrap();
            let obj = value.as_object().unwrap();
            assert_eq!(obj.len(), 2);
            assert_eq!(obj.get("stage").unwrap(), stage);
            assert_eq!(obj.get("duration_ms").unwrap(), 12.34);
            assert!(!serialized.contains("secret"));
            assert!(!serialized.contains("token"));
            assert!(!serialized.contains("http"));
            assert!(!serialized.contains('/'));
        }
    }

    #[test]
    fn write_ops_and_captured_files_emit_safely_when_enabled() {
        let wrapper = CaptureWrapper {
            captured: Captured::default(),
        };
        wrapper.captured.lock().unwrap().insert(
            object_store::path::Path::from("chunks.lance/data/test.lance"),
            Some(CapturedFile::from_bytes(b"metric_test_bytes").unwrap()),
        );
        let files = captured_files(&wrapper);
        assert_eq!(files.len(), 1);

        let (ops, _dir) = write_ops(&files).unwrap();
        assert_eq!(ops.len(), 1);
    }

    #[test]
    fn upload_guards_keep_capture_files_alive_until_commit_finishes() {
        let body = CapturedFile::from_bytes(b"durable upload input").unwrap();
        let path = body.path().to_path_buf();
        let files = BTreeMap::from([("chunks.lance/data/upload.lance".to_string(), body)]);
        let (_ops, guards) = write_ops(&files).unwrap();
        drop(files);
        assert_eq!(std::fs::read(&path).unwrap(), b"durable upload input");
        drop(guards);
        assert!(
            !path.exists(),
            "commit guards own cleanup, not the discarded capture map"
        );
    }

    #[test]
    fn hf_http_phase_is_bounded_and_does_not_expose_urls() {
        assert_eq!(
            hf_http_phase("https://huggingface.co/api/datasets/a/b/preupload/main"),
            "preupload"
        );
        assert_eq!(
            hf_http_phase("https://huggingface.co/api/datasets/a/b/commit/main"),
            "commit"
        );
        assert_eq!(
            hf_http_phase("https://huggingface.co/api/datasets/a/b/tree/main"),
            "unknown"
        );
    }

    #[tokio::test]
    async fn capturing_write_path_reads_fetch_cache_and_never_forwards_writes() {
        let inner = Arc::new(InMemory::new());
        let source = object_store::path::Path::from("chunks.lance/data/source.lance");
        inner
            .put_opts(&source, PutPayload::from_static(b"remote"), PutOptions::default())
            .await
            .unwrap();

        let fetch = Arc::new(FetchWrapper {
            fetcher: Arc::new(StaticFetcher {
                dir: tempfile::tempdir().unwrap(),
                body: Bytes::from_static(b"cached"),
            }),
        });
        let capture = Arc::new(CaptureWrapper {
            captured: Captured::default(),
        });
        let store = capturing_fetch_wrapper(fetch, capture.clone()).wrap("", inner.clone());

        let read = store
            .get_opts(&source, GetOptions::default())
            .await
            .unwrap()
            .bytes()
            .await
            .unwrap();
        assert_eq!(read, Bytes::from_static(b"cached"), "reads must use FetchStore");

        let written = object_store::path::Path::from("chunks.lance/data/new.lance");
        store
            .put_opts(&written, PutPayload::from_static(b"captured"), PutOptions::default())
            .await
            .unwrap();
        assert_eq!(
            capture
                .captured
                .lock()
                .unwrap()
                .get(&written)
                .unwrap()
                .as_ref()
                .unwrap()
                .bytes(),
            Bytes::from_static(b"captured")
        );
        assert!(
            inner.get_opts(&written, GetOptions::default()).await.is_err(),
            "writes must not reach the remote store"
        );
    }

    /// Every object in `store`: path → size.
    async fn objects(store: &dyn OSObjectStore) -> BTreeMap<String, u64> {
        store
            .list(None)
            .map_ok(|m| (m.location.to_string(), m.size))
            .try_collect()
            .await
            .unwrap()
    }

    /// An in-memory store, not a local dataset: Lance writes part of a local dataset straight to
    /// disk, around the wrapper. Every Hub write goes through it.
    #[tokio::test]
    async fn refresh_through_the_capture_writes_nothing_to_the_store() {
        let uri = "shared-memory://capture-refresh/chunks.lance";
        let (store, _) = LanceObjectStore::from_uri(uri).await.unwrap();
        let schema = Arc::new(Schema::new(vec![Field::new("text", DataType::Utf8, false)]));
        let rows = |text: &str| {
            let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(StringArray::from(vec![text]))]).unwrap();
            RecordBatchIterator::new([Ok(batch)], schema.clone())
        };
        let mut ds = Dataset::write(rows("base"), uri, None).await.unwrap();
        dataset::build_indexes(&mut ds, |_| {}).await.unwrap();
        for text in ["one", "two", "three"] {
            ds.append(rows(text), None).await.unwrap();
        }
        let before = objects(store.inner.as_ref()).await;

        let wrapper = Arc::new(CaptureWrapper {
            captured: Captured::default(),
        });
        let mut ds = dataset::open_wrapped(uri, HashMap::new(), wrapper.clone())
            .await
            .unwrap();
        let read = ds.version().version;
        refresh(&mut ds, read, |_| {}).await.unwrap();
        assert_eq!(objects(store.inner.as_ref()).await, before, "nothing reaches the store");

        // Apply the captured writes as the commit would.
        let captured = wrapper.captured.lock().unwrap().clone();
        for (path, body) in captured {
            store.inner.put(&path, body.unwrap().bytes().into()).await.unwrap();
        }
        let ds = Dataset::open(uri).await.unwrap();
        assert_eq!(ds.get_fragments().len(), 2, "the appended fragments are merged");
        assert_eq!(ds.count_rows(None).await.unwrap(), 4);
        assert!(!dataset::fts_needs_refresh(&ds).await.unwrap());
    }

    fn text_rows(text: &str) -> impl arrow_array::RecordBatchReader + Send + 'static {
        let schema = Arc::new(Schema::new(vec![Field::new("text", DataType::Utf8, false)]));
        let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(StringArray::from(vec![text]))]).unwrap();
        RecordBatchIterator::new([Ok(batch)], schema)
    }

    /// A Hub at `hub/` holding an indexed row and three pushed ones, and a frozen copy of it at
    /// `snap/`, as a pinned revision reads it. Returns the store, the dataset and its version.
    async fn hub_and_snapshot(authority: &str) -> (Arc<LanceObjectStore>, Dataset, u64) {
        let hub_uri = format!("shared-memory://{authority}/hub/chunks.lance");
        let (store, _) = LanceObjectStore::from_uri(&hub_uri).await.unwrap();
        let mut ds = Dataset::write(text_rows("base"), &hub_uri, None).await.unwrap();
        dataset::build_indexes(&mut ds, |_| {}).await.unwrap();
        for text in ["one", "two", "three"] {
            ds.append(text_rows(text), None).await.unwrap();
        }
        for (path, _) in objects(store.inner.as_ref()).await {
            let body = store
                .inner
                .get(&path.as_str().into())
                .await
                .unwrap()
                .bytes()
                .await
                .unwrap();
            let copy = path.replacen("hub/", "snap/", 1);
            store.inner.put(&copy.as_str().into(), body.into()).await.unwrap();
        }
        let read = ds.version().version;
        (store, ds, read)
    }

    /// Compact the copy at `snap/`, upload its new files to `hub/`, and return its commits.
    async fn compact_snapshot(authority: &str, store: &LanceObjectStore, read: u64) -> Vec<Transaction> {
        let wrapper = Arc::new(CaptureWrapper {
            captured: Captured::default(),
        });
        let mut session = Dataset::open(&format!("shared-memory://{authority}/snap/chunks.lance"))
            .await
            .unwrap()
            .with_object_store_wrappers([wrapper.clone() as Arc<dyn WrappingObjectStore>]);
        let txns = refresh(&mut session, read, |_| {}).await.unwrap();
        let captured = wrapper.captured.lock().unwrap().clone();
        for (path, body) in captured {
            let path = path.as_ref().replacen("snap/", "hub/", 1);
            if let Some(body) = body.filter(|_| !is_version_file(&path)) {
                store
                    .inner
                    .put(&path.as_str().into(), body.bytes().into())
                    .await
                    .unwrap();
            }
        }
        txns
    }

    #[tokio::test]
    async fn the_cleanup_spares_a_file_uploaded_ahead_of_its_manifest() {
        let uri = "shared-memory://staged/m/chunks.lance";
        let (store, _) = LanceObjectStore::from_uri(uri).await.unwrap();
        let mut ds = Dataset::write(text_rows("base"), uri, None).await.unwrap();
        let staged = OPath::from("m/chunks.lance/data/staged.lance");
        store.inner.put(&staged, "x".into()).await.unwrap();
        ds.append(text_rows("one"), None).await.unwrap();

        delete_old_versions_keeping_uploads(&ds).await.unwrap();

        assert_eq!(ds.versions().await.unwrap().len(), 1);
        assert!(store.inner.head(&staged).await.is_ok(), "the staged file is kept");
    }

    #[tokio::test]
    async fn a_compaction_replays_onto_a_push_that_landed_meanwhile() {
        let (store, ds, read) = hub_and_snapshot("replay").await;
        let compacted = dataset::fragments_to_compact(&ds).await.unwrap();
        assert_eq!(compacted, 3);

        // On the Hub: the compaction reserves its ids, then another host pushes.
        let reserve = Operation::ReserveFragments {
            num_fragments: compacted as u32,
        };
        let mut hub = CommitBuilder::new(Arc::new(ds))
            .execute(Transaction::new(read, reserve, None))
            .await
            .unwrap();
        hub.append(text_rows("pushed"), None).await.unwrap();

        let txns = compact_snapshot("replay", &store, read).await;
        let head = replay_onto(hub, read, &txns)
            .await
            .unwrap()
            .expect("only a push landed");
        delete_old_versions_keeping_uploads(&head).await.unwrap();

        head.validate().await.unwrap();
        assert_eq!(
            head.versions().await.unwrap().len(),
            1,
            "the cleanup leaves one version"
        );
        assert_eq!(head.count_rows(None).await.unwrap(), 5);
        let ids: Vec<usize> = head.get_fragments().iter().map(|f| f.id()).collect();
        assert_eq!(ids.len(), 3, "the indexed row, the merged ones, the push: {ids:?}");
        let mut indexed = head.scan();
        indexed
            .full_text_search(FullTextSearchQuery::new("two".to_string()))
            .unwrap()
            .fast_search();
        assert_eq!(indexed.count_rows().await.unwrap(), 1, "the index finds a merged row");
        let mut unindexed = head.scan();
        unindexed
            .full_text_search(FullTextSearchQuery::new("pushed".to_string()))
            .unwrap();
        assert_eq!(
            unindexed.count_rows().await.unwrap(),
            1,
            "the pushed row is still found"
        );
    }

    #[tokio::test]
    async fn a_compaction_replays_onto_a_push_that_counted_its_languages() {
        let (store, ds, read) = hub_and_snapshot("languages").await;
        let reserve = Operation::ReserveFragments {
            num_fragments: dataset::fragments_to_compact(&ds).await.unwrap() as u32,
        };
        let mut hub = CommitBuilder::new(Arc::new(ds))
            .execute(Transaction::new(read, reserve, None))
            .await
            .unwrap();
        hub.append(text_rows("pushed"), None).await.unwrap();
        let counts = dataset::Languages {
            rows: 5,
            counts: BTreeMap::from([("eng".to_string(), 5)]),
        };
        dataset::stamp_languages(&mut hub, &counts).await.unwrap();

        let txns = compact_snapshot("languages", &store, read).await;
        let head = replay_onto(hub, read, &txns)
            .await
            .unwrap()
            .expect("only a push landed");
        assert_eq!(dataset::languages(&head), Some(counts));
    }

    #[tokio::test]
    async fn a_compaction_does_not_replay_over_deleted_history() {
        let (store, mut hub, read) = hub_and_snapshot("history").await;
        hub.append(text_rows("pushed"), None).await.unwrap();
        hub.append(text_rows("again"), None).await.unwrap();
        // Another compaction landed, then its cleanup deleted the versions in between.
        let manifest = format!("hub/chunks.lance/_versions/{}.manifest", u64::MAX - (read + 1));
        store.inner.delete(&manifest.as_str().into()).await.unwrap();

        let txns = compact_snapshot("history", &store, read).await;
        assert!(replay_onto(hub, read, &txns).await.unwrap().is_none());
    }

    #[tokio::test]
    async fn a_compaction_does_not_replay_onto_another_compaction() {
        let (store, mut ds, read) = hub_and_snapshot("refuse").await;
        dataset::compact_fragments(&mut ds, |_| {}).await.unwrap();

        let txns = compact_snapshot("refuse", &store, read).await;
        assert!(replay_onto(ds, read, &txns).await.unwrap().is_none());
    }

    #[tokio::test]
    async fn an_append_refuses_rows_of_another_model() {
        let (e5, bge) = (EmbeddingModel::MultilingualE5Small, EmbeddingModel::BgeSmallEn);
        let rows = RecordBatchIterator::new(std::iter::empty(), dataset::schema(e5));
        let ds = Dataset::write(rows, "shared-memory://model/chunks.lance", None)
            .await
            .unwrap();
        assert!(same_model(&ds, &dataset::schema(e5)).is_ok());
        let err = same_model(&ds, &dataset::schema(bge)).unwrap_err().to_string();
        assert!(err.contains(e5.id()) && err.contains(bge.id()), "{err}");
    }

    #[tokio::test]
    async fn captured_deletes_leave_out_the_folders() {
        let wrapper = CaptureWrapper {
            captured: Captured::default(),
        };
        for path in [
            "m/_indices/u",
            "m/_indices/u/a.lance",
            "m/_indices/u-2.lance",
            "m/_versions/1.manifest",
        ] {
            wrapper.captured.lock().unwrap().insert(path.into(), None);
        }
        let body = serde_json::json!([
            {"type":"file", "path":"m/_indices/u-2.lance", "size":1, "oid":"a"},
            {"type":"file", "path":"m/_indices/u/a.lance", "size":1, "oid":"b"},
            {"type":"file", "path":"m/_versions/1.manifest", "size":1, "oid":"c"}
        ])
        .to_string();
        let (repo, requests) = mock_commit_repo(vec![(200, None, body)]);
        let paths: Vec<String> = captured_deletes(&wrapper, &repo, "snapshot")
            .await
            .unwrap()
            .into_iter()
            .map(|op| match op {
                CommitOperation::Delete { path_in_repo } => path_in_repo,
                CommitOperation::Add { .. } => unreachable!(),
            })
            .collect();
        assert_eq!(
            paths,
            ["m/_indices/u-2.lance", "m/_indices/u/a.lance", "m/_versions/1.manifest"]
        );
        assert_eq!(requests.join().unwrap().len(), 1);
    }

    #[tokio::test]
    async fn cleanup_resolves_only_existing_flat_and_sharded_files_at_the_snapshot() {
        let wrapper = CaptureWrapper {
            captured: Captured::default(),
        };
        let flat = "nested/chunks.lance/data/flat.lance";
        let sharded = "nested/chunks.lance/data/sharded.lance";
        let both = "nested/chunks.lance/_versions/1.manifest";
        for path in [
            flat,
            sharded,
            both,
            "nested/chunks.lance/data/missing.lance",
            "nested/chunks.lance/_indices/empty",
        ] {
            wrapper.captured.lock().unwrap().insert(path.into(), None);
        }
        wrapper.captured.lock().unwrap().insert(
            "nested/chunks.lance/data/new.lance".into(),
            Some(CapturedFile::from_bytes(b"new").unwrap()),
        );
        let expected = BTreeSet::from([
            flat.to_string(),
            physical_path(sharded).unwrap(),
            both.to_string(),
            physical_path(both).unwrap(),
        ]);
        let mut entries: Vec<_> = expected
            .iter()
            .map(|path| serde_json::json!({"type":"file", "path":path, "size":1, "oid":"blob"}))
            .collect();
        entries
            .push(serde_json::json!({"type":"directory", "path":"nested/chunks.lance/_indices/empty", "oid":"tree"}));
        let (repo, requests) = mock_commit_repo(vec![(200, None, serde_json::to_string(&entries).unwrap())]);
        let deletes = captured_deletes(&wrapper, &repo, "pinned-sha").await.unwrap();
        assert_eq!(
            deletes
                .iter()
                .map(|op| operation_path(op).to_string())
                .collect::<BTreeSet<_>>(),
            expected
        );
        let requests = requests.join().unwrap();
        assert_eq!(requests.len(), 1);
        assert!(requests[0].starts_with("POST /api/datasets/test/memory/paths-info/pinned-sha "));
        assert!(!requests[0].contains("new.lance"));
    }

    #[tokio::test]
    async fn cleanup_resolution_is_bounded_and_missing_files_are_a_noop() {
        let wrapper = CaptureWrapper {
            captured: Captured::default(),
        };
        for i in 0..MAX_COMMIT_OPS + 1 {
            wrapper
                .captured
                .lock()
                .unwrap()
                .insert(format!("chunks.lance/data/{i}.lance").into(), None);
        }
        let (repo, requests) = mock_commit_repo(vec![(200, None, "[]"); 3]);
        let deletes = captured_deletes(&wrapper, &repo, "snapshot").await.unwrap();
        assert!(deletes.is_empty());
        assert!(
            commit_in_parts(&repo, "main", deletes, "cleanup", Some("snapshot".into()))
                .await
                .unwrap()
                .is_none()
        );
        let requests = requests.join().unwrap();
        assert_eq!(requests.len(), 3);
        assert!(requests.iter().all(|request| request.contains("/paths-info/snapshot ")));
        let (repo, requests) = mock_commit_repo(Vec::<(u16, Option<u64>, &str)>::new());
        assert!(captured_deletes(
            &CaptureWrapper {
                captured: Captured::default()
            },
            &repo,
            "snapshot"
        )
        .await
        .unwrap()
        .is_empty());
        assert!(requests.join().unwrap().is_empty());
    }

    #[tokio::test]
    async fn cleanup_resolution_fails_closed_on_errors_or_unrequested_files() {
        let wrapper = CaptureWrapper {
            captured: Captured::default(),
        };
        wrapper
            .captured
            .lock()
            .unwrap()
            .insert("chunks.lance/data/old.lance".into(), None);
        for (status, body) in [
            (401, r#"{"error":"unauthorized"}"#),
            (503, r#"{"error":"unavailable"}"#),
            (
                200,
                r#"[{"type":"file","path":"chunks.lance/data/active.lance","size":1,"oid":"blob"}]"#,
            ),
        ] {
            let (repo, requests) = mock_commit_repo(vec![(status, None, body)]);
            assert!(captured_deletes(&wrapper, &repo, "snapshot").await.is_err());
            assert_eq!(requests.join().unwrap().len(), 1);
        }
    }

    #[tokio::test]
    async fn cleanup_commits_chain_the_pinned_parent_and_stop_on_conflict_or_missing_oid() {
        let deletes = || {
            (0..MAX_COMMIT_OPS + 1)
                .map(|i| CommitOperation::delete(format!("old-{i}")))
                .collect()
        };
        let (repo, requests) = mock_commit_repo(vec![
            (200, None, r#"{"commitOid":"first"}"#),
            (200, None, r#"{"commitOid":"second"}"#),
        ]);
        assert_eq!(
            commit_in_parts(&repo, "main", deletes(), "cleanup", Some("snapshot".into()))
                .await
                .unwrap()
                .as_deref(),
            Some("second")
        );
        let requests = requests.join().unwrap();
        assert_eq!(requests.len(), 2);
        assert!(requests[0].contains("snapshot"));
        assert!(requests[1].contains("first"));
        for responses in [
            vec![(412, None, r#"{"error":"head moved"}"#)],
            vec![
                (200, None, r#"{"commitOid":"first"}"#),
                (412, None, r#"{"error":"head moved"}"#),
            ],
        ] {
            let count = responses.len();
            let (repo, requests) = mock_commit_repo(responses);
            assert!(
                commit_in_parts(&repo, "main", deletes(), "cleanup", Some("snapshot".into()))
                    .await
                    .is_err()
            );
            assert_eq!(requests.join().unwrap().len(), count);
        }
        for body in [r#"{}"#, r#"{"commitOid":""}"#] {
            let (repo, requests) = mock_commit_repo(vec![
                (200, None, body),
                (
                    200,
                    None,
                    r#"{"branches":[{"name":"main","ref":"refs/heads/main","targetCommit":"concurrent-head"}],"tags":[]}"#,
                ),
            ]);
            let error = commit_in_parts(&repo, "main", deletes(), "cleanup", Some("snapshot".into()))
                .await
                .unwrap_err();
            assert!(error.to_string().contains("cleanup commit returned no oid"));
            let requests = requests.join().unwrap();
            assert_eq!(requests.len(), 2);
            assert!(
                requests[1].starts_with("GET "),
                "a later head must not authorize more deletes"
            );
        }
    }

    #[tokio::test]
    async fn indexed_delete_append_replaces_only_selected_sources() {
        let batch = |identities: &[&str], texts: &[&str]| {
            let schema = Arc::new(Schema::new(vec![
                Field::new("source_identity", DataType::Utf8, false),
                Field::new("text", DataType::Utf8, false),
            ]));
            let rows = RecordBatch::try_new(
                schema.clone(),
                vec![
                    Arc::new(StringArray::from(identities.to_vec())),
                    Arc::new(StringArray::from(texts.to_vec())),
                ],
            )
            .unwrap();
            (rows, schema)
        };
        let dir = tempfile::tempdir().unwrap();
        let uri = dir.path().join("replace.lance");
        let (initial, schema) = batch(&["replace-me", "keep-me"], &["old", "untouched"]);
        let mut ds = Dataset::write(
            RecordBatchIterator::new([Ok(initial)], schema),
            uri.to_str().unwrap(),
            None,
        )
        .await
        .unwrap();
        ds.create_index(
            &["source_identity"],
            IndexType::Scalar,
            Some("source_identity_idx".to_string()),
            &ScalarIndexParams::default(),
            true,
        )
        .await
        .unwrap();

        let (replacement, _) = batch(&["replace-me"], &["new"]);
        replace_dataset_rows(&mut ds, vec![replacement], "source_identity IN ('replace-me')")
            .await
            .unwrap();

        let rows = dataset::scan_rows(&ds, &["source_identity", "text"], None, None)
            .await
            .unwrap();
        let mut actual = rows
            .iter()
            .flat_map(|row| {
                let identities = row
                    .column_by_name("source_identity")
                    .unwrap()
                    .as_any()
                    .downcast_ref::<StringArray>()
                    .unwrap();
                let texts = row
                    .column_by_name("text")
                    .unwrap()
                    .as_any()
                    .downcast_ref::<StringArray>()
                    .unwrap();
                (0..row.num_rows()).map(|index| (identities.value(index).to_string(), texts.value(index).to_string()))
            })
            .collect::<Vec<_>>();
        actual.sort();
        assert_eq!(
            actual,
            vec![
                ("keep-me".to_string(), "untouched".to_string()),
                ("replace-me".to_string(), "new".to_string()),
            ]
        );
    }

    #[test]
    fn human_bytes_scales_to_binary_units() {
        assert_eq!(human_bytes(0), "0 B");
        assert_eq!(human_bytes(512), "512 B");
        assert_eq!(human_bytes(1023), "1023 B");
        assert_eq!(human_bytes(1024), "1.0 KiB");
        assert_eq!(human_bytes(1536), "1.5 KiB");
        assert_eq!(human_bytes(5 * 1024 * 1024), "5.0 MiB");
        assert_eq!(human_bytes(3 * 1024 * 1024 * 1024), "3.0 GiB");
    }

    #[test]
    fn server_reason_classifies_and_redacts_hub_messages() {
        assert_eq!(
            safe_server_reason("Your push was rejected because it contains files larger than 10 MiB; see https://huggingface.co/docs/hub"),
            Some("file_size_or_lfs".to_string())
        );
        assert_eq!(
            safe_server_reason("You can't create a commit with more than 1000 files"),
            Some("file_count_limit".to_string())
        );
        let redacted =
            safe_server_reason("unexpected failure at /private/data with https://example.invalid and token=secret")
                .unwrap();
        assert!(!redacted.contains("/private/data"));
        assert!(!redacted.contains("https://example.invalid"));
        assert!(!redacted.contains("secret"));
    }

    #[test]
    fn commit_chunks_bound_operations_and_activate_metadata_last() {
        let mut ops = (0..(MAX_COMMIT_OPERATIONS * 2 + 7))
            .map(|i| CommitOperation::add_bytes(format!("chunks.lance/data/{i}.bin"), Bytes::from_static(b"x")))
            .collect::<Vec<_>>();
        // Transactions sort before data in real Hub table paths. Place this first so an
        // unrecognized table prefix cannot accidentally pass by landing in the final chunk.
        ops.insert(
            0,
            CommitOperation::add_bytes("chunks.lance/_transactions/7.txn", Bytes::from_static(b"txn")),
        );
        ops.push(CommitOperation::add_bytes(
            "chunks.lance/_versions/7.manifest",
            Bytes::from_static(b"manifest"),
        ));
        ops.push(CommitOperation::add_bytes(
            "chunks.lance/_versions/latest_version_hint.json",
            Bytes::from_static(b"hint"),
        ));
        ops.push(CommitOperation::add_bytes("README.md", Bytes::from_static(b"readme")));

        let chunks = split_commit_operations(ops).unwrap();
        assert!(chunks.iter().all(|chunk| chunk.len() <= MAX_COMMIT_OPERATIONS));
        let last = chunks.last().unwrap();
        let first_activation = last
            .iter()
            .position(|op| is_activation_path(operation_path(op)))
            .expect("activation metadata must be present in the last chunk");
        assert!(last[first_activation..]
            .iter()
            .all(|op| is_activation_path(operation_path(op))));
        assert!(chunks[..chunks.len() - 1]
            .iter()
            .flatten()
            .all(|op| !is_activation_path(operation_path(op))));
        for path in [
            "chunks.lance/_transactions/7.txn",
            "chunks.lance/_versions/7.manifest",
            "chunks.lance/_versions/latest_version_hint.json",
            "README.md",
        ] {
            assert!(
                last.iter().any(|op| operation_path(op) == path),
                "activation path missing from final chunk: {path}"
            );
        }
    }

    #[test]
    fn commit_chunks_reject_oversized_activation_metadata_before_upload() {
        let ops = (0..=MAX_COMMIT_OPERATIONS)
            .map(|i| {
                CommitOperation::add_bytes(format!("chunks.lance/_transactions/{i}.txn"), Bytes::from_static(b"x"))
            })
            .collect::<Vec<_>>();
        let error = split_commit_operations(ops).unwrap_err();
        assert!(matches!(error, HFError::InvalidParameter(message) if message.contains("activation metadata")));
    }

    #[test]
    fn captured_lance_files_are_sharded_but_repo_metadata_stays_flat() {
        let wrapper = CaptureWrapper {
            captured: Captured::default(),
        };
        for logical in [
            "chunks.lance/data/new.lance",
            "chunks.lance/_versions/10001.manifest",
            "chunks.lance/_transactions/10001.txn",
            "README.md",
        ] {
            wrapper.captured.lock().unwrap().insert(
                object_store::path::Path::from(logical),
                Some(CapturedFile::from_bytes(b"unchanged").unwrap()),
            );
        }
        let files = captured_files(&wrapper);
        assert_eq!(files.len(), 4);
        assert!(files.contains_key("README.md"));
        for logical in [
            "chunks.lance/data/new.lance",
            "chunks.lance/_versions/10001.manifest",
            "chunks.lance/_transactions/10001.txn",
        ] {
            assert!(!files.contains_key(logical));
            assert_eq!(
                files.get(&physical_path(logical).unwrap()).unwrap().bytes(),
                Bytes::from_static(b"unchanged")
            );
        }
    }

    #[test]
    fn sharded_manifest_transaction_and_hint_activate_in_the_final_commit() {
        let mut ops = (0..(MAX_COMMIT_OPERATIONS + 3))
            .map(|i| {
                let logical = format!("chunks.lance/data/{i}.lance");
                CommitOperation::add_bytes(physical_path(&logical).unwrap(), Bytes::from_static(b"data"))
            })
            .collect::<Vec<_>>();
        let activation = [
            "chunks.lance/_transactions/10001.txn",
            "chunks.lance/_versions/10001.manifest",
            "chunks.lance/_versions/_latest.manifest",
        ]
        .map(|logical| physical_path(logical).unwrap());
        for path in &activation {
            ops.insert(
                0,
                CommitOperation::add_bytes(path.clone(), Bytes::from_static(b"metadata")),
            );
        }
        let chunks = split_commit_operations(ops).unwrap();
        assert_eq!(chunks.len(), 2);
        assert!(chunks[0].iter().all(|op| !is_activation_path(operation_path(op))));
        let last = chunks.last().unwrap();
        assert!(activation
            .iter()
            .all(|path| last.iter().any(|op| operation_path(op) == path)));
        assert!(last[last.len() - activation.len()..]
            .iter()
            .all(|op| is_activation_path(operation_path(op))));
    }

    #[tokio::test]
    async fn real_lance_reopens_and_appends_sharded_files_without_changing_flat_history() {
        use futures::TryStreamExt;
        use lance::dataset::builder::DatasetBuilder;
        use lance_io::object_store::ObjectStoreParams;
        use object_store::memory::InMemory;
        use object_store::path::Path as OPath;
        use object_store::{GetOptions, PutOptions, PutPayload};

        let store = Arc::new(InMemory::new());
        let uri = "memory://shard-regression/chunks.lance";
        let batch = |texts: Vec<String>| {
            let schema = Arc::new(Schema::new(vec![Field::new("text", DataType::Utf8, false)]));
            let rows = RecordBatch::try_new(schema.clone(), vec![Arc::new(StringArray::from(texts))]);
            RecordBatchIterator::new([rows], schema)
        };
        let initial = Dataset::write(
            batch(vec!["original raw memory".to_string()]),
            uri,
            Some(WriteParams {
                store_params: Some(ObjectStoreParams {
                    object_store_wrapper: Some(Arc::new(TestStoreWrapper {
                        store: store.clone(),
                        sharded: false,
                        captured: None,
                    })),
                    ..Default::default()
                }),
                ..Default::default()
            }),
        )
        .await
        .unwrap();
        let initial_version = initial.version().version;
        let flat_objects = store.list(None).try_collect::<Vec<_>>().await.unwrap();
        let mut flat_bytes = BTreeMap::new();
        for meta in &flat_objects {
            flat_bytes.insert(
                meta.location.clone(),
                store.get(&meta.location).await.unwrap().bytes().await.unwrap(),
            );
        }

        for expected_rows in 2..=3 {
            let captured = Captured::default();
            let mut ds = dataset::open_wrapped(
                uri,
                HashMap::new(),
                Arc::new(TestStoreWrapper {
                    store: store.clone(),
                    sharded: true,
                    captured: Some(captured.clone()),
                }),
            )
            .await
            .unwrap();
            ds.append(batch(vec![format!("new raw memory {expected_rows}")]), None)
                .await
                .unwrap();
            let files = captured_files(&CaptureWrapper { captured });
            assert!(!files.is_empty());
            assert!(files.keys().all(|path| path.starts_with("__funes_shards__/v1/")));
            for (path, bytes) in &files {
                store
                    .put_opts(
                        &OPath::from(path.as_str()),
                        PutPayload::from(bytes.bytes()),
                        PutOptions::default(),
                    )
                    .await
                    .unwrap();
            }
            let read_wrapper = Arc::new(TestStoreWrapper {
                store: store.clone(),
                sharded: true,
                captured: None,
            });
            let reopened = dataset::open_wrapped(uri, HashMap::new(), read_wrapper.clone())
                .await
                .unwrap();
            assert_eq!(reopened.count_rows(None).await.unwrap(), expected_rows);
            let rows = dataset::scan_rows(&reopened, &["text"], None, None).await.unwrap();
            let actual = rows
                .iter()
                .flat_map(|row| {
                    let text = row.column(0).as_any().downcast_ref::<StringArray>().unwrap();
                    (0..row.num_rows()).map(|i| text.value(i).to_string())
                })
                .collect::<Vec<_>>();
            assert!(actual.contains(&"original raw memory".to_string()));
            assert!(actual.contains(&format!("new raw memory {expected_rows}")));

            let old = DatasetBuilder::from_uri(uri)
                .with_store_params(ObjectStoreParams {
                    object_store_wrapper: Some(read_wrapper),
                    ..Default::default()
                })
                .with_version(initial_version)
                .load()
                .await
                .unwrap();
            assert_eq!(old.count_rows(None).await.unwrap(), 1);
            for (path, bytes) in &flat_bytes {
                assert_eq!(
                    &store
                        .get_opts(path, GetOptions::default())
                        .await
                        .unwrap()
                        .bytes()
                        .await
                        .unwrap(),
                    bytes
                );
            }
        }

        // Compact and clean the mixed layout through the production decorators, without the Hub.
        let captured = Captured::default();
        let mut compacting = dataset::open_wrapped(
            uri,
            HashMap::new(),
            Arc::new(TestStoreWrapper {
                store: store.clone(),
                sharded: true,
                captured: Some(captured.clone()),
            }),
        )
        .await
        .unwrap();
        let read = compacting.version().version;
        refresh(&mut compacting, read, |_| {}).await.unwrap();
        for (path, body) in captured_files(&CaptureWrapper { captured }) {
            store.put(&OPath::from(path), body.bytes().into()).await.unwrap();
        }
        let staged = [
            "chunks.lance/data/staged-flat.lance".to_string(),
            physical_path("chunks.lance/data/staged-shard.lance").unwrap(),
        ];
        for path in &staged {
            store
                .put(&OPath::from(path.as_str()), "pending-upload".into())
                .await
                .unwrap();
        }
        let captured = Captured::default();
        let cleaning = dataset::open_wrapped(
            uri,
            HashMap::new(),
            Arc::new(TestStoreWrapper {
                store: store.clone(),
                sharded: true,
                captured: Some(captured.clone()),
            }),
        )
        .await
        .unwrap();
        let before = objects(store.as_ref()).await;
        delete_old_versions_keeping_uploads(&cleaning).await.unwrap();
        assert_eq!(
            objects(store.as_ref()).await,
            before,
            "cleanup is captured, not forwarded"
        );
        let candidates: BTreeSet<String> = captured
            .lock()
            .unwrap()
            .iter()
            .filter(|(_, body)| body.is_none())
            .flat_map(|(path, _)| std::iter::once(path.to_string()).chain(physical_path(path.as_ref())))
            .collect();
        assert!(candidates.len() <= MAX_COMMIT_OPS);
        let entries: Vec<_> = store.list(None).try_collect::<Vec<_>>().await.unwrap().into_iter()
            .filter(|meta| candidates.contains(meta.location.as_ref()))
            .map(|meta| serde_json::json!({"type":"file", "path":meta.location.to_string(), "size":meta.size, "oid":"blob"}))
            .collect();
        let (repo, requests) = mock_commit_repo(vec![(200, None, serde_json::to_string(&entries).unwrap())]);
        let deletes = captured_deletes(&CaptureWrapper { captured }, &repo, "snapshot")
            .await
            .unwrap();
        assert!(deletes
            .iter()
            .any(|op| operation_path(op).starts_with("__funes_shards__/")));
        assert!(deletes
            .iter()
            .any(|op| !operation_path(op).starts_with("__funes_shards__/")));
        for op in deletes {
            let path = operation_path(&op);
            assert!(!staged.iter().any(|staged| staged == path));
            store.delete(&OPath::from(path)).await.unwrap();
        }
        assert_eq!(requests.join().unwrap().len(), 1);
        let reopened = dataset::open_wrapped(
            uri,
            HashMap::new(),
            Arc::new(TestStoreWrapper {
                store: store.clone(),
                sharded: true,
                captured: None,
            }),
        )
        .await
        .unwrap();
        reopened.validate().await.unwrap();
        assert_eq!(reopened.count_rows(None).await.unwrap(), 3);
        assert_eq!(reopened.versions().await.unwrap().len(), 1);
        let mut search = reopened.scan();
        search
            .full_text_search(FullTextSearchQuery::new("original".to_string()))
            .unwrap()
            .fast_search();
        assert_eq!(search.count_rows().await.unwrap(), 1);
        for path in staged {
            assert!(store.head(&OPath::from(path)).await.is_ok());
        }
    }
}
