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
//! ```text
//!   Lance Dataset → object_store → OpenDAL hf service → HF Hub
//!       put = XET upload + one git commit, per file
//! ```
//!
//! A multi-file write would then be several commits — non-atomic, no CAS. So the op runs through a
//! [`CaptureStore`](super::capture_store::CaptureStore) installed via Lance's
//! [`WrappingObjectStore`] seam: Lance's writes are captured in memory instead of hitting the Hub,
//! and we ship the whole set as a bounded sequence of guarded `create_commit` calls.
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

use std::collections::{BTreeMap, HashMap};
use std::future::Future;
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{ensure, Context, Result};
use arrow_array::{new_null_array, RecordBatch, RecordBatchIterator};
use arrow_schema::{Schema, SchemaRef};
use async_trait::async_trait;
use bytes::Bytes;
use futures::FutureExt;
use hf_hub::progress::{Progress, ProgressEvent, ProgressHandler, UploadEvent};
use hf_hub::repository::{CommitInfo, CommitOperation};
use hf_hub::{HFError, HFRepository, RepoTypeDataset};
use lance::dataset::{Dataset, NewColumnTransform, WriteParams};
use lance::index::DatasetIndexExt;
use lance_index::optimize::OptimizeOptions;
use lance_io::object_store::WrappingObjectStore;
use object_store::ObjectStore as OSObjectStore;
use serde::Serialize;

use super::capture_store::{CaptureStore, Captured};
use super::dataset;
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

fn emit_phase_metric(stage: &'static str, duration: Duration) {
    if !ingest_metrics_enabled() {
        return;
    }
    let metric = IngestPhaseMetric {
        stage,
        duration_ms: round_ms(duration),
    };
    if let Ok(json) = serde_json::to_string(&metric) {
        eprintln!("funes_metric {json}");
    }
}

/// Outcome of an [`append`] commit.
pub(crate) enum Appended {
    /// The data was committed; carries the new commit oid and the resulting unindexed-row backlog.
    Committed { oid: String, unindexed: u64 },
    /// The branch head moved before our commit; the caller may retry against the new head.
    Conflict,
}

/// Outcome of a [`reindex`] commit.
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
/// writes only data — a new fragment, manifest, and transaction — and leaves the new rows
/// unindexed (refresh the index separately with [`reindex`]). `extra_files` (repo path → bytes,
/// e.g. the dataset card) ride the same guarded commit; cloned per attempt, so a conflict retry
/// re-attaches them. Returns [`Appended::Committed`] with the new commit oid and the resulting
/// unindexed-row backlog (the largest across the dataset's indexes — what `push` thresholds on),
/// or [`Appended::Conflict`] if the head moved first — a single attempt against the head it read,
/// so the caller drives the retry.
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
    repo: &HFRepository<RepoTypeDataset>,
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
    let (mut ds, wrapper) = open_capturing(repo, &parent, dataset_uri, storage_options).await?;
    if schema.column_with_name("source_identity").is_some() {
        dataset::ensure_canonical_columns(&mut ds).await?;
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

    // Snapshot the captured writes before optionally reading index stats: `index_statistics` can
    // write a stats migration through the same wrapper, and that must not leak into the data commit.
    let mut files = captured_files(&wrapper);
    // Canonical ingestion discards this result, so it skips the expensive statistics reads.
    let unindexed = if measure_unindexed {
        max_unindexed_rows(&ds).await
    } else {
        0
    };
    for (path, body) in extra_files {
        files.insert(path.clone(), body.clone());
    }

    let (ops, _dir) = write_ops(&files)?;
    match send_commit(commit_repo, ops, parent, rev, message).await {
        Ok(info) => Ok(Appended::Committed {
            oid: info.commit_oid.unwrap_or_else(|| "?".to_string()),
            unindexed,
        }),
        Err(e) if head_moved(&e) => Ok(Appended::Conflict),
        Err(e) => Err(commit_error("data commit failed", &e)),
    }
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
    on_phase: impl Fn(&str),
) -> Result<Option<String>> {
    let staging = tempfile::tempdir()?;
    // Empty prefix = dataset at the repo root; joining "" would leave a stray trailing separator.
    let db_dir = if prefix.is_empty() {
        staging.path().to_path_buf()
    } else {
        staging.path().join(prefix)
    };
    std::fs::create_dir_all(&db_dir)?;
    let table_uri = dataset::table_uri(&db_dir.to_string_lossy());
    let reader = RecordBatchIterator::new(batches.into_iter().map(Ok), schema);
    let mut ds = Dataset::write(reader, &table_uri, Some(WriteParams::default()))
        .await
        .context("building the dataset for first publish")?;
    dataset::build_indexes(&mut ds, on_phase).await?;

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
    let (extra_ops, _extra_dir) = write_ops(extra_files)?;
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
    let table_uri = dataset::table_uri(&db_dir.to_string_lossy());
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
    match send_commit(repo, ops, expected_parent.to_string(), rev, message).await {
        Ok(info) => Ok(Replaced::Committed(info.commit_oid.unwrap_or_else(|| "?".to_string()))),
        Err(e) if head_moved(&e) => Ok(Replaced::Conflict),
        Err(e) => Err(commit_error("canonical data commit failed", &e)),
    }
}

/// Fold an index's delta sub-indexes back into one once this many pile up. Queries fan out across
/// every delta (and per-segment BM25 stats drift), so the pile must stay bounded. Only the deltas
/// are merged — the base is never re-read, which would be the full-index rewrite [`reindex`]
/// exists to avoid.
const COMPACT_DELTAS: usize = 8;

/// Refresh the remote dataset's indexes and land the delta in one `create_commit` on branch `rev`,
/// guarded by the current head. The backlog is appended as a delta sub-index — merging it into the
/// existing index would re-read the whole index over the network — until [`COMPACT_DELTAS`] pile
/// up and the deltas are folded back into one. [`Reindexed::AlreadyCurrent`] if there was nothing
/// to optimize, [`Reindexed::Conflict`] if the head moved first (retry against the new head).
pub(crate) async fn reindex(
    repo: &HFRepository<RepoTypeDataset>,
    dataset_uri: &str,
    storage_options: HashMap<String, String>,
    rev: &str,
    message: String,
) -> Result<Reindexed> {
    let parent = head_oid(repo, rev).await?;
    let (mut ds, wrapper) = open_capturing(repo, &parent, dataset_uri, storage_options).await?;
    dataset::ensure_required_indexes(&mut ds, |_| {}).await?;

    for (name, subs) in sub_index_counts(&ds).await? {
        // subs = base + deltas; merge(deltas) folds every delta into one, sparing the base.
        let deltas = subs - 1;
        let opts = if deltas >= COMPACT_DELTAS {
            eprintln!("  compacting {name} ({deltas} delta sub-indexes)…");
            OptimizeOptions::merge(deltas)
        } else {
            OptimizeOptions::append()
        };
        ds.optimize_indices(&opts.index_names(vec![name]))
            .await
            .context("optimizing the remote index")?;
    }

    let files = captured_files(&wrapper);
    if files.is_empty() {
        return Ok(Reindexed::AlreadyCurrent);
    }
    let (ops, _dir) = write_ops(&files)?;
    match send_commit(repo, ops, parent, rev, message).await {
        Ok(info) => Ok(Reindexed::Committed(info.commit_oid.unwrap_or_else(|| "?".to_string()))),
        Err(e) if head_moved(&e) => Ok(Reindexed::Conflict),
        Err(e) => Err(commit_error("reindex commit failed", &e)),
    }
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
    let (mut ds, wrapper) = open_capturing(repo, &parent, dataset_uri, storage_options).await?;
    ds.add_columns(transform, Some(read_columns), None)
        .await
        .context("adding the remote column")?;
    let files = captured_files(&wrapper);
    ensure!(!files.is_empty(), "add_columns produced no files to commit");
    let (ops, _dir) = write_ops(&files)?;
    let info = send_commit(repo, ops, parent, rev, message)
        .await
        .map_err(|e| commit_error("add_column commit failed", &e))?;
    Ok(info.commit_oid.unwrap_or_else(|| "?".to_string()))
}

/// Atomically replace every split belonging to selected canonical sources. The indexed delete and
/// append run through the same capture store and land in one guarded Hub commit.
#[allow(clippy::too_many_arguments)] // Keep the selected snapshot and CAS target explicit at this write boundary.
pub(crate) async fn replace_documents(
    repo: &HFRepository<RepoTypeDataset>,
    commit_repo: &HFRepository<RepoTypeDataset>,
    dataset_uri: &str,
    storage_options: HashMap<String, String>,
    expected_parent: &str,
    rev: &str,
    message: String,
    batches: Vec<RecordBatch>,
    delete_filter: &str,
) -> Result<Replaced> {
    let (mut ds, wrapper) = open_capturing(repo, expected_parent, dataset_uri, storage_options).await?;
    dataset::ensure_canonical_columns(&mut ds).await?;
    replace_dataset_rows(&mut ds, batches, delete_filter).await?;

    let files = captured_files(&wrapper);
    ensure!(!files.is_empty(), "canonical replace produced no files to commit");
    let (ops, _dir) = write_ops(&files)?;
    match send_commit(commit_repo, ops, expected_parent.to_string(), rev, message).await {
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

/// Open one immutable remote revision with read-through caching inside the write capture. Reads
/// must be wrapped before `load`: otherwise Lance's initial manifest/index requests bypass the
/// cache and can wait indefinitely in the live OpenDAL HF store.
async fn open_capturing(
    repo: &HFRepository<RepoTypeDataset>,
    revision: &str,
    dataset_uri: &str,
    mut storage_options: HashMap<String, String>,
) -> Result<(Dataset, Arc<CaptureWrapper>)> {
    storage_options.insert("revision".to_string(), revision.to_string());
    storage_options.insert("hf_revision".to_string(), revision.to_string());
    let wrapper = Arc::new(CaptureWrapper {
        captured: Captured::default(),
        fetcher: Arc::new(HubFetcher {
            repo: Arc::new(repo.clone()),
            revision: revision.to_string(),
        }),
    });
    let ds = dataset::open_wrapped(
        dataset_uri,
        storage_options,
        wrapper.clone() as Arc<dyn WrappingObjectStore>,
    )
    .await
    .context("opening the remote dataset")?;
    Ok((ds, wrapper))
}

/// The captured writes as physical repo-path → bytes, ready to commit. Existing flat files are
/// unchanged; new Lance files live in deterministic buckets so no directory hits Hub's file cap.
fn captured_files(wrapper: &CaptureWrapper) -> BTreeMap<String, Bytes> {
    let start = Instant::now();
    let files = wrapper
        .captured
        .lock()
        .unwrap()
        .iter()
        .map(|(p, b)| {
            let logical = p.to_string();
            (physical_path(&logical).unwrap_or(logical), b.clone())
        })
        .collect();
    emit_phase_metric("captured_files", start.elapsed());
    files
}

/// The largest `num_unindexed_rows` across the dataset's indexes — how many rows aren't yet folded
/// into an index (and so are answered by a brute-force scan at query time). 0 when there are no
/// indexes. Best-effort: a stats read that errors is skipped rather than failing the caller.
pub(crate) async fn max_unindexed_rows(ds: &Dataset) -> u64 {
    let Ok(indices) = ds.load_indices().await else {
        return 0;
    };
    let mut max = 0u64;
    for idx in indices.iter() {
        if let Ok(json) = ds.index_statistics(&idx.name).await {
            if let Ok(v) = serde_json::from_str::<serde_json::Value>(&json) {
                if let Some(n) = v.get("num_unindexed_rows").and_then(|x| x.as_u64()) {
                    max = max.max(n);
                }
            }
        }
    }
    max
}

/// Sub-index count per index name (the base plus its deltas, which share the index's name), from
/// the index metadata — not `index_statistics`, which can write a stats migration through the
/// capture wrapper.
async fn sub_index_counts(ds: &Dataset) -> Result<Vec<(String, usize)>> {
    let indices = ds.load_indices().await.context("listing the remote indexes")?;
    let mut counts: BTreeMap<String, usize> = BTreeMap::new();
    for idx in indices.iter() {
        *counts.entry(idx.name.clone()).or_default() += 1;
    }
    Ok(counts.into_iter().collect())
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

/// Write captured files (path → bytes) to a scratch dir and turn them into add-file commit
/// operations — hf-hub uploads from local paths. The returned `TempDir` must outlive all chunks.
fn write_ops(files: &BTreeMap<String, Bytes>) -> Result<(Vec<CommitOperation>, tempfile::TempDir)> {
    let start = Instant::now();
    let dir = tempfile::tempdir()?;
    let mut ops = Vec::with_capacity(files.len());
    for (i, (repo_path, body)) in files.iter().enumerate() {
        let local = dir.path().join(format!("f{i}"));
        std::fs::write(&local, body)?;
        ops.push(CommitOperation::add_file(repo_path.clone(), local));
    }
    emit_phase_metric("write_ops", start.elapsed());
    Ok((ops, dir))
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

/// Upload `ops` in bounded chained `create_commit` calls on branch `rev`, guarded by `parent`.
/// The returned result is the final Hub response so callers can tell a head-moved
/// [`HFError::Conflict`] from other failures. Activation metadata is forced into the last chunk.
async fn send_commit(
    repo: &HFRepository<RepoTypeDataset>,
    ops: Vec<CommitOperation>,
    parent: String,
    rev: &str,
    message: String,
) -> std::result::Result<CommitInfo, HFError> {
    send_commit_chunks(repo, ops, Some(parent), rev, message).await
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
                eprintln!("funes_metric {json}");
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
            async {
                let commit = repo
                    .create_commit()
                    .operations(chunk.clone())
                    .commit_message(chunk_message.clone())
                    .revision(rev.to_string())
                    .progress(upload_progress());
                if let Some(expected_parent) = &parent {
                    commit.parent_commit(expected_parent.clone()).send().await
                } else {
                    commit.send().await
                }
            }
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
/// [`HFError::Http`] (only 409 is typed as [`HFError::Conflict`]).
fn head_moved(e: &HFError) -> bool {
    match e {
        HFError::Conflict { .. } => true,
        HFError::Http { context } => context.status.as_u16() == 412,
        _ => false,
    }
}

/// A live stderr byte-bar for an upload `create_commit`, redrawn in place (`\r`) as xet streams the
/// data. Small commits skip the byte phase (no `Progress` events) — then nothing is drawn and the
/// caller's "uploading…" line is the only trace. `Send + Sync`: hf-hub calls it off the main thread.
struct UploadBar;

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
            UploadEvent::Committing => {
                eprint!("\r    committing…                                        ");
                let _ = std::io::stderr().flush();
            }
            UploadEvent::Complete => {
                eprintln!("\r    upload complete                                     ");
            }
            UploadEvent::Start { .. } => {}
        }
    }
}

/// The upload progress handler for `create_commit`, shared by [`send_commit`] and the first-publish
/// commit in [`crate::commands::push`]. See [`UploadBar`].
pub(crate) fn upload_progress() -> Progress {
    Progress::new(UploadBar)
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
    fetcher: Arc<dyn FileFetcher>,
}

impl WrappingObjectStore for CaptureWrapper {
    fn wrap(&self, _prefix: &str, original: Arc<dyn OSObjectStore>) -> Arc<dyn OSObjectStore> {
        let cached = Arc::new(FetchStore::new(original, self.fetcher.clone()));
        let logical = Arc::new(ShardStore::new(cached));
        Arc::new(CaptureStore::new(logical, self.captured.clone()))
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
        self.repo
            .download_file()
            .filename(filename)
            .revision(self.revision.clone())
            .send()
            .await
            .with_context(|| format!("caching {filename}@{}", self.revision))
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
    use arrow_array::StringArray;
    use arrow_schema::{DataType, Field, Schema};
    use lance_index::scalar::{InvertedIndexParams, ScalarIndexParams};
    use lance_index::IndexType;
    use object_store::ObjectStoreExt;

    // Real hf-hub errors, without network credentials or Hub/Voyage calls.
    fn mock_commit_repo(
        responses: Vec<(u16, Option<u64>, &'static str)>,
    ) -> (HFRepository<RepoTypeDataset>, std::thread::JoinHandle<Vec<String>>) {
        use std::io::Read;
        use std::net::TcpListener;

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
                CaptureWrapper {
                    captured: captured.clone(),
                    fetcher: Arc::new(UnusedFetcher),
                }
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
            fetcher: Arc::new(UnusedFetcher),
        };
        wrapper.captured.lock().unwrap().insert(
            object_store::path::Path::from("chunks.lance/data/test.lance"),
            Bytes::from_static(b"metric_test_bytes"),
        );
        let files = captured_files(&wrapper);
        assert_eq!(files.len(), 1);

        let (ops, _dir) = write_ops(&files).unwrap();
        assert_eq!(ops.len(), 1);
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

    /// Pins the Lance behavior [`reindex`] relies on: `append()` adds one delta sub-index per
    /// backlog, and `merge(deltas)` folds the deltas back into one without touching the base.
    #[tokio::test]
    async fn append_optimize_stacks_deltas_and_merge_spares_the_base() {
        let batch = |texts: &[&str]| {
            let schema = Arc::new(Schema::new(vec![Field::new("text", DataType::Utf8, false)]));
            let rows = RecordBatch::try_new(schema.clone(), vec![Arc::new(StringArray::from(texts.to_vec()))]);
            RecordBatchIterator::new([rows], schema)
        };
        let dir = tempfile::tempdir().unwrap();
        let uri = dir.path().join("t.lance");
        let mut ds = Dataset::write(batch(&["alpha bravo"]), uri.to_str().unwrap(), None)
            .await
            .unwrap();
        ds.create_index(
            &["text"],
            IndexType::Inverted,
            None,
            &InvertedIndexParams::default(),
            true,
        )
        .await
        .unwrap();
        assert_eq!(sub_index_counts(&ds).await.unwrap(), vec![("text_idx".to_string(), 1)]);
        let base_uuid = ds.load_indices().await.unwrap()[0].uuid;

        for i in 0..3 {
            ds.append(batch(&[&format!("charlie delta {i}")]), None).await.unwrap();
            ds.optimize_indices(&OptimizeOptions::append()).await.unwrap();
        }
        assert_eq!(sub_index_counts(&ds).await.unwrap(), vec![("text_idx".to_string(), 4)]);

        ds.optimize_indices(&OptimizeOptions::merge(3)).await.unwrap();
        assert_eq!(sub_index_counts(&ds).await.unwrap(), vec![("text_idx".to_string(), 2)]);
        let after = ds.load_indices().await.unwrap();
        assert!(
            after.iter().any(|i| i.uuid == base_uuid),
            "the base index must survive untouched"
        );
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
            fetcher: Arc::new(UnusedFetcher),
        };
        for logical in [
            "chunks.lance/data/new.lance",
            "chunks.lance/_versions/10001.manifest",
            "chunks.lance/_transactions/10001.txn",
            "README.md",
        ] {
            wrapper.captured.lock().unwrap().insert(
                object_store::path::Path::from(logical),
                Bytes::from_static(b"unchanged"),
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
                files.get(&physical_path(logical).unwrap()),
                Some(&Bytes::from_static(b"unchanged"))
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
            let files = captured_files(&CaptureWrapper {
                captured,
                fetcher: Arc::new(UnusedFetcher),
            });
            assert!(!files.is_empty());
            assert!(files.keys().all(|path| path.starts_with("__funes_shards__/v1/")));
            for (path, bytes) in &files {
                store
                    .put_opts(
                        &OPath::from(path.as_str()),
                        PutPayload::from(bytes.clone()),
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
    }
}
