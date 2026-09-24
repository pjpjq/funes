//! Native ingestion for canonical documents. These records enter Lance directly; they are never
//! rendered as synthetic Codex/Pi/Claude transcripts.

use crate::chunk::{self, Chunk};
use crate::hub;
use crate::inference::{self, embed_batched, Embedder, EmbeddingProfile};
use crate::memory::dataset::{self, build_batch_for_schema, schema_for};
use crate::memory::remote::{self, Replaced};
use crate::memory::{lock, Memory, Reachability};
use crate::scan::{self, SecretScanner};
use anyhow::{bail, Context, Result};
use arrow_array::{Array, FixedSizeListArray, Float32Array, RecordBatch, RecordBatchIterator, StringArray};
use arrow_schema::Schema;
use chrono::DateTime;
use futures::TryStreamExt;
use lance::dataset::{Dataset, MergeInsertBuilder, WhenMatched, WhenNotMatched, WhenNotMatchedBySource, WriteParams};
use serde::Deserialize;
use serde_json::{Map, Value};
use sha2::{Digest, Sha256};
use std::borrow::Cow;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::sync::Arc;

const MAX_COMMIT_RETRIES: u32 = 10;
const HELD_SOURCE_ID_DOMAIN: &[u8] = b"funes-held-source-v1\0";
const EMBEDDING_GENERATION_PREFIX: &str = "~funes-eg-v1:";
const STORED_REVISION_COLUMNS: [&str; 5] = [
    "source_identity",
    "source_version",
    "updated_at",
    "content_hash",
    "metadata_json",
];

#[derive(Clone, Debug, Deserialize)]
struct InputDocument {
    source_identity: String,
    source_version: String,
    #[serde(default)]
    raw_text: Option<String>,
    #[serde(default)]
    retrieval_text: Option<String>,
    content_hash: String,
    updated_at: String,
    #[serde(default)]
    metadata: Value,
    #[serde(default)]
    source_agent: Option<String>,
    #[serde(default)]
    source_type: Option<String>,
    #[serde(default)]
    project: Option<String>,
    #[serde(default)]
    repo: Option<String>,
    #[serde(default)]
    device_id: Option<String>,
    #[serde(default)]
    content_type: Option<String>,
    #[serde(default)]
    source_missing: Option<bool>,
    #[serde(default)]
    session_id: Option<String>,
    #[serde(default)]
    role: Option<String>,
    #[serde(default)]
    timestamp: Option<String>,
    #[serde(default)]
    source_path: Option<String>,
    #[serde(default)]
    worktree: Option<String>,
    #[serde(default)]
    message_id: Option<String>,
    #[serde(default)]
    ordinal: Option<i64>,
}

#[derive(Clone, Debug)]
struct Document {
    source_identity: String,
    source_version: String,
    raw_text: String,
    content_hash: String,
    updated_at: String,
    updated_key: i128,
    metadata_json: String,
    source_agent: Option<String>,
    source_type: Option<String>,
    project: Option<String>,
    repo: Option<String>,
    device_id: Option<String>,
    content_type: Option<String>,
    source_missing: Option<bool>,
    session_id: Option<String>,
    role: Option<String>,
    timestamp: Option<String>,
    source_path: Option<String>,
    worktree: Option<String>,
    message_id: Option<String>,
    ordinal: Option<i64>,
}

#[derive(Clone, Debug)]
struct StoredRevision {
    source_version: String,
    updated_key: i128,
    content_hash: String,
    metadata_json: String,
}

#[derive(Default, Debug)]
struct Selection<'a> {
    changed: Vec<&'a Document>,
    unchanged: usize,
    stale: usize,
    stale_source_ids: Vec<String>,
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
struct EmbeddingCacheKey {
    profile_fingerprint: String,
    embedding_generation: u64,
    chunk_id: String,
    text: String,
}

#[derive(Default, Debug)]
struct Report {
    sources: usize,
    chunks: usize,
    unchanged: usize,
    stale: usize,
    held_source_ids: Vec<String>,
    stale_source_ids: Vec<String>,
    commit: Option<String>,
}

#[allow(dead_code)] // both variants exist only to hold their advisory lock until drop
enum LocalWriteLock {
    Default(lock::MemoryLock),
    Explicit(lock::FileLock),
}

fn acquire_local_lock(path: &Path) -> Result<LocalWriteLock> {
    if path == Path::new(&dataset::local_memory_dir()) {
        return lock::MemoryLock::acquire().map(LocalWriteLock::Default);
    }
    lock::try_lock_file(&path.join(".funes-write.lock"))?
        .map(LocalWriteLock::Explicit)
        .ok_or_else(|| anyhow::anyhow!("another funes memory operation is in progress; retry once it finishes"))
}

impl Report {
    fn render(&self) -> String {
        let commit = self
            .commit
            .as_ref()
            .map(|oid| format!(" commit={oid}"))
            .unwrap_or_default();
        // A source identity is itself secret-scanned, so never echo it.  The stable digest lets
        // callers classify a mixed clean/held batch without bisecting and re-running embeddings.
        let held_source_ids = if self.held_source_ids.is_empty() {
            String::new()
        } else {
            let encoded =
                serde_json::to_string(&self.held_source_ids).expect("serializing source-id digests cannot fail");
            format!(" held_source_ids={encoded}")
        };
        let stale_source_ids = if self.stale_source_ids.is_empty() {
            String::new()
        } else {
            let encoded = serde_json::to_string(&self.stale_source_ids)
                .expect("serializing stale source-id digests cannot fail");
            format!(" stale_source_ids={encoded}")
        };
        format!(
            "ingested sources={} chunks={} unchanged={} stale={} held={}{}{}{}\n",
            self.sources,
            self.chunks,
            self.unchanged,
            self.stale,
            self.held_source_ids.len(),
            commit,
            held_source_ids,
            stale_source_ids,
        )
    }
}

/// Ingest canonical JSONL into a local or Hugging Face memory.
pub async fn run(input: &Path, memory: Memory) -> Result<String> {
    let docs = read_documents(input)?;
    let scanner = scan::Trufflehog::find()?;
    let (docs, held_source_ids) = secret_gate(docs, &scanner)?;
    let profile = inference::embedding_profile()?;
    let mut embedder = inference::embedder_for(&profile)?;
    let report = match memory {
        Memory::Local { path } => ingest_local(path, &docs, &held_source_ids, embedder.as_mut(), &profile).await?,
        remote_memory @ Memory::Remote { .. } => {
            ingest_remote(remote_memory, &docs, &held_source_ids, embedder.as_mut(), &profile).await?
        }
    };
    Ok(report.render())
}

fn read_documents(input: &Path) -> Result<Vec<Document>> {
    let file = File::open(input).with_context(|| format!("opening canonical JSONL at {}", input.display()))?;
    let mut by_identity: BTreeMap<String, Document> = BTreeMap::new();
    for (line_no, line) in BufReader::new(file).lines().enumerate() {
        let line = line.with_context(|| format!("reading canonical JSONL line {}", line_no + 1))?;
        if line.trim().is_empty() {
            continue;
        }
        let input: InputDocument = serde_json::from_str(&line)
            .with_context(|| format!("invalid canonical JSONL record at line {}", line_no + 1))?;
        let doc =
            normalize(input).with_context(|| format!("invalid canonical JSONL record at line {}", line_no + 1))?;
        match by_identity.get(&doc.source_identity) {
            Some(current) if revision_order(&doc) <= revision_order(current) => {}
            _ => {
                by_identity.insert(doc.source_identity.clone(), doc);
            }
        }
    }
    if by_identity.is_empty() {
        bail!("canonical JSONL contains no records");
    }
    Ok(by_identity.into_values().collect())
}

fn normalize(input: InputDocument) -> Result<Document> {
    for (name, value) in [
        ("source_identity", input.source_identity.as_str()),
        ("source_version", input.source_version.as_str()),
        ("content_hash", input.content_hash.as_str()),
        ("updated_at", input.updated_at.as_str()),
    ] {
        if value.trim().is_empty() {
            bail!("{name} must not be empty");
        }
        if value.contains('\0') {
            bail!("{name} must not contain NUL");
        }
    }
    embedding_generation(&input.source_version)?;
    // `raw_text` is the canonical source.  Accept the legacy `retrieval_text`
    // envelope only so an interrupted rollout can replay an older queued
    // batch; new producers always send raw text.
    let raw_text = clean_opt(input.raw_text)
        .or_else(|| clean_opt(input.retrieval_text))
        .context("raw_text must not be empty")?;
    let updated_key = timestamp_key(&input.updated_at)?;
    let metadata = canonical_metadata(input.metadata)?;
    let meta_string = |name: &str| {
        metadata
            .get(name)
            .and_then(Value::as_str)
            .map(str::to_string)
            .filter(|value| !value.trim().is_empty())
    };
    let pick = |direct: Option<String>, name: &str| clean_opt(direct).or_else(|| meta_string(name));
    let source_missing = input
        .source_missing
        .or_else(|| metadata.get("source_missing").and_then(Value::as_bool));
    Ok(Document {
        source_identity: input.source_identity,
        source_version: input.source_version,
        raw_text,
        content_hash: input.content_hash,
        updated_at: input.updated_at,
        updated_key,
        metadata_json: serde_json::to_string(&metadata)?,
        source_agent: pick(input.source_agent, "source_agent"),
        source_type: pick(input.source_type, "source_type"),
        project: pick(input.project, "project"),
        repo: pick(input.repo, "repo"),
        device_id: pick(input.device_id, "device_id"),
        content_type: pick(input.content_type, "content_type"),
        source_missing,
        session_id: pick(input.session_id, "session_id"),
        role: pick(input.role, "role"),
        timestamp: pick(input.timestamp, "timestamp"),
        source_path: pick(input.source_path, "source_path"),
        worktree: pick(input.worktree, "worktree"),
        message_id: pick(input.message_id, "message_id"),
        ordinal: input
            .ordinal
            .or_else(|| metadata.get("ordinal").and_then(Value::as_i64)),
    })
}

fn canonical_metadata(value: Value) -> Result<Map<String, Value>> {
    let object = match value {
        Value::Null => Map::new(),
        Value::Object(object) => object,
        _ => bail!("metadata must be an object or null"),
    };
    let Value::Object(canonical) = sort_and_strip(Value::Object(object)) else {
        unreachable!("object stays an object")
    };
    Ok(canonical)
}

/// Canonicalize object key order and guarantee raw source text never lands in `metadata_json`.
fn sort_and_strip(value: Value) -> Value {
    match value {
        Value::Object(object) => {
            let mut entries: Vec<_> = object.into_iter().filter(|(key, _)| key != "raw_text").collect();
            entries.sort_by(|left, right| left.0.cmp(&right.0));
            Value::Object(
                entries
                    .into_iter()
                    .map(|(key, value)| (key, sort_and_strip(value)))
                    .collect(),
            )
        }
        Value::Array(values) => Value::Array(values.into_iter().map(sort_and_strip).collect()),
        scalar => scalar,
    }
}

fn clean_opt(value: Option<String>) -> Option<String> {
    value.filter(|value| !value.trim().is_empty())
}

fn embedding_generation(source_version: &str) -> Result<u64> {
    let Some(encoded) = source_version.strip_prefix(EMBEDDING_GENERATION_PREFIX) else {
        return Ok(0);
    };
    let (generation, revision) = encoded
        .split_once(':')
        .context("reserved embedding generation source_version is malformed")?;
    if generation.len() != 20 || !generation.bytes().all(|byte| byte.is_ascii_digit()) || revision.is_empty() {
        bail!("reserved embedding generation source_version is malformed");
    }
    generation
        .parse()
        .context("reserved embedding generation source_version is out of range")
}

fn timestamp_key(value: &str) -> Result<i128> {
    if let Ok(seconds) = value.parse::<f64>() {
        if seconds.is_finite() {
            return Ok((seconds * 1_000_000.0).round() as i128);
        }
    }
    DateTime::parse_from_rfc3339(value)
        .map(|stamp| stamp.timestamp_micros() as i128)
        .with_context(|| "updated_at must be RFC3339 or Unix seconds")
}

fn revision_order(doc: &Document) -> (i128, &str, &str, &str) {
    (
        doc.updated_key,
        &doc.source_version,
        &doc.content_hash,
        &doc.metadata_json,
    )
}

fn stored_revision_order(revision: &StoredRevision) -> (i128, &str, &str, &str) {
    (
        revision.updated_key,
        &revision.source_version,
        &revision.content_hash,
        &revision.metadata_json,
    )
}

fn opaque_source_id(source_identity: &str) -> String {
    let mut digest = Sha256::new();
    digest.update(HELD_SOURCE_ID_DOMAIN);
    digest.update(source_identity.as_bytes());
    format!("sha256:{}", hex::encode(digest.finalize()))
}

fn secret_gate(docs: Vec<Document>, scanner: &dyn SecretScanner) -> Result<(Vec<Document>, Vec<String>)> {
    // Scan every persisted field without JSON-escaping its bytes. A finding in text, metadata, an
    // origin path, or any facet holds back every split, so a shorter dirty revision cannot delete
    // a clean tail.
    let source_revisions: Vec<Vec<Cow<'_, str>>> = docs.iter().map(persisted_revision_fields).collect();
    let mut texts = Vec::new();
    let mut owners = Vec::new();
    for (owner, fields) in source_revisions.iter().enumerate() {
        for field in fields {
            texts.push(field.as_ref());
            owners.push(owner);
        }
    }
    let findings = scan::scan_blocks(&texts, scanner)?;
    let mut dirty = vec![false; docs.len()];
    for (owner, found) in owners.into_iter().zip(findings) {
        dirty[owner] |= !found.is_empty();
    }
    let mut clean = Vec::with_capacity(docs.len());
    let mut held_source_ids = Vec::new();
    for (doc, found) in docs.into_iter().zip(dirty) {
        if !found {
            clean.push(doc);
        } else {
            held_source_ids.push(opaque_source_id(&doc.source_identity));
        }
    }
    held_source_ids.sort_unstable();
    Ok((clean, held_source_ids))
}

fn persisted_revision_fields(doc: &Document) -> Vec<Cow<'_, str>> {
    let mut fields = vec![
        Cow::Borrowed(doc.raw_text.as_str()),
        Cow::Borrowed(doc.source_identity.as_str()),
        Cow::Borrowed(doc.source_version.as_str()),
        Cow::Borrowed(doc.content_hash.as_str()),
        Cow::Borrowed(doc.updated_at.as_str()),
        Cow::Borrowed(doc.metadata_json.as_str()),
    ];
    fields.extend(
        [
            doc.source_agent.as_deref(),
            doc.source_type.as_deref(),
            doc.project.as_deref(),
            doc.repo.as_deref(),
            doc.device_id.as_deref(),
            doc.content_type.as_deref(),
            doc.session_id.as_deref(),
            doc.role.as_deref(),
            doc.timestamp.as_deref(),
            doc.source_path.as_deref(),
            doc.worktree.as_deref(),
            doc.message_id.as_deref(),
        ]
        .into_iter()
        .flatten()
        .map(Cow::Borrowed),
    );
    if let Some(source_missing) = doc.source_missing {
        fields.push(Cow::Borrowed(if source_missing { "true" } else { "false" }));
    }
    if let Some(ordinal) = doc.ordinal {
        fields.push(Cow::Owned(ordinal.to_string()));
    }
    fields
}

fn select_documents<'a>(docs: &'a [Document], stored: &HashMap<String, StoredRevision>) -> Selection<'a> {
    let mut selection = Selection::default();
    for doc in docs {
        match stored.get(&doc.source_identity) {
            Some(current) if current.source_version == doc.source_version => selection.unchanged += 1,
            Some(current) if revision_order(doc) <= stored_revision_order(current) => {
                selection.stale += 1;
                selection
                    .stale_source_ids
                    .push(opaque_source_id(&doc.source_identity));
            }
            _ => selection.changed.push(doc),
        }
    }
    selection
}

fn append_schema_allows_fast_path(schema: &Schema) -> bool {
    schema.column_with_name("source_identity").is_none()
        || STORED_REVISION_COLUMNS
            .iter()
            .all(|name| schema.column_with_name(name).is_some())
}

fn append_only(
    selection: &Selection<'_>,
    stored: &HashMap<String, StoredRevision>,
    schema_allows_fast_path: bool,
) -> bool {
    schema_allows_fast_path
        && !selection.changed.is_empty()
        && selection
            .changed
            .iter()
            .all(|doc| !stored.contains_key(&doc.source_identity))
}

async fn stored_revisions(ds: &Dataset, docs: &[Document]) -> Result<HashMap<String, StoredRevision>> {
    if docs.is_empty() {
        return Ok(HashMap::new());
    }
    let arrow = Schema::from(ds.schema());
    for name in STORED_REVISION_COLUMNS {
        if arrow.column_with_name(name).is_none() {
            return Ok(HashMap::new());
        }
    }
    let mut stream = stored_revisions_scan(ds, docs)?.try_into_stream().await?;
    let mut batches = Vec::new();
    while let Some(batch) = stream.try_next().await? {
        batches.push(batch);
    }
    let mut revisions: HashMap<String, StoredRevision> = HashMap::new();
    for batch in batches {
        let identities = string_col(&batch, "source_identity")?;
        let versions = string_col(&batch, "source_version")?;
        let updated = string_col(&batch, "updated_at")?;
        let hashes = string_col(&batch, "content_hash")?;
        let metadata = string_col(&batch, "metadata_json")?;
        for row in 0..batch.num_rows() {
            if identities.is_null(row) {
                continue;
            }
            if versions.is_null(row) || updated.is_null(row) || hashes.is_null(row) || metadata.is_null(row) {
                bail!("stored canonical source has incomplete revision metadata");
            }
            let revision = StoredRevision {
                source_version: versions.value(row).to_string(),
                updated_key: timestamp_key(updated.value(row))
                    .context("stored canonical source has invalid updated_at")?,
                content_hash: hashes.value(row).to_string(),
                metadata_json: metadata.value(row).to_string(),
            };
            match revisions.get(identities.value(row)) {
                Some(existing) if stored_revision_order(existing) != stored_revision_order(&revision) => {
                    bail!("stored canonical source has inconsistent split revisions")
                }
                Some(_) => {}
                None => {
                    revisions.insert(identities.value(row).to_string(), revision);
                }
            }
        }
    }
    Ok(revisions)
}

fn stored_revisions_scan(ds: &Dataset, docs: &[Document]) -> Result<lance::dataset::scanner::Scanner> {
    // A rebuild sends bounded source batches. Read only those identities through the scalar index
    // instead of downloading every fragment on every batch, which would make a backfill O(n²).
    let identities = docs
        .iter()
        .map(|doc| format!("'{}'", doc.source_identity.replace('\'', "''")))
        .collect::<Vec<_>>()
        .join(", ");
    let filter = format!("source_identity IN ({identities})");
    let mut scan = ds.scan();
    scan.project(&[
        "source_identity",
        "source_version",
        "updated_at",
        "content_hash",
        "metadata_json",
    ])?;
    scan.filter(&filter)?;
    Ok(scan)
}

fn string_col<'a>(batch: &'a RecordBatch, name: &str) -> Result<&'a StringArray> {
    batch
        .column_by_name(name)
        .and_then(|column| column.as_any().downcast_ref::<StringArray>())
        .with_context(|| format!("memory column {name:?} is not utf8"))
}

fn document_chunks(doc: &Document) -> Vec<Chunk> {
    let session_id = doc.session_id.clone().unwrap_or_else(|| doc.source_identity.clone());
    let turn_uuid = doc.message_id.clone().unwrap_or_else(|| doc.source_identity.clone());
    let workdir = doc.worktree.clone().or_else(|| doc.project.clone()).unwrap_or_default();
    let harness = doc.source_agent.clone().unwrap_or_else(|| "canonical".to_string());
    let ts = doc.timestamp.clone().unwrap_or_else(|| doc.updated_at.clone());
    chunk::split_document(&doc.raw_text)
        .into_iter()
        .enumerate()
        .map(|(split_idx, text)| Chunk {
            id: chunk::canonical_cid(&doc.source_identity, split_idx as i64),
            text,
            session_id: session_id.clone(),
            workdir: workdir.clone(),
            turn_uuid: turn_uuid.clone(),
            parent_uuid: None,
            seq: doc.ordinal.unwrap_or(0),
            ts: ts.clone(),
            role: doc.role.clone().unwrap_or_else(|| "document".to_string()),
            block_type: "text".to_string(),
            tool_name: None,
            source_path: doc.source_path.clone().unwrap_or_default(),
            block_idx: 0,
            split_idx: split_idx as i64,
            harness: harness.clone(),
            repo: doc.repo.clone(),
            source_identity: Some(doc.source_identity.clone()),
            source_version: Some(doc.source_version.clone()),
            content_hash: Some(doc.content_hash.clone()),
            updated_at: Some(doc.updated_at.clone()),
            source_agent: doc.source_agent.clone(),
            source_type: doc.source_type.clone(),
            project: doc.project.clone(),
            device_id: doc.device_id.clone(),
            content_type: doc.content_type.clone(),
            source_missing: doc.source_missing,
            metadata_json: Some(doc.metadata_json.clone()),
        })
        .collect()
}

fn build_chunks(selection: &[&Document]) -> Vec<Chunk> {
    selection.iter().flat_map(|doc| document_chunks(doc)).collect()
}

fn embed_chunks(chunks: &[Chunk], embedder: &mut dyn Embedder) -> Result<Vec<Vec<f32>>> {
    let texts: Vec<&str> = chunks.iter().map(|chunk| chunk.text.as_str()).collect();
    embed_batched(embedder, &texts, |_| {})
}

fn embedding_cache_key(chunk: &Chunk, profile: &EmbeddingProfile) -> EmbeddingCacheKey {
    EmbeddingCacheKey {
        profile_fingerprint: profile.fingerprint.clone(),
        embedding_generation: embedding_generation(chunk.source_version.as_deref().unwrap_or_default())
            .expect("canonical source versions are validated before chunking"),
        chunk_id: chunk.id.clone(),
        text: chunk.text.clone(),
    }
}

async fn seed_stored_embeddings(
    ds: &Dataset,
    selection: &[&Document],
    profile: &EmbeddingProfile,
    cache: &mut HashMap<EmbeddingCacheKey, Vec<f32>>,
) -> Result<()> {
    if selection.is_empty() {
        return Ok(());
    }
    let schema = Schema::from(ds.schema());
    if ["source_identity", "source_version", "id", "text", "vector"]
        .iter()
        .any(|name| schema.column_with_name(name).is_none())
    {
        return Ok(());
    }
    let filter = delete_filter(selection);
    for batch in dataset::scan_rows(ds, &["id", "text", "source_version", "vector"], Some(&filter), None).await? {
        let ids = string_col(&batch, "id")?;
        let texts = string_col(&batch, "text")?;
        let source_versions = string_col(&batch, "source_version")?;
        let vectors = batch
            .column_by_name("vector")
            .and_then(|column| column.as_any().downcast_ref::<FixedSizeListArray>())
            .context("memory column \"vector\" is not a fixed-size list")?;
        for row in 0..batch.num_rows() {
            if ids.is_null(row) || texts.is_null(row) || source_versions.is_null(row) || vectors.is_null(row) {
                continue;
            }
            let Ok(embedding_generation) = embedding_generation(source_versions.value(row)) else {
                continue;
            };
            let values = vectors.value(row);
            let Some(values) = values.as_any().downcast_ref::<Float32Array>() else {
                continue;
            };
            if values.len() != profile.dimensions || values.null_count() > 0 {
                continue;
            }
            let vector = (0..values.len()).map(|index| values.value(index)).collect::<Vec<_>>();
            if vector.iter().any(|value| !value.is_finite()) {
                continue;
            }
            cache.insert(
                EmbeddingCacheKey {
                    profile_fingerprint: profile.fingerprint.clone(),
                    embedding_generation,
                    chunk_id: ids.value(row).to_string(),
                    text: texts.value(row).to_string(),
                },
                vector,
            );
        }
    }
    Ok(())
}

fn cached_selection_embeddings(
    selection: &[&Document],
    profile: &EmbeddingProfile,
    embedder: &mut dyn Embedder,
    cache: &mut HashMap<EmbeddingCacheKey, Vec<f32>>,
) -> Result<(Vec<Chunk>, Vec<Vec<f32>>)> {
    let chunks = build_chunks(selection);
    let mut scheduled = HashSet::new();
    let mut missing = Vec::new();
    for chunk in &chunks {
        let key = embedding_cache_key(chunk, profile);
        if !cache.contains_key(&key) && scheduled.insert(key.clone()) {
            missing.push((key, chunk.clone()));
        }
    }

    let missing_chunks = missing.iter().map(|(_, chunk)| chunk.clone()).collect::<Vec<_>>();
    let missing_vectors = embed_chunks(&missing_chunks, embedder)?;
    if missing_vectors.len() != missing_chunks.len() {
        bail!(
            "embedding provider returned {} vectors for {} chunks",
            missing_vectors.len(),
            missing_chunks.len()
        );
    }
    for ((key, _), vector) in missing.into_iter().zip(missing_vectors) {
        cache.insert(key, vector);
    }

    let vectors = chunks
        .iter()
        .map(|chunk| {
            cache
                .get(&embedding_cache_key(chunk, profile))
                .cloned()
                .context("embedding cache omitted a selected chunk")
        })
        .collect::<Result<Vec<_>>>()?;
    Ok((chunks, vectors))
}

fn delete_filter(selection: &[&Document]) -> String {
    let identities = selection
        .iter()
        .map(|doc| format!("'{}'", doc.source_identity.replace('\'', "''")))
        .collect::<Vec<_>>()
        .join(", ");
    format!("source_identity IN ({identities})")
}

async fn ingest_local(
    path: PathBuf,
    docs: &[Document],
    held_source_ids: &[String],
    embedder: &mut dyn Embedder,
    profile: &EmbeddingProfile,
) -> Result<Report> {
    std::fs::create_dir_all(&path).with_context(|| format!("creating memory at {}", path.display()))?;
    let _lock = acquire_local_lock(&path)?;
    let uri = dataset::table_uri(&path.to_string_lossy());
    let mut ds = dataset::open(&uri, HashMap::new()).await.ok();
    if let Some(current) = &ds {
        crate::memory::check_compat_with_profile(current, profile)?;
    }
    let stored = match &ds {
        Some(dataset) => stored_revisions(dataset, docs).await?,
        None => HashMap::new(),
    };
    let selection = select_documents(docs, &stored);
    if selection.changed.is_empty() {
        return Ok(Report {
            unchanged: selection.unchanged,
            stale: selection.stale,
            held_source_ids: held_source_ids.to_vec(),
            stale_source_ids: selection.stale_source_ids.clone(),
            ..Report::default()
        });
    }

    let mut embedding_cache = HashMap::new();
    if let Some(current) = &ds {
        seed_stored_embeddings(current, &selection.changed, profile, &mut embedding_cache).await?;
    }
    let (chunks, vectors) = cached_selection_embeddings(&selection.changed, profile, embedder, &mut embedding_cache)?;
    if let Some(current) = &mut ds {
        dataset::ensure_canonical_columns(current).await?;
        let target_schema = Arc::new(Schema::from(current.schema()));
        let batch = build_batch_for_schema(&chunks, &vectors, target_schema)?;
        let delete = WhenNotMatchedBySource::delete_if(current, &delete_filter(&selection.changed))?;
        let mut builder = MergeInsertBuilder::try_new(Arc::new(current.clone()), vec!["id".to_string()])?;
        builder
            .when_matched(WhenMatched::UpdateAll)
            .when_not_matched(WhenNotMatched::InsertAll)
            .when_not_matched_by_source(delete)
            .conflict_retries(0);
        builder
            .try_build()?
            .execute_batches(vec![batch])
            .await
            .context("replacing canonical document rows")?;
    } else {
        let target_schema = schema_for(profile);
        let batch = build_batch_for_schema(&chunks, &vectors, target_schema.clone())?;
        let reader = RecordBatchIterator::new(vec![Ok(batch)], target_schema);
        let mut created = Dataset::write(reader, &uri, Some(WriteParams::default()))
            .await
            .context("creating canonical document memory")?;
        dataset::build_indexes(&mut created, |_| {}).await?;
    }
    Ok(Report {
        sources: selection.changed.len(),
        chunks: chunks.len(),
        unchanged: selection.unchanged,
        stale: selection.stale,
        held_source_ids: held_source_ids.to_vec(),
        stale_source_ids: selection.stale_source_ids.clone(),
        commit: None,
    })
}

async fn ingest_remote(
    memory: Memory,
    docs: &[Document],
    held_source_ids: &[String],
    embedder: &mut dyn Embedder,
    profile: &EmbeddingProfile,
) -> Result<Report> {
    let Memory::Remote { uri } = &memory else {
        unreachable!("remote ingestion receives a remote memory")
    };
    let (owner, name, prefix) = hub::parse_hf(uri)?;
    let token = hub::hf_token().context("no HF token (set HF_TOKEN) — required to ingest")?;
    let repo = hub::client(Some(&token), true)?.dataset(owner, name);
    let rev = "main".to_string();
    let dataset_uri = dataset::table_uri(uri);
    let opts = HashMap::from([("hf_token".to_string(), token)]);

    match crate::memory::remote_reachability(uri).await {
        Reachability::Ok => {}
        Reachability::Missing => return Err(memory.missing_error()),
        Reachability::Offline => bail!("{} is unreachable", memory.label()),
    }

    let mut conflicts = 0u32;
    let mut embedding_cache = HashMap::new();
    loop {
        let expected_parent = remote::head_oid(&repo, &rev).await?;
        let (current, first, schema_allows_append) = match memory.open_remote_revision(&expected_parent).await {
            Ok(ds) => {
                let arrow = Schema::from(ds.schema());
                let schema_allows_append = append_schema_allows_fast_path(&arrow);
                (Some(ds), false, schema_allows_append)
            }
            Err(error) if crate::memory::dataset_absent(&error) => (None, true, true),
            Err(error) => {
                return Err(error.context(format!(
                    "{} exists but cannot be read; refusing to replace canonical sources",
                    memory.label()
                )))
            }
        };
        let stored = match &current {
            Some(ds) => stored_revisions(ds, docs).await?,
            None => HashMap::new(),
        };
        let selection = select_documents(docs, &stored);
        if selection.changed.is_empty() {
            return Ok(Report {
                unchanged: selection.unchanged,
                stale: selection.stale,
                held_source_ids: held_source_ids.to_vec(),
                stale_source_ids: selection.stale_source_ids.clone(),
                ..Report::default()
            });
        }
        if let Some(ds) = &current {
            seed_stored_embeddings(ds, &selection.changed, profile, &mut embedding_cache).await?;
        }
        let (chunks, vectors) =
            cached_selection_embeddings(&selection.changed, profile, embedder, &mut embedding_cache)?;
        let target_schema = schema_for(profile);
        let batch = build_batch_for_schema(&chunks, &vectors, target_schema.clone())?;
        let message = format!("funes ingest-docs: {} source revision(s)", selection.changed.len());
        let result = if first {
            remote::first_document_publish(
                &repo,
                &prefix,
                vec![batch],
                target_schema,
                &expected_parent,
                &rev,
                message,
            )
            .await?
        } else if append_only(&selection, &stored, schema_allows_append) {
            remote::append_documents(
                &repo,
                &dataset_uri,
                opts.clone(),
                &expected_parent,
                &rev,
                message,
                vec![batch],
                target_schema,
            )
            .await?
        } else {
            remote::replace_documents(
                &repo,
                &dataset_uri,
                opts.clone(),
                &expected_parent,
                &rev,
                message,
                vec![batch],
                &delete_filter(&selection.changed),
            )
            .await?
        };
        match result {
            Replaced::Committed(oid) => {
                return Ok(Report {
                    sources: selection.changed.len(),
                    chunks: chunks.len(),
                    unchanged: selection.unchanged,
                    stale: selection.stale,
                    held_source_ids: held_source_ids.to_vec(),
                    stale_source_ids: selection.stale_source_ids.clone(),
                    commit: Some(oid),
                })
            }
            Replaced::Conflict => {
                record_conflict(&mut conflicts)?;
                // The next iteration reopens the moved head and re-evaluates every source version.
            }
        }
    }
}

fn record_conflict(conflicts: &mut u32) -> Result<()> {
    *conflicts += 1;
    if *conflicts > MAX_COMMIT_RETRIES {
        bail!("canonical data commit kept conflicting after {MAX_COMMIT_RETRIES} retries")
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::dataset::DIM;
    use crate::scan::Finding;

    struct Clean;

    impl SecretScanner for Clean {
        fn scan(&self, texts: &[&str]) -> Result<Vec<Vec<Finding>>> {
            Ok(texts.iter().map(|_| Vec::new()).collect())
        }
    }

    struct FakeEmbedder;

    impl Embedder for FakeEmbedder {
        fn embed(&mut self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
            Ok(texts.iter().map(|_| vec![0.0; DIM as usize]).collect())
        }
    }

    #[derive(Default)]
    struct CountingEmbedder {
        calls: usize,
        texts: usize,
    }

    impl Embedder for CountingEmbedder {
        fn embed(&mut self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
            self.calls += 1;
            self.texts += texts.len();
            Ok(texts.iter().map(|_| vec![0.0; DIM as usize]).collect())
        }
    }

    struct Dirty;

    impl SecretScanner for Dirty {
        fn scan(&self, texts: &[&str]) -> Result<Vec<Vec<Finding>>> {
            Ok(texts
                .iter()
                .map(|_| {
                    vec![Finding {
                        detector: "test".to_string(),
                        raw: "not-logged".to_string(),
                        decoder: "PLAIN".to_string(),
                    }]
                })
                .collect())
        }
    }

    struct Contains(&'static str);

    impl SecretScanner for Contains {
        fn scan(&self, texts: &[&str]) -> Result<Vec<Vec<Finding>>> {
            Ok(texts
                .iter()
                .map(|text| {
                    text.contains(self.0)
                        .then(|| Finding {
                            detector: "test".to_string(),
                            raw: self.0.to_string(),
                            decoder: "PLAIN".to_string(),
                        })
                        .into_iter()
                        .collect()
                })
                .collect())
        }
    }

    fn write_docs(path: &Path, docs: &[Value]) {
        let body = docs.iter().map(Value::to_string).collect::<Vec<_>>().join("\n");
        std::fs::write(path, format!("{body}\n")).unwrap();
    }

    fn doc(version: &str, updated_at: &str, text: &str, metadata: Value) -> Value {
        serde_json::json!({
            "source_identity": "stable-doc",
            "source_version": version,
            "raw_text": text,
            "retrieval_text": "legacy English shadow must not be stored",
            "content_hash": format!("hash-{version}"),
            "updated_at": updated_at,
            "source_agent": "codex",
            "source_type": "agents_md",
            "session_id": "original-session",
            "project": "demo",
            "repo": "acme/demo",
            "device_id": "device-hash",
            "content_type": "agents_md",
            "metadata": metadata
        })
    }

    fn stored(doc: &Document) -> StoredRevision {
        StoredRevision {
            source_version: doc.source_version.clone(),
            updated_key: doc.updated_key,
            content_hash: doc.content_hash.clone(),
            metadata_json: doc.metadata_json.clone(),
        }
    }

    async fn rows(memory: &Path) -> Vec<RecordBatch> {
        let ds = dataset::open(&dataset::table_uri(&memory.to_string_lossy()), HashMap::new())
            .await
            .unwrap();
        dataset::scan_rows(&ds, &[], None, None).await.unwrap()
    }

    fn texts(rows: &[RecordBatch]) -> Vec<String> {
        rows.iter()
            .flat_map(|batch| {
                let text = batch
                    .column_by_name("text")
                    .unwrap()
                    .as_any()
                    .downcast_ref::<StringArray>()
                    .unwrap();
                (0..batch.num_rows()).map(|row| text.value(row).to_string())
            })
            .collect()
    }

    fn strings(rows: &[RecordBatch], name: &str) -> Vec<Option<String>> {
        rows.iter()
            .flat_map(|batch| {
                let values = batch
                    .column_by_name(name)
                    .unwrap()
                    .as_any()
                    .downcast_ref::<StringArray>()
                    .unwrap();
                (0..batch.num_rows()).map(|row| (!values.is_null(row)).then(|| values.value(row).to_string()))
            })
            .collect()
    }

    #[tokio::test]
    async fn canonical_revisions_replace_atomically_and_dedupe() {
        let root = tempfile::tempdir().unwrap();
        let memory = root.path().join("memory");
        let input = root.path().join("docs.jsonl");
        let clean = Clean;
        let mut embedder = CountingEmbedder::default();
        let profile = EmbeddingProfile::local();

        let long = format!("{} old-tail", "long source text ".repeat(180));
        write_docs(
            &input,
            &[doc(
                "v1",
                "2026-09-01T00:00:00Z",
                &long,
                serde_json::json!({"raw_text":"hidden", "label":"before"}),
            )],
        );
        let docs = read_documents(&input).unwrap();
        let (docs, held) = secret_gate(docs, &clean).unwrap();
        let first = ingest_local(memory.clone(), &docs, &held, &mut embedder, &profile)
            .await
            .unwrap();
        assert!(first.chunks > 1);
        let stored_ds = dataset::open(&dataset::table_uri(&memory.to_string_lossy()), HashMap::new())
            .await
            .unwrap();
        let plan = stored_revisions_scan(&stored_ds, &docs)
            .unwrap()
            .explain_plan(false)
            .await
            .unwrap();
        assert!(
            plan.contains("ScalarIndexQuery"),
            "stored revision lookup must use source_identity_idx: {plan}"
        );
        let before = stored_ds.version();

        // Same source revision is a true no-op: no Lance commit/version churn.
        let same = ingest_local(memory.clone(), &docs, &held, &mut embedder, &profile)
            .await
            .unwrap();
        let after = dataset::open(&dataset::table_uri(&memory.to_string_lossy()), HashMap::new())
            .await
            .unwrap()
            .version();
        assert_eq!(same.unchanged, 1);
        assert_eq!(before.version, after.version);

        // Newer, shorter raw text deletes every old tail split and updates metadata.
        write_docs(
            &input,
            &[doc(
                "v2",
                "2026-09-02T00:00:00Z",
                "short replacement",
                serde_json::json!({"raw_text":"still hidden", "label":"after"}),
            )],
        );
        let docs = read_documents(&input).unwrap();
        let (docs, held) = secret_gate(docs, &clean).unwrap();
        let second = ingest_local(memory.clone(), &docs, &held, &mut embedder, &profile)
            .await
            .unwrap();
        assert_eq!(second.sources, 1);
        let embedded_before_metadata_only = embedder.texts;
        let stored = rows(&memory).await;
        assert_eq!(texts(&stored), vec!["short replacement"]);
        assert_eq!(strings(&stored, "source_version"), vec![Some("v2".to_string())]);
        assert_eq!(strings(&stored, "source_agent"), vec![Some("codex".to_string())]);
        assert_eq!(
            strings(&stored, "metadata_json"),
            vec![Some(r#"{"label":"after"}"#.to_string())]
        );
        let encoded = format!("{stored:?}");
        assert!(!encoded.contains("legacy English shadow must not be stored"));
        assert!(!encoded.contains("still hidden"));

        // A metadata-only revision still updates, while an older timestamp cannot roll it back.
        write_docs(
            &input,
            &[doc(
                "v3",
                "2026-09-03T00:00:00Z",
                "short replacement",
                serde_json::json!({"label":"metadata-only"}),
            )],
        );
        let docs = read_documents(&input).unwrap();
        let third = ingest_local(memory.clone(), &docs, &[], &mut embedder, &profile)
            .await
            .unwrap();
        assert_eq!(third.sources, 1);
        assert_eq!(
            embedder.texts, embedded_before_metadata_only,
            "metadata-only revisions must reuse the stored vector"
        );
        assert_eq!(
            strings(&rows(&memory).await, "metadata_json"),
            vec![Some(r#"{"label":"metadata-only"}"#.to_string())]
        );
        let got = crate::commands::recall::get(
            Memory::Local { path: memory.clone() },
            "stable-doc".to_string(),
            crate::commands::recall::TurnRange::default(),
        )
        .await
        .unwrap();
        assert!(got.contains("short replacement"));
        write_docs(
            &input,
            &[doc(
                "v4",
                "2026-09-01T00:00:00Z",
                "stale replacement",
                serde_json::json!({}),
            )],
        );
        let docs = read_documents(&input).unwrap();
        let stale = ingest_local(memory.clone(), &docs, &[], &mut embedder, &profile)
            .await
            .unwrap();
        assert_eq!(stale.stale, 1);
        assert_eq!(texts(&rows(&memory).await), vec!["short replacement"]);

        // A dirty new source revision is held as a whole and cannot evict the clean stored one.
        write_docs(
            &input,
            &[doc(
                "v5",
                "2026-09-04T00:00:00Z",
                "dirty revision",
                serde_json::json!({}),
            )],
        );
        let docs = read_documents(&input).unwrap();
        let (docs, held) = secret_gate(docs, &Dirty).unwrap();
        assert!(docs.is_empty());
        let held_report = ingest_local(memory.clone(), &docs, &held, &mut embedder, &profile)
            .await
            .unwrap();
        assert_eq!(held_report.held_source_ids.len(), 1);
        assert_eq!(texts(&rows(&memory).await), vec!["short replacement"]);
    }

    #[tokio::test]
    async fn local_ingest_reuses_matching_chunks_and_embeds_only_changed_chunks() {
        let root = tempfile::tempdir().unwrap();
        let memory = root.path().join("memory");
        let profile = EmbeddingProfile::local();
        let mut embedder = CountingEmbedder::default();
        let first_text = format!("{}\n{}", "A".repeat(800), "B".repeat(600));
        let first: InputDocument = serde_json::from_value(doc(
            "v1",
            "2026-09-01T00:00:00Z",
            &first_text,
            serde_json::json!({"label":"first"}),
        ))
        .unwrap();
        let first = normalize(first).unwrap();
        assert_eq!(document_chunks(&first).len(), 2);
        ingest_local(
            memory.clone(),
            std::slice::from_ref(&first),
            &[],
            &mut embedder,
            &profile,
        )
        .await
        .unwrap();
        assert_eq!(embedder.texts, 2);

        let second: InputDocument = serde_json::from_value(doc(
            "v2",
            "2026-09-02T00:00:00Z",
            &first_text,
            serde_json::json!({"label":"metadata-only"}),
        ))
        .unwrap();
        let second = normalize(second).unwrap();
        ingest_local(
            memory.clone(),
            std::slice::from_ref(&second),
            &[],
            &mut embedder,
            &profile,
        )
        .await
        .unwrap();
        assert_eq!(embedder.texts, 2);
        assert_eq!(
            strings(&rows(&memory).await, "source_version"),
            vec![Some("v2".to_string()), Some("v2".to_string())]
        );

        let changed_text = format!("{}\n{}", "A".repeat(800), "C".repeat(600));
        let third: InputDocument = serde_json::from_value(doc(
            "v3",
            "2026-09-03T00:00:00Z",
            &changed_text,
            serde_json::json!({"label":"one-chunk-changed"}),
        ))
        .unwrap();
        let third = normalize(third).unwrap();
        ingest_local(
            memory.clone(),
            std::slice::from_ref(&third),
            &[],
            &mut embedder,
            &profile,
        )
        .await
        .unwrap();
        assert_eq!(embedder.texts, 3);
        assert_eq!(
            texts(&rows(&memory).await),
            document_chunks(&third)
                .into_iter()
                .map(|chunk| chunk.text)
                .collect::<Vec<_>>()
        );
    }

    #[tokio::test]
    async fn explicit_embedding_generation_reembeds_unchanged_text_once() {
        let root = tempfile::tempdir().unwrap();
        let memory = root.path().join("memory");
        let profile = EmbeddingProfile::local();
        let mut embedder = CountingEmbedder::default();
        let text = "same canonical source text";

        let mut initial = doc(
            "0a-initial",
            "2026-09-01T00:00:00Z",
            text,
            serde_json::json!({"label":"initial"}),
        );
        initial["content_hash"] = Value::String("stable-content-hash".to_string());
        let initial = normalize(serde_json::from_value(initial).unwrap()).unwrap();
        ingest_local(
            memory.clone(),
            std::slice::from_ref(&initial),
            &[],
            &mut embedder,
            &profile,
        )
        .await
        .unwrap();
        assert_eq!(embedder.texts, 1);

        let mut forced = doc(
            "~funes-eg-v1:00000000000000000001:forced",
            "2026-09-01T00:00:00Z",
            text,
            serde_json::json!({"label":"forced"}),
        );
        forced["content_hash"] = Value::String("stable-content-hash".to_string());
        let forced = normalize(serde_json::from_value(forced).unwrap()).unwrap();
        ingest_local(
            memory.clone(),
            std::slice::from_ref(&forced),
            &[],
            &mut embedder,
            &profile,
        )
        .await
        .unwrap();
        assert_eq!(
            embedder.texts, 2,
            "a new explicit embedding generation must not reuse the old vector"
        );

        let mut metadata_only = doc(
            "~funes-eg-v1:00000000000000000001:metadata-only",
            "2026-09-02T00:00:00Z",
            text,
            serde_json::json!({"label":"metadata-only"}),
        );
        metadata_only["content_hash"] = Value::String("stable-content-hash".to_string());
        let metadata_only = normalize(serde_json::from_value(metadata_only).unwrap()).unwrap();
        ingest_local(
            memory.clone(),
            std::slice::from_ref(&metadata_only),
            &[],
            &mut embedder,
            &profile,
        )
        .await
        .unwrap();
        assert_eq!(
            embedder.texts, 2,
            "metadata-only revisions inside one generation must reuse the vector"
        );
    }

    #[test]
    fn append_only_requires_every_changed_source_to_be_absent() {
        let current: InputDocument =
            serde_json::from_value(doc("v1", "2026-09-01T00:00:00Z", "current", serde_json::json!({}))).unwrap();
        let current = normalize(current).unwrap();
        let mut new_value = doc("v1", "2026-09-02T00:00:00Z", "new", serde_json::json!({}));
        new_value["source_identity"] = Value::String("new-source".to_string());
        let new = normalize(serde_json::from_value(new_value).unwrap()).unwrap();
        let stored = HashMap::from([(current.source_identity.clone(), stored(&current))]);

        let new_selection = select_documents(std::slice::from_ref(&new), &stored);
        assert!(append_only(&new_selection, &stored, true));
        assert!(!append_only(&new_selection, &stored, false));

        let update: InputDocument =
            serde_json::from_value(doc("v2", "2026-09-03T00:00:00Z", "update", serde_json::json!({}))).unwrap();
        let update = normalize(update).unwrap();
        let mixed_docs = [new, update];
        let mixed = select_documents(&mixed_docs, &stored);
        assert!(!append_only(&mixed, &stored, true));

        let complete = schema_for(&EmbeddingProfile::local());
        assert!(append_schema_allows_fast_path(complete.as_ref()));
        let partial = Schema::new(
            complete
                .fields()
                .iter()
                .filter(|field| field.name() != "metadata_json")
                .cloned()
                .collect::<Vec<_>>(),
        );
        assert!(!append_schema_allows_fast_path(&partial));
        let legacy = Schema::new(
            complete
                .fields()
                .iter()
                .filter(|field| field.name() != "source_identity")
                .cloned()
                .collect::<Vec<_>>(),
        );
        assert!(append_schema_allows_fast_path(&legacy));
    }

    #[tokio::test]
    async fn unindexed_append_is_visible_to_revision_lookup() {
        let root = tempfile::tempdir().unwrap();
        let memory = root.path().join("memory");
        let initial: InputDocument =
            serde_json::from_value(doc("v1", "2026-09-01T00:00:00Z", "initial", serde_json::json!({}))).unwrap();
        let initial = normalize(initial).unwrap();
        let profile = EmbeddingProfile::local();
        let mut embedder = FakeEmbedder;
        ingest_local(
            memory.clone(),
            std::slice::from_ref(&initial),
            &[],
            &mut embedder,
            &profile,
        )
        .await
        .unwrap();

        let mut appended_value = doc(
            "v1",
            "2026-09-02T00:00:00Z",
            "unindexed appended row",
            serde_json::json!({}),
        );
        appended_value["source_identity"] = Value::String("appended-source".to_string());
        let appended = normalize(serde_json::from_value(appended_value).unwrap()).unwrap();
        let chunks = document_chunks(&appended);
        let vectors = vec![vec![0.0; DIM as usize]; chunks.len()];
        let uri = dataset::table_uri(&memory.to_string_lossy());
        let mut ds = dataset::open(&uri, HashMap::new()).await.unwrap();
        let schema = Arc::new(Schema::from(ds.schema()));
        let batch = build_batch_for_schema(&chunks, &vectors, schema.clone()).unwrap();
        ds.append(RecordBatchIterator::new(vec![Ok(batch)], schema), None)
            .await
            .unwrap();

        let revisions = stored_revisions(&ds, std::slice::from_ref(&appended)).await.unwrap();
        assert_eq!(
            revisions.get("appended-source").unwrap().source_version,
            appended.source_version
        );
        let selection = select_documents(std::slice::from_ref(&appended), &revisions);
        assert_eq!(selection.unchanged, 1);
        assert!(selection.changed.is_empty());
    }

    #[test]
    fn duplicate_batch_selection_is_order_stable() {
        let root = tempfile::tempdir().unwrap();
        let a = root.path().join("a.jsonl");
        let b = root.path().join("b.jsonl");
        let old = doc("v1", "2026-09-01T00:00:00Z", "old", serde_json::json!({}));
        let new = doc("v2", "2026-09-02T00:00:00Z", "new", serde_json::json!({}));
        write_docs(&a, &[old.clone(), new.clone()]);
        write_docs(&b, &[new, old]);
        let left = read_documents(&a).unwrap();
        let right = read_documents(&b).unwrap();
        assert_eq!(left.len(), 1);
        assert_eq!(left[0].source_version, "v2");
        assert_eq!(revision_order(&left[0]), revision_order(&right[0]));
    }

    #[test]
    fn legacy_retrieval_text_is_only_an_input_fallback() {
        let mut value = doc("v1", "2026-09-01T00:00:00Z", "authoritative raw", serde_json::json!({}));
        value.as_object_mut().unwrap().remove("raw_text");
        value["retrieval_text"] = Value::String("legacy queued text".to_string());
        let input: InputDocument = serde_json::from_value(value).unwrap();
        assert_eq!(normalize(input).unwrap().raw_text, "legacy queued text");
    }

    #[test]
    fn embedding_generation_prefix_is_explicit_and_strict() {
        assert_eq!(embedding_generation("legacy-source-version").unwrap(), 0);
        assert_eq!(
            embedding_generation("~funes-eg-v1:00000000000000000042:revision").unwrap(),
            42
        );
        assert!(embedding_generation("~funes-eg-v1:42:revision").is_err());
        assert!(embedding_generation("~funes-eg-v1:00000000000000000042:").is_err());
    }

    #[test]
    fn equal_timestamp_revisions_converge_and_facets_pass_the_secret_gate() {
        let newer: InputDocument =
            serde_json::from_value(doc("v2", "2026-09-02T00:00:00Z", "winner", serde_json::json!({}))).unwrap();
        let older: InputDocument =
            serde_json::from_value(doc("v1", "2026-09-02T00:00:00Z", "loser", serde_json::json!({}))).unwrap();
        let newer = normalize(newer).unwrap();
        let older = normalize(older).unwrap();
        let current = HashMap::from([(newer.source_identity.clone(), stored(&newer))]);
        let retry = select_documents(std::slice::from_ref(&older), &current);
        assert!(retry.changed.is_empty());
        assert_eq!(retry.stale, 1);

        let mut value = doc("v3", "2026-09-03T00:00:00Z", "clean retrieval", serde_json::json!({}));
        value["source_path"] = Value::String("facet-secret-marker".to_string());
        let input: InputDocument = serde_json::from_value(value).unwrap();
        let (clean, held) = secret_gate(vec![normalize(input).unwrap()], &Contains("facet-secret-marker")).unwrap();
        assert!(clean.is_empty());
        assert_eq!(held.len(), 1);
    }

    #[test]
    fn secret_gate_scans_multiline_source_bytes_without_json_escaping() {
        let marker = "review-secret-begin\nreview-secret-end";
        let input: InputDocument =
            serde_json::from_value(doc("v1", "2026-09-03T00:00:00Z", marker, serde_json::json!({}))).unwrap();
        let (clean, held) = secret_gate(vec![normalize(input).unwrap()], &Contains(marker)).unwrap();
        assert!(clean.is_empty());
        assert_eq!(held.len(), 1);
    }

    #[test]
    fn held_report_uses_a_stable_opaque_source_id() {
        assert_eq!(
            Report::default().render(),
            "ingested sources=0 chunks=0 unchanged=0 stale=0 held=0\n"
        );
        let identity = "identity-with-secret-marker";
        let mut value = doc("v1", "2026-09-03T00:00:00Z", "otherwise clean", serde_json::json!({}));
        value["source_identity"] = Value::String(identity.to_string());
        let input: InputDocument = serde_json::from_value(value).unwrap();
        let (clean, held_source_ids) =
            secret_gate(vec![normalize(input).unwrap()], &Contains("secret-marker")).unwrap();
        assert!(clean.is_empty());
        assert_eq!(
            held_source_ids,
            vec!["sha256:1cfefd9638aac282a2a112dc1cfa62a18376b837315a9b67f6687f697168af9c"]
        );

        let rendered = Report {
            held_source_ids,
            ..Report::default()
        }
        .render();
        assert!(rendered.contains("held=1 held_source_ids=[\"sha256:"));
        assert!(!rendered.contains(identity));
    }

    #[tokio::test]
    async fn ingest_migrates_an_old_transcript_schema_without_rewriting_its_provenance() {
        let root = tempfile::tempdir().unwrap();
        let memory = root.path().join("memory");
        std::fs::create_dir_all(&memory).unwrap();
        let mut legacy = document_chunks(
            &normalize(
                serde_json::from_value(doc(
                    "legacy",
                    "2026-09-01T00:00:00Z",
                    "legacy transcript row",
                    serde_json::json!({}),
                ))
                .unwrap(),
            )
            .unwrap(),
        )
        .remove(0);
        legacy.id = "legacy-transcript-id".to_string();
        legacy.source_identity = None;
        legacy.source_version = None;
        legacy.content_hash = None;
        legacy.updated_at = None;
        let full = schema_for(&EmbeddingProfile::local());
        let old_schema = Arc::new(Schema::new_with_metadata(
            full.fields()[..17]
                .iter()
                .map(|field| field.as_ref().clone())
                .collect::<Vec<_>>(),
            full.metadata().clone(),
        ));
        let old_batch = build_batch_for_schema(&[legacy], &[vec![0.0; DIM as usize]], old_schema.clone()).unwrap();
        let reader = RecordBatchIterator::new(vec![Ok(old_batch)], old_schema);
        Dataset::write(
            reader,
            &dataset::table_uri(&memory.to_string_lossy()),
            Some(WriteParams::default()),
        )
        .await
        .unwrap();

        let input = root.path().join("docs.jsonl");
        write_docs(
            &input,
            &[doc(
                "v1",
                "2026-09-02T00:00:00Z",
                "canonical row",
                serde_json::json!({}),
            )],
        );
        let docs = read_documents(&input).unwrap();
        let mut embedder = FakeEmbedder;
        let profile = EmbeddingProfile::local();
        ingest_local(memory.clone(), &docs, &[], &mut embedder, &profile)
            .await
            .unwrap();
        let stored = rows(&memory).await;
        let identities = strings(&stored, "source_identity");
        assert_eq!(identities.len(), 2);
        assert_eq!(identities.iter().filter(|value| value.is_none()).count(), 1);
        assert_eq!(
            identities
                .iter()
                .filter(|value| value.as_deref() == Some("stable-doc"))
                .count(),
            1
        );
    }

    #[test]
    fn cas_conflict_path_reloads_and_reclassifies_the_moved_head() {
        let input: InputDocument =
            serde_json::from_value(doc("v2", "2026-09-02T00:00:00Z", "new", serde_json::json!({}))).unwrap();
        let candidate = normalize(input).unwrap();
        let old = HashMap::from([(
            candidate.source_identity.clone(),
            StoredRevision {
                source_version: "v1".to_string(),
                updated_key: timestamp_key("2026-09-01T00:00:00Z").unwrap(),
                content_hash: "hash-v1".to_string(),
                metadata_json: "{}".to_string(),
            },
        )]);
        let first = select_documents(std::slice::from_ref(&candidate), &old);
        assert_eq!(first.changed.len(), 1);

        let profile = EmbeddingProfile::local();
        let mut embedder = CountingEmbedder::default();
        let mut cache = HashMap::new();
        let first_embedded = cached_selection_embeddings(&first.changed, &profile, &mut embedder, &mut cache).unwrap();
        assert_eq!(embedder.calls, 1);
        assert_eq!(embedder.texts, first_embedded.0.len());

        let mut conflicts = 0;
        record_conflict(&mut conflicts).unwrap();
        // An unrelated head movement keeps this source selected, but the same source/profile/
        // version/hash must reuse its paid embedding rather than call Voyage again.
        let retry_same = select_documents(std::slice::from_ref(&candidate), &old);
        let retry_embedded =
            cached_selection_embeddings(&retry_same.changed, &profile, &mut embedder, &mut cache).unwrap();
        assert_eq!(retry_embedded.0.len(), first_embedded.0.len());
        assert_eq!(retry_embedded.1, first_embedded.1);
        assert_eq!(embedder.calls, 1);
        assert_eq!(embedder.texts, first_embedded.0.len());

        // This is what the production loop sees after reopening the head another writer moved.
        let moved = HashMap::from([(candidate.source_identity.clone(), stored(&candidate))]);
        let retry = select_documents(std::slice::from_ref(&candidate), &moved);
        assert!(retry.changed.is_empty());
        assert_eq!(retry.unchanged, 1);

        let next: InputDocument =
            serde_json::from_value(doc("v3", "2026-09-03T00:00:00Z", "newer", serde_json::json!({}))).unwrap();
        let next = normalize(next).unwrap();
        let next_selection = select_documents(std::slice::from_ref(&next), &moved);
        let previous_texts = embedder.texts;
        cached_selection_embeddings(&next_selection.changed, &profile, &mut embedder, &mut cache).unwrap();
        assert_eq!(embedder.calls, 2);
        assert!(embedder.texts > previous_texts);
        assert_eq!(conflicts, 1);
    }
}
