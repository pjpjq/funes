//! Native ingestion for canonical documents. These records enter Lance directly; they are never
//! rendered as synthetic Codex/Pi/Claude transcripts.

use crate::chunk::{self, Chunk};
use crate::hub;
use crate::inference::{self, embed_batched, Embedder};
use crate::memory::dataset::{self, build_batch_for_schema, schema};
use crate::memory::remote::{self, Replaced};
use crate::memory::{lock, Memory, Reachability};
use crate::scan::{self, SecretScanner};
use anyhow::{bail, Context, Result};
use arrow_array::{Array, RecordBatch, RecordBatchIterator, StringArray};
use arrow_schema::Schema;
use chrono::DateTime;
use lance::dataset::{Dataset, MergeInsertBuilder, WhenMatched, WhenNotMatched, WhenNotMatchedBySource, WriteParams};
use serde::Deserialize;
use serde_json::{Map, Value};
use std::borrow::Cow;
use std::collections::{BTreeMap, HashMap};
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::sync::Arc;

const MAX_COMMIT_RETRIES: u32 = 10;

#[derive(Clone, Debug, Deserialize)]
struct InputDocument {
    source_identity: String,
    source_version: String,
    retrieval_text: String,
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
    retrieval_text: String,
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
}

#[derive(Default, Debug)]
struct Report {
    sources: usize,
    chunks: usize,
    unchanged: usize,
    stale: usize,
    held: usize,
    commit: Option<String>,
}

#[allow(dead_code)] // both variants exist only to hold their advisory lock until drop
enum LocalWriteLock {
    Default(lock::MemoryLock),
    Explicit(File),
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
        format!(
            "ingested sources={} chunks={} unchanged={} stale={} held={}{}\n",
            self.sources, self.chunks, self.unchanged, self.stale, self.held, commit
        )
    }
}

/// Ingest canonical JSONL into a local or Hugging Face memory.
pub async fn run(input: &Path, memory: Memory) -> Result<String> {
    let docs = read_documents(input)?;
    let scanner = scan::Trufflehog::find()?;
    let (docs, held) = secret_gate(docs, &scanner)?;
    let mut embedder = inference::embedder()?;
    let report = match memory {
        Memory::Local { path } => ingest_local(path, &docs, held, embedder.as_mut()).await?,
        remote_memory @ Memory::Remote { .. } => ingest_remote(remote_memory, &docs, held, embedder.as_mut()).await?,
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
    if input.retrieval_text.trim().is_empty() {
        bail!("retrieval_text must not be empty");
    }
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
        retrieval_text: input.retrieval_text,
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

fn secret_gate(docs: Vec<Document>, scanner: &dyn SecretScanner) -> Result<(Vec<Document>, usize)> {
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
    let mut held = 0usize;
    for (doc, found) in docs.into_iter().zip(dirty) {
        if !found {
            clean.push(doc);
        } else {
            held += 1;
        }
    }
    Ok((clean, held))
}

fn persisted_revision_fields(doc: &Document) -> Vec<Cow<'_, str>> {
    let mut fields = vec![
        Cow::Borrowed(doc.retrieval_text.as_str()),
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
            Some(current) if revision_order(doc) <= stored_revision_order(current) => selection.stale += 1,
            _ => selection.changed.push(doc),
        }
    }
    selection
}

async fn stored_revisions(ds: &Dataset) -> Result<HashMap<String, StoredRevision>> {
    let arrow = Schema::from(ds.schema());
    for name in [
        "source_identity",
        "source_version",
        "updated_at",
        "content_hash",
        "metadata_json",
    ] {
        if arrow.column_with_name(name).is_none() {
            return Ok(HashMap::new());
        }
    }
    let batches = dataset::scan_rows(
        ds,
        &[
            "source_identity",
            "source_version",
            "updated_at",
            "content_hash",
            "metadata_json",
        ],
        Some("source_identity IS NOT NULL"),
        None,
    )
    .await?;
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
    chunk::split_document(&doc.retrieval_text)
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

fn delete_filter(selection: &[&Document]) -> String {
    let identities = selection
        .iter()
        .map(|doc| format!("'{}'", doc.source_identity.replace('\'', "''")))
        .collect::<Vec<_>>()
        .join(", ");
    format!("source_identity IN ({identities})")
}

async fn ingest_local(path: PathBuf, docs: &[Document], held: usize, embedder: &mut dyn Embedder) -> Result<Report> {
    std::fs::create_dir_all(&path).with_context(|| format!("creating memory at {}", path.display()))?;
    let _lock = acquire_local_lock(&path)?;
    let uri = dataset::table_uri(&path.to_string_lossy());
    let mut ds = dataset::open(&uri, HashMap::new()).await.ok();
    let stored = match &ds {
        Some(dataset) => stored_revisions(dataset).await?,
        None => HashMap::new(),
    };
    let selection = select_documents(docs, &stored);
    if selection.changed.is_empty() {
        return Ok(Report {
            unchanged: selection.unchanged,
            stale: selection.stale,
            held,
            ..Report::default()
        });
    }

    let chunks = build_chunks(&selection.changed);
    let vectors = embed_chunks(&chunks, embedder)?;
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
        let batch = build_batch_for_schema(&chunks, &vectors, schema())?;
        let reader = RecordBatchIterator::new(vec![Ok(batch)], schema());
        let mut created = Dataset::write(reader, &uri, Some(WriteParams::default()))
            .await
            .context("creating canonical document memory")?;
        dataset::build_indexes(&mut created, |_| {}).await;
    }
    Ok(Report {
        sources: selection.changed.len(),
        chunks: chunks.len(),
        unchanged: selection.unchanged,
        stale: selection.stale,
        held,
        commit: None,
    })
}

async fn ingest_remote(memory: Memory, docs: &[Document], held: usize, embedder: &mut dyn Embedder) -> Result<Report> {
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
    loop {
        let expected_parent = remote::head_oid(&repo, &rev).await?;
        let (stored, first) = match memory.open_remote_revision(&expected_parent).await {
            Ok(ds) => (stored_revisions(&ds).await?, false),
            Err(error) if crate::memory::dataset_absent(&error) => (HashMap::new(), true),
            Err(error) => {
                return Err(error.context(format!(
                    "{} exists but cannot be read; refusing to replace canonical sources",
                    memory.label()
                )))
            }
        };
        let selection = select_documents(docs, &stored);
        if selection.changed.is_empty() {
            return Ok(Report {
                unchanged: selection.unchanged,
                stale: selection.stale,
                held,
                ..Report::default()
            });
        }
        let chunks = build_chunks(&selection.changed);
        let vectors = embed_chunks(&chunks, embedder)?;
        let batch = build_batch_for_schema(&chunks, &vectors, schema())?;
        let message = format!("funes ingest-docs: {} source revision(s)", selection.changed.len());
        let result = if first {
            remote::first_document_publish(&repo, &prefix, vec![batch], schema(), &expected_parent, &rev, message)
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
                    held,
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
            "retrieval_text": text,
            "content_hash": format!("hash-{version}"),
            "updated_at": updated_at,
            "source_agent": "codex",
            "source_type": "agents_md",
            "session_id": "original-session",
            "project": "demo",
            "repo": "acme/demo",
            "device_id": "device-hash",
            "content_type": "agents_md",
            "metadata": metadata,
            "raw_text": "must never be stored"
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
        let mut embedder = FakeEmbedder;

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
        let first = ingest_local(memory.clone(), &docs, held, &mut embedder).await.unwrap();
        assert!(first.chunks > 1);
        let before = dataset::open(&dataset::table_uri(&memory.to_string_lossy()), HashMap::new())
            .await
            .unwrap()
            .version();

        // Same source revision is a true no-op: no Lance commit/version churn.
        let same = ingest_local(memory.clone(), &docs, held, &mut embedder).await.unwrap();
        let after = dataset::open(&dataset::table_uri(&memory.to_string_lossy()), HashMap::new())
            .await
            .unwrap()
            .version();
        assert_eq!(same.unchanged, 1);
        assert_eq!(before.version, after.version);

        // Newer, shorter text deletes every old tail split and updates metadata without raw text.
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
        let second = ingest_local(memory.clone(), &docs, held, &mut embedder).await.unwrap();
        assert_eq!(second.sources, 1);
        let stored = rows(&memory).await;
        assert_eq!(texts(&stored), vec!["short replacement"]);
        assert_eq!(strings(&stored, "source_version"), vec![Some("v2".to_string())]);
        assert_eq!(strings(&stored, "source_agent"), vec![Some("codex".to_string())]);
        assert_eq!(
            strings(&stored, "metadata_json"),
            vec![Some(r#"{"label":"after"}"#.to_string())]
        );
        let encoded = format!("{stored:?}");
        assert!(!encoded.contains("must never be stored"));
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
        let third = ingest_local(memory.clone(), &docs, 0, &mut embedder).await.unwrap();
        assert_eq!(third.sources, 1);
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
        let stale = ingest_local(memory.clone(), &docs, 0, &mut embedder).await.unwrap();
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
        let held_report = ingest_local(memory.clone(), &docs, held, &mut embedder).await.unwrap();
        assert_eq!(held_report.held, 1);
        assert_eq!(texts(&rows(&memory).await), vec!["short replacement"]);
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
        assert_eq!(held, 1);
    }

    #[test]
    fn secret_gate_scans_multiline_source_bytes_without_json_escaping() {
        let marker = "review-secret-begin\nreview-secret-end";
        let input: InputDocument =
            serde_json::from_value(doc("v1", "2026-09-03T00:00:00Z", marker, serde_json::json!({}))).unwrap();
        let (clean, held) = secret_gate(vec![normalize(input).unwrap()], &Contains(marker)).unwrap();
        assert!(clean.is_empty());
        assert_eq!(held, 1);
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
        let full = schema();
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
        ingest_local(memory.clone(), &docs, 0, &mut embedder).await.unwrap();
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
        let doc = normalize(input).unwrap();
        let old = HashMap::from([(
            doc.source_identity.clone(),
            StoredRevision {
                source_version: "v1".to_string(),
                updated_key: timestamp_key("2026-09-01T00:00:00Z").unwrap(),
                content_hash: "hash-v1".to_string(),
                metadata_json: "{}".to_string(),
            },
        )]);
        assert_eq!(select_documents(std::slice::from_ref(&doc), &old).changed.len(), 1);

        let mut conflicts = 0;
        record_conflict(&mut conflicts).unwrap();
        // This is what the production loop sees after reopening the head another writer moved.
        let moved = HashMap::from([(doc.source_identity.clone(), stored(&doc))]);
        let retry = select_documents(std::slice::from_ref(&doc), &moved);
        assert!(retry.changed.is_empty());
        assert_eq!(retry.unchanged, 1);
        assert_eq!(conflicts, 1);
    }
}
