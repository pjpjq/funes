//! Shared memory helpers: the local memory location, the `chunks` table's schema and rows, opening a
//! dataset, plain scans, and building the FTS/IVF indexes. funes's home is `$FUNES_HOME`/`~/.funes` —
//! it holds the incremental state and the local memory at `…/memory` (the `chunks` Lance dataset).

use crate::chunk;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{Context, Result};
use arrow_array::types::Float32Type;
use arrow_array::{new_null_array, ArrayRef, BooleanArray, FixedSizeListArray, Int64Array, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema};
use futures::TryStreamExt;
use lance::dataset::builder::DatasetBuilder;
use lance::dataset::{BatchUDF, Dataset, NewColumnTransform};
use lance::index::vector::VectorIndexParams;
use lance::index::DatasetIndexExt;
use lance_index::scalar::InvertedIndexParams;
use lance_index::vector::ivf::IvfBuildParams;
use lance_index::vector::pq::PQBuildParams;
use lance_index::IndexType;
use lance_io::object_store::{ObjectStoreParams, WrappingObjectStore};
use lance_linalg::distance::MetricType;

/// The table (Lance dataset) name within a memory.
pub const TABLE: &str = "chunks";

/// The embedding model a memory's vectors are built with, and their width. Pinned in the schema
/// metadata and enforced on open ([`super::Memory::open`]): a memory built with another model
/// can't be queried with funes's embeddings.
pub const MODEL: &str = "BAAI/bge-small-en-v1.5";
pub const DIM: i32 = 384;

/// Columns introduced additively after the original transcript schema. The order matches
/// [`schema`], because `add_columns` appends fields in exactly the order supplied here.
const EVOLVABLE_COLUMNS: &[&str] = &[
    "harness",
    "repo",
    "source_identity",
    "source_version",
    "content_hash",
    "updated_at",
    "source_agent",
    "source_type",
    "project",
    "device_id",
    "content_type",
    "source_missing",
    "metadata_json",
];

/// funes's home directory: `$FUNES_HOME`, else `~/.funes`. Holds the incremental state and the
/// local memory.
pub fn funes_dir() -> PathBuf {
    if let Ok(d) = std::env::var("FUNES_HOME") {
        return PathBuf::from(d);
    }
    let home = std::env::var("HOME").unwrap_or_default();
    PathBuf::from(home).join(".funes")
}

/// Directory holding the local memory (the `chunks` dataset is at `<dir>/chunks.lance`).
pub fn local_memory_dir() -> String {
    let dir = funes_dir().join("memory");
    // Migrate a pre-rename layout in place: an earlier funes kept the local memory at `<home>/store`.
    // Rename it once, when the current path doesn't exist yet. Best-effort — a failed rename just
    // leaves the old memory unfound, which reads as "no index yet".
    let legacy = funes_dir().join("store");
    if legacy.is_dir() && !dir.exists() {
        let _ = std::fs::rename(&legacy, &dir);
    }
    dir.to_string_lossy().into_owned()
}

/// The `chunks` dataset URI under a memory base (a local directory or a remote URI prefix).
pub fn table_uri(base: &str) -> String {
    format!("{base}/{TABLE}.lance")
}

/// Open the `chunks` dataset at `uri`; `storage_options` carries the backend credentials/revision a
/// remote needs (empty for a local memory).
pub async fn open(uri: &str, storage_options: HashMap<String, String>) -> Result<Dataset> {
    DatasetBuilder::from_uri(uri)
        .with_storage_options(storage_options)
        .load()
        .await
        .context("opening the dataset")
}

/// Open the `chunks` dataset at `uri` with `wrapper` decorating its object store. It is installed
/// before load, so it sees every read Lance issues, including those during load. `storage_options`
/// carries the backend credentials/revision a remote needs; the caller supplies the wrapper.
pub async fn open_wrapped(
    uri: &str,
    storage_options: HashMap<String, String>,
    wrapper: Arc<dyn WrappingObjectStore>,
) -> Result<Dataset> {
    // Order matters: `with_store_params` replaces the params wholesale, so install the wrapper
    // first, then layer the storage options on top (`with_storage_options` merges into them).
    DatasetBuilder::from_uri(uri)
        .with_store_params(ObjectStoreParams {
            object_store_wrapper: Some(wrapper),
            ..Default::default()
        })
        .with_storage_options(storage_options)
        .load()
        .await
        .context("opening the wrapped dataset")
}

/// Project `columns` (empty = all columns; optionally filtered by a SQL predicate, optionally
/// limited) and collect the matching rows. Plain scans aren't limit-capped, so callers pass `None`
/// to read everything.
pub async fn scan_rows(
    ds: &Dataset,
    columns: &[&str],
    filter: Option<&str>,
    limit: Option<i64>,
) -> Result<Vec<RecordBatch>> {
    let mut scan = ds.scan();
    if !columns.is_empty() {
        scan.project(columns)?;
    }
    if let Some(f) = filter {
        scan.filter(f)?;
    }
    scan.limit(limit, None)?;
    let mut stream = scan.try_into_stream().await?;
    let mut batches = Vec::new();
    while let Some(batch) = stream.try_next().await? {
        batches.push(batch);
    }
    Ok(batches)
}

/// Best-effort: build the FTS index on `text` and the IVF_PQ index on `vector`. A small corpus
/// can't train IVF (lance needs ~256 rows) — that's fine, recall falls back to brute force.
///
/// `on_phase` is called with a human label before each index is built, so a caller can report
/// progress around these opaque (no incremental hook), potentially slow Lance calls. Pass `|_| {}`
/// to stay silent.
pub async fn build_indexes(ds: &mut Dataset, on_phase: impl Fn(&str)) {
    on_phase("text search index");
    let _ = ds
        .create_index(
            &["text"],
            IndexType::Inverted,
            None,
            &InvertedIndexParams::default(),
            true,
        )
        .await;
    if let Some(params) = ivf_pq_params(ds) {
        on_phase("vector index");
        let _ = ds
            .create_index(&["vector"], IndexType::Vector, None, &params, true)
            .await;
    }
}

/// IVF_PQ parameters sized from the `vector` column's dimension (matching lancedb's defaults).
/// `None` if there is no fixed-size `vector` column.
fn ivf_pq_params(ds: &Dataset) -> Option<VectorIndexParams> {
    let arrow = arrow_schema::Schema::from(ds.schema());
    let arrow_schema::DataType::FixedSizeList(_, dim) = arrow.field_with_name("vector").ok()?.data_type() else {
        return None;
    };
    let dim = *dim as usize;
    let num_sub_vectors = if dim.is_multiple_of(16) {
        dim / 16
    } else if dim.is_multiple_of(8) {
        dim / 8
    } else {
        1
    };
    let mut pq = PQBuildParams::new(num_sub_vectors, 8);
    pq.max_iters = 50;
    Some(VectorIndexParams::with_ivf_pq_params(
        MetricType::L2,
        IvfBuildParams::default(),
        pq,
    ))
}

/// The table schema (column order is load-bearing for Lance).
pub(crate) fn schema() -> Arc<Schema> {
    let utf8 = |name: &str| Field::new(name, DataType::Utf8, true);
    let i64f = |name: &str| Field::new(name, DataType::Int64, true);
    Arc::new(Schema::new_with_metadata(
        vec![
            utf8("id"),
            utf8("text"),
            utf8("session_id"),
            utf8("workdir"),
            utf8("turn_uuid"),
            utf8("parent_uuid"),
            i64f("seq"),
            utf8("ts"),
            utf8("role"),
            utf8("block_type"),
            utf8("tool_name"),
            utf8("source_path"),
            i64f("block_idx"),
            i64f("split_idx"),
            Field::new(
                "vector",
                DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, true)), DIM),
                true,
            ),
            // After `vector`: `add_columns` appends a migrated column at the end, so a
            // freshly-built memory must match that order (the tripwire test pins it). `harness`
            // came first, then `repo` — each appended in turn.
            utf8("harness"),
            utf8("repo"),
            utf8("source_identity"),
            utf8("source_version"),
            utf8("content_hash"),
            utf8("updated_at"),
            utf8("source_agent"),
            utf8("source_type"),
            utf8("project"),
            utf8("device_id"),
            utf8("content_type"),
            Field::new("source_missing", DataType::Boolean, true),
            utf8("metadata_json"),
        ],
        HashMap::from([("embedding_model".to_string(), MODEL.to_string())]),
    ))
}

/// Build the null-valued additive migration needed before canonical rows can be merged into `ds`.
/// Existing transcript rows intentionally receive null provenance rather than invented values.
pub(crate) fn canonical_column_migration(ds: &Dataset) -> Option<(NewColumnTransform, Vec<String>)> {
    let current = Schema::from(ds.schema());
    let desired = schema();
    let fields: Vec<Field> = EVOLVABLE_COLUMNS
        .iter()
        .filter(|name| current.column_with_name(name).is_none())
        .map(|name| {
            desired
                .field_with_name(name)
                .expect("evolvable field is in schema")
                .clone()
        })
        .collect();
    if fields.is_empty() {
        return None;
    }
    let output_schema = Arc::new(Schema::new(fields));
    let mapper_schema = output_schema.clone();
    let transform = NewColumnTransform::BatchUDF(BatchUDF {
        mapper: Box::new(move |batch: &RecordBatch| {
            let columns = mapper_schema
                .fields()
                .iter()
                .map(|field| new_null_array(field.data_type(), batch.num_rows()))
                .collect();
            RecordBatch::try_new(mapper_schema.clone(), columns).map_err(lance::Error::from)
        }),
        output_schema,
        result_checkpoint: None,
    });
    Some((transform, vec!["id".to_string()]))
}

/// Add every missing canonical facet to a local or wrapped dataset in one schema-evolution commit.
pub(crate) async fn ensure_canonical_columns(ds: &mut Dataset) -> Result<bool> {
    let Some((transform, read_columns)) = canonical_column_migration(ds) else {
        return Ok(false);
    };
    ds.add_columns(transform, Some(read_columns), None)
        .await
        .context("adding canonical document columns")?;
    Ok(true)
}

pub(crate) fn build_batch(chunks: &[chunk::Chunk], vectors: &[Vec<f32>]) -> Result<RecordBatch> {
    build_batch_for_schema(chunks, vectors, schema())
}

/// Build rows against `target`, preserving its column order. This keeps ordinary transcript
/// appends compatible with older memories while new memories use the extended canonical schema.
pub(crate) fn build_batch_for_schema(
    chunks: &[chunk::Chunk],
    vectors: &[Vec<f32>],
    target: Arc<Schema>,
) -> Result<RecordBatch> {
    let s = |f: &dyn Fn(&chunk::Chunk) -> Option<String>| -> ArrayRef {
        Arc::new(chunks.iter().map(f).collect::<StringArray>())
    };
    let i = |f: &dyn Fn(&chunk::Chunk) -> i64| -> ArrayRef {
        Arc::new(chunks.iter().map(|c| Some(f(c))).collect::<Int64Array>())
    };
    let vector = FixedSizeListArray::from_iter_primitive::<Float32Type, _, _>(
        vectors
            .iter()
            .map(|v| Some(v.iter().map(|&x| Some(x)).collect::<Vec<_>>())),
        DIM,
    );
    let mut columns: Vec<ArrayRef> = Vec::with_capacity(target.fields().len());
    for field in target.fields() {
        columns.push(match field.name().as_str() {
            "id" => s(&|c| Some(c.id.clone())),
            "text" => s(&|c| Some(c.text.clone())),
            "session_id" => s(&|c| Some(c.session_id.clone())),
            "workdir" => s(&|c| Some(c.workdir.clone())),
            "turn_uuid" => s(&|c| Some(c.turn_uuid.clone())),
            "parent_uuid" => s(&|c| c.parent_uuid.clone()),
            "seq" => i(&|c| c.seq),
            "ts" => s(&|c| Some(c.ts.clone())),
            "role" => s(&|c| Some(c.role.clone())),
            "block_type" => s(&|c| Some(c.block_type.clone())),
            "tool_name" => s(&|c| c.tool_name.clone()),
            "source_path" => s(&|c| Some(c.source_path.clone())),
            "block_idx" => i(&|c| c.block_idx),
            "split_idx" => i(&|c| c.split_idx),
            "vector" => Arc::new(vector.clone()),
            "harness" => s(&|c| Some(c.harness.clone())),
            "repo" => s(&|c| c.repo.clone()),
            "source_identity" => s(&|c| c.source_identity.clone()),
            "source_version" => s(&|c| c.source_version.clone()),
            "content_hash" => s(&|c| c.content_hash.clone()),
            "updated_at" => s(&|c| c.updated_at.clone()),
            "source_agent" => s(&|c| c.source_agent.clone()),
            "source_type" => s(&|c| c.source_type.clone()),
            "project" => s(&|c| c.project.clone()),
            "device_id" => s(&|c| c.device_id.clone()),
            "content_type" => s(&|c| c.content_type.clone()),
            "source_missing" => Arc::new(chunks.iter().map(|c| c.source_missing).collect::<BooleanArray>()),
            "metadata_json" => s(&|c| c.metadata_json.clone()),
            other => anyhow::bail!("unsupported memory column {other:?}"),
        });
    }
    Ok(RecordBatch::try_new(target, columns)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn schema_column_order_is_load_bearing() {
        // Column order must match build_batch's array order exactly, or Lance writes the
        // wrong column. Pin it so a reorder can't slip through.
        let s = schema();
        let names: Vec<&str> = s.fields().iter().map(|f| f.name().as_str()).collect();
        assert_eq!(
            names,
            vec![
                "id",
                "text",
                "session_id",
                "workdir",
                "turn_uuid",
                "parent_uuid",
                "seq",
                "ts",
                "role",
                "block_type",
                "tool_name",
                "source_path",
                "block_idx",
                "split_idx",
                "vector",
                "harness",
                "repo",
                "source_identity",
                "source_version",
                "content_hash",
                "updated_at",
                "source_agent",
                "source_type",
                "project",
                "device_id",
                "content_type",
                "source_missing",
                "metadata_json",
            ]
        );
    }
}
