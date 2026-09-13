//! Shared memory helpers: the local memory location, the `chunks` table's schema and rows, opening a
//! dataset, plain scans, and building the FTS/IVF indexes. funes's home is `$FUNES_HOME`/`~/.funes` —
//! it holds the incremental state and the local memory at `…/memory` (the `chunks` Lance dataset).

use crate::chunk;
use crate::inference::EmbeddingProfile;
use std::collections::{HashMap, HashSet};
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
use lance_index::scalar::{InvertedIndexParams, ScalarIndexParams};
use lance_index::vector::ivf::IvfBuildParams;
use lance_index::vector::pq::PQBuildParams;
use lance_index::IndexType;
use lance_io::object_store::{ObjectStoreParams, WrappingObjectStore};
use lance_linalg::distance::MetricType;

/// The table (Lance dataset) name within a memory.
pub const TABLE: &str = "chunks";

/// Legacy local-BGE profile constants. New memories pin the complete runtime
/// [`EmbeddingProfile`] in their schema metadata; these remain public for old callers and tests.
pub const MODEL: &str = "BAAI/bge-small-en-v1.5";
pub const DIM: i32 = 384;
pub const EMBEDDING_PROVIDER_KEY: &str = "embedding_provider";
pub const EMBEDDING_MODEL_KEY: &str = "embedding_model";
pub const EMBEDDING_DIMENSIONS_KEY: &str = "embedding_dimensions";
pub const EMBEDDING_SCHEMA_VERSION_KEY: &str = "embedding_schema_version";
pub const EMBEDDING_FINGERPRINT_KEY: &str = "embedding_fingerprint";

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

const VECTOR_INDEX_MIN_ROWS: usize = 256;
const TEXT_INDEX_NAME: &str = "text_idx";
const VECTOR_INDEX_NAME: &str = "vector_idx";
const SOURCE_IDENTITY_INDEX_NAME: &str = "source_identity_idx";

/// Build the FTS index on `text` and, once it has enough rows to train, the IVF_PQ index on
/// `vector`. A small corpus falls back to brute-force vector recall until it reaches Lance's
/// training minimum.
///
/// `on_phase` is called with a human label before each index is built, so a caller can report
/// progress around these opaque (no incremental hook), potentially slow Lance calls. Pass `|_| {}`
/// to stay silent.
pub async fn build_indexes(
    ds: &mut Dataset,
    on_phase: impl Fn(&str),
) -> Result<()> {
    maintain_required_indexes(ds, on_phase, true).await?;
    Ok(())
}

/// Create every index required by the dataset's current schema and row count, leaving existing
/// bases intact. This is also the remote reindex entry point: a memory first published below the
/// IVF training floor gains its vector index once later appends make it large enough.
pub(crate) async fn ensure_required_indexes(
    ds: &mut Dataset,
    on_phase: impl Fn(&str),
) -> Result<bool> {
    maintain_required_indexes(ds, on_phase, false).await
}

async fn maintain_required_indexes(
    ds: &mut Dataset,
    on_phase: impl Fn(&str),
    replace_existing: bool,
) -> Result<bool> {
    let mut existing = ds
        .load_indices()
        .await
        .context("listing indexes before index maintenance")?
        .iter()
        .map(|index| index.name.clone())
        .collect::<HashSet<_>>();
    let mut created = false;

    if replace_existing || !existing.contains(TEXT_INDEX_NAME) {
        on_phase("text search index");
        ds.create_index(
            &["text"],
            IndexType::Inverted,
            Some(TEXT_INDEX_NAME.to_string()),
            &InvertedIndexParams::default(),
            replace_existing,
        )
        .await
        .context("creating text search index")?;
        existing.insert(TEXT_INDEX_NAME.to_string());
        created = true;
    }
    if vector_index_required(ds).await? {
        if replace_existing || !existing.contains(VECTOR_INDEX_NAME) {
            let params = ivf_pq_params(ds).expect("required vector index has a vector column");
            on_phase("vector index");
            ds.create_index(
                &["vector"],
                IndexType::Vector,
                Some(VECTOR_INDEX_NAME.to_string()),
                &params,
                replace_existing,
            )
            .await
            .context("creating vector index")?;
            existing.insert(VECTOR_INDEX_NAME.to_string());
            created = true;
        }
    }
    if Schema::from(ds.schema()).column_with_name("source_identity").is_some()
        && (replace_existing || !existing.contains(SOURCE_IDENTITY_INDEX_NAME))
    {
        on_phase("source identity index");
        ds.create_index(
            &["source_identity"],
            IndexType::BTree,
            Some(SOURCE_IDENTITY_INDEX_NAME.to_string()),
            &ScalarIndexParams::default(),
            replace_existing,
        )
        .await
        .context("creating source identity index")?;
        created = true;
    }
    Ok(created)
}

/// Whether a no-new-chunk indexing pass must rebuild its indexes. This closes the crash window
/// after an append: the source state may already say every chunk is stored, while the interrupted
/// finalization left an absent FTS/IVF index or a delta with unindexed rows behind.
pub(crate) async fn indexes_need_rebuild(ds: &Dataset) -> Result<bool> {
    let rows = ds.count_rows(None).await.context("counting rows for index health")?;
    if rows == 0 {
        return Ok(false);
    }
    let indexes = ds.load_indices().await.context("listing indexes for index health")?;
    let mut required = vec![TEXT_INDEX_NAME];
    if vector_index_required(ds).await? {
        required.push(VECTOR_INDEX_NAME);
    }
    if Schema::from(ds.schema()).column_with_name("source_identity").is_some() {
        required.push(SOURCE_IDENTITY_INDEX_NAME);
    }
    for name in required {
        if !indexes.iter().any(|index| index.name == name) {
            return Ok(true);
        }
        let statistics = ds
            .index_statistics(name)
            .await
            .with_context(|| format!("reading index health for {name}"))?;
        let unindexed = serde_json::from_str::<serde_json::Value>(&statistics)
            .with_context(|| format!("parsing index health for {name}"))?
            .get("num_unindexed_rows")
            .and_then(serde_json::Value::as_u64)
            .context("index health omitted num_unindexed_rows")?;
        if unindexed > 0 {
            return Ok(true);
        }
    }
    Ok(false)
}

async fn vector_index_required(ds: &Dataset) -> Result<bool> {
    let rows = ds.count_rows(None).await.context("counting rows for vector index")?;
    Ok(ivf_pq_params(ds).is_some() && rows >= VECTOR_INDEX_MIN_ROWS)
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
pub(crate) fn schema_for(profile: &EmbeddingProfile) -> Arc<Schema> {
    let utf8 = |name: &str| Field::new(name, DataType::Utf8, true);
    let i64f = |name: &str| Field::new(name, DataType::Int64, true);
    let dimensions = i32::try_from(profile.dimensions).expect("embedding dimension fits in i32");
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
                DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, true)), dimensions),
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
        HashMap::from([
            (EMBEDDING_PROVIDER_KEY.to_string(), profile.provider.clone()),
            (EMBEDDING_MODEL_KEY.to_string(), profile.model.clone()),
            (
                EMBEDDING_DIMENSIONS_KEY.to_string(),
                profile.dimensions.to_string(),
            ),
            (
                EMBEDDING_SCHEMA_VERSION_KEY.to_string(),
                profile.schema_version.clone(),
            ),
            (
                EMBEDDING_FINGERPRINT_KEY.to_string(),
                profile.fingerprint.clone(),
            ),
        ]),
    ))
}

/// Legacy local schema used by fixtures and migrations whose vector contract is already known.
pub(crate) fn schema() -> Arc<Schema> {
    schema_for(&EmbeddingProfile::local())
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

#[cfg(test)]
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
    if chunks.len() != vectors.len() {
        anyhow::bail!(
            "embedding provider returned {} vectors for {} chunks",
            vectors.len(),
            chunks.len()
        );
    }
    let dimension = match target.field_with_name("vector")?.data_type() {
        DataType::FixedSizeList(_, dimension) => *dimension,
        _ => anyhow::bail!("memory `vector` column is not a fixed-size list"),
    };
    for (index, vector) in vectors.iter().enumerate() {
        if vector.len() != dimension as usize {
            anyhow::bail!(
                "embedding vector {index} has dimension {}, expected {dimension}",
                vector.len()
            );
        }
        if vector.iter().any(|value| !value.is_finite()) {
            anyhow::bail!("embedding vector {index} contains a non-finite value");
        }
    }
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
        dimension,
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
    use arrow_array::RecordBatchIterator;
    use lance::dataset::WriteParams;
    use lance_index::optimize::OptimizeOptions;

    fn text_schema() -> Arc<Schema> {
        Arc::new(Schema::new(vec![Field::new("text", DataType::Utf8, false)]))
    }

    fn text_batch(texts: &[&str]) -> RecordBatch {
        RecordBatch::try_new(
            text_schema(),
            vec![Arc::new(StringArray::from(texts.to_vec()))],
        )
        .unwrap()
    }

    fn indexable_schema() -> Arc<Schema> {
        Arc::new(Schema::new(vec![
            Field::new("text", DataType::Utf8, false),
            Field::new(
                "vector",
                DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, true)), 16),
                false,
            ),
            Field::new("source_identity", DataType::Utf8, true),
        ]))
    }

    fn indexable_batch(start: usize, rows: usize) -> RecordBatch {
        let schema = indexable_schema();
        let texts = (start..start + rows)
            .map(|row| format!("canonical document {row}"))
            .collect::<Vec<_>>();
        let identities = (start..start + rows)
            .map(|row| Some(format!("source-{row}")))
            .collect::<Vec<_>>();
        let vectors = FixedSizeListArray::from_iter_primitive::<Float32Type, _, _>(
            (start..start + rows).map(|row| {
                Some(
                    (0..16)
                        .map(|column| Some(((row * 17 + column * 31) % 997) as f32 / 997.0))
                        .collect::<Vec<_>>(),
                )
            }),
            16,
        );
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(StringArray::from(texts)),
                Arc::new(vectors),
                Arc::new(StringArray::from(identities)),
            ],
        )
        .unwrap()
    }

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

    #[test]
    fn voyage_schema_pins_the_complete_embedding_profile() {
        let profile = EmbeddingProfile::voyage("voyage-4-lite", 1024).unwrap();
        let schema = schema_for(&profile);
        let DataType::FixedSizeList(_, dimension) = schema.field_with_name("vector").unwrap().data_type() else {
            panic!("vector must be fixed-size")
        };
        assert_eq!(*dimension, 1024);
        assert_eq!(schema.metadata().get(EMBEDDING_PROVIDER_KEY), Some(&"voyage".to_string()));
        assert_eq!(schema.metadata().get(EMBEDDING_MODEL_KEY), Some(&"voyage-4-lite".to_string()));
        assert_eq!(schema.metadata().get(EMBEDDING_DIMENSIONS_KEY), Some(&"1024".to_string()));
        assert_eq!(schema.metadata().get(EMBEDDING_SCHEMA_VERSION_KEY), Some(&"2".to_string()));
        assert_eq!(
            schema.metadata().get(EMBEDDING_FINGERPRINT_KEY),
            Some(&profile.fingerprint)
        );
    }

    #[tokio::test]
    async fn interrupted_index_finalization_is_repaired_without_new_source_rows() {
        let dir = tempfile::tempdir().unwrap();
        let uri = dir.path().join("chunks.lance");
        let first = text_batch(&["already stored source row"]);
        let mut ds = Dataset::write(
            RecordBatchIterator::new(vec![Ok(first)].into_iter(), text_schema()),
            uri.to_str().unwrap(),
            Some(WriteParams::default()),
        )
        .await
        .unwrap();

        // This is the state after append + persisted source state, then a crash before finalize:
        // the retry will find no new source chunks, but FTS is still absent.
        assert!(indexes_need_rebuild(&ds).await.unwrap());
        build_indexes(&mut ds, |_| {}).await.unwrap();
        assert!(!indexes_need_rebuild(&ds).await.unwrap());

        // An append after an existing FTS leaves an index delta. A no-new-source retry must also
        // detect that debt and fold the stored row into a rebuilt index.
        let appended = text_batch(&["stored while finalization was interrupted"]);
        ds.append(
            RecordBatchIterator::new(vec![Ok(appended)].into_iter(), text_schema()),
            None,
        )
        .await
        .unwrap();
        assert!(indexes_need_rebuild(&ds).await.unwrap());
        build_indexes(&mut ds, |_| {}).await.unwrap();
        assert!(!indexes_need_rebuild(&ds).await.unwrap());
    }

    #[tokio::test]
    async fn index_maintenance_adds_vector_and_source_identity_indexes_after_growth() {
        let dir = tempfile::tempdir().unwrap();
        let uri = dir.path().join("chunks.lance");
        let first = indexable_batch(0, VECTOR_INDEX_MIN_ROWS - 1);
        let mut ds = Dataset::write(
            RecordBatchIterator::new(vec![Ok(first)].into_iter(), indexable_schema()),
            uri.to_str().unwrap(),
            Some(WriteParams::default()),
        )
        .await
        .unwrap();

        build_indexes(&mut ds, |_| {}).await.unwrap();
        let initial = ds
            .load_indices()
            .await
            .unwrap()
            .iter()
            .map(|index| index.name.clone())
            .collect::<HashSet<_>>();
        assert!(initial.contains(TEXT_INDEX_NAME));
        assert!(initial.contains(SOURCE_IDENTITY_INDEX_NAME));
        assert!(!initial.contains(VECTOR_INDEX_NAME));

        let appended = indexable_batch(VECTOR_INDEX_MIN_ROWS - 1, 77);
        ds.append(
            RecordBatchIterator::new(vec![Ok(appended)].into_iter(), indexable_schema()),
            None,
        )
        .await
        .unwrap();
        assert!(indexes_need_rebuild(&ds).await.unwrap());
        assert!(ensure_required_indexes(&mut ds, |_| {}).await.unwrap());
        ds.optimize_indices(&OptimizeOptions::append()).await.unwrap();

        let maintained = ds
            .load_indices()
            .await
            .unwrap()
            .iter()
            .map(|index| index.name.clone())
            .collect::<HashSet<_>>();
        assert!(maintained.contains(VECTOR_INDEX_NAME));
        assert!(!indexes_need_rebuild(&ds).await.unwrap());
    }

    #[tokio::test]
    async fn build_indexes_returns_text_index_errors() {
        let dir = tempfile::tempdir().unwrap();
        let uri = dir.path().join("chunks.lance");
        let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Utf8, false)]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(StringArray::from(vec!["row"]))],
        )
        .unwrap();
        let mut ds = Dataset::write(
            RecordBatchIterator::new(vec![Ok(batch)].into_iter(), schema),
            uri.to_str().unwrap(),
            Some(WriteParams::default()),
        )
        .await
        .unwrap();

        let error = build_indexes(&mut ds, |_| {}).await.unwrap_err();
        assert!(
            error.to_string().contains("creating text search index"),
            "missing text column should propagate the FTS creation error: {error:#}"
        );
    }
}
