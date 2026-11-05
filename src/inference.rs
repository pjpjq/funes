//! The inference backend behind funes' two model operations — embedding and reranking. The rest of
//! funes talks to these traits via the [`embedder`]/[`reranker`] factories, never a concrete ML
//! stack, so an alternative backend slots in behind the same interface. The backend is chosen at
//! build time in one place — the `Default*` aliases below: default build → BLAS (a from-scratch
//! forward on Accelerate/faer); `--no-default-features --features onnx` → fastembed/ort. On macOS
//! the default build also carries `metal`, the same encoders on the GPU, which [`embedder`] and
//! [`reranker`] hand their bulk calls to (see [`metal::Hybrid`] and [`metal::HybridReranker`]).

#[cfg(feature = "blas")]
pub mod blas;
#[cfg(all(feature = "metal", target_os = "macos"))]
pub mod metal;

pub mod voyage;

use self::voyage::{VoyageEmbedder, VoyageReranker};
use anyhow::{anyhow, bail, Context, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::env;
use std::path::{Path, PathBuf};

// The single backend-selection point. One of these alias pairs is compiled; the factories below
// box whichever it names. When both backends are compiled in (the backend benchmark builds that
// way to use ONNX as its reference), BLAS is the one funes runs.
#[cfg(all(feature = "onnx", not(feature = "blas")))]
use self::{OnnxEmbedder as DefaultEmbedder, OnnxReranker as DefaultReranker};
#[cfg(feature = "blas")]
use blas::{BlasEmbedder as DefaultEmbedder, GteReranker as DefaultReranker};
#[cfg(not(any(feature = "blas", feature = "onnx")))]
compile_error!("funes needs an inference backend: feature `blas` (default) or `onnx`");

/// How many texts one `embed` call takes.
const EMBED_BATCH: usize = 256;

/// The most tokens an encoder reads: a longer text, or query and passage pair, is cut to it. The
/// BERT models' learned positions stop there.
const MAX_LEN: usize = 512;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum EmbeddingModel {
    BgeSmallEn,
    MultilingualE5Small,
}

impl EmbeddingModel {
    pub const ALL: [EmbeddingModel; 2] = [EmbeddingModel::BgeSmallEn, EmbeddingModel::MultilingualE5Small];

    pub const fn id(self) -> &'static str {
        match self {
            EmbeddingModel::BgeSmallEn => "BAAI/bge-small-en-v1.5",
            EmbeddingModel::MultilingualE5Small => "intfloat/multilingual-e5-small",
        }
    }

    pub const fn from_id(id: &str) -> Option<EmbeddingModel> {
        let mut i = 0;
        while i < EmbeddingModel::ALL.len() {
            if same_bytes(EmbeddingModel::ALL[i].id().as_bytes(), id.as_bytes()) {
                return Some(EmbeddingModel::ALL[i]);
            }
            i += 1;
        }
        None
    }

    /// e5 was trained with these query and passage prefixes, and its model card asks for them.
    fn prefixes(self) -> (&'static str, &'static str) {
        match self {
            EmbeddingModel::BgeSmallEn => ("", ""),
            EmbeddingModel::MultilingualE5Small => ("query: ", "passage: "),
        }
    }
}

const fn same_bytes(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut i = 0;
    while i < a.len() {
        if a[i] != b[i] {
            return false;
        }
        i += 1;
    }
    true
}

const LOCAL_EMBEDDING_MODEL: &str = "BAAI/bge-small-en-v1.5";
const LOCAL_EMBEDDING_DIMENSIONS: usize = 384;
const LOCAL_EMBEDDING_SCHEMA_VERSION: &str = "1";
const VOYAGE_EMBEDDING_MODEL: &str = "voyage-4-lite";
const VOYAGE_EMBEDDING_DIMENSIONS: usize = 1024;
const VOYAGE_EMBEDDING_SCHEMA_VERSION: &str = "2";

pub(super) fn voyage_dimensions_supported(dimensions: usize) -> bool {
    matches!(dimensions, 256 | 512 | 1024 | 2048)
}

/// The exact embedding-space contract persisted beside every memory.
///
/// A different fingerprint means vectors must not be mixed, even when their dimensions happen
/// to match. The normalization and metric are fixed contract inputs included in the fingerprint.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct EmbeddingProfile {
    pub provider: String,
    pub model: String,
    pub dimensions: usize,
    pub schema_version: String,
    pub fingerprint: String,
}

impl EmbeddingProfile {
    fn new(provider: &str, model: &str, dimensions: usize, schema_version: &str) -> Self {
        let mut profile = Self {
            provider: provider.to_string(),
            model: model.to_string(),
            dimensions,
            schema_version: schema_version.to_string(),
            fingerprint: String::new(),
        };
        profile.fingerprint = profile_fingerprint(&profile);
        profile
    }

    /// The fixed local BGE embedding space used by existing memories.
    pub fn local() -> Self {
        Self::new(
            "local",
            LOCAL_EMBEDDING_MODEL,
            LOCAL_EMBEDDING_DIMENSIONS,
            LOCAL_EMBEDDING_SCHEMA_VERSION,
        )
    }

    /// Local backends keep their model-specific query and passage prefixes.
    pub fn local_model(model: EmbeddingModel) -> Self {
        Self::new(
            "local",
            model.id(),
            LOCAL_EMBEDDING_DIMENSIONS,
            LOCAL_EMBEDDING_SCHEMA_VERSION,
        )
    }

    pub fn validate(&self) -> Result<()> {
        if self.fingerprint != profile_fingerprint(self) {
            bail!("embedding profile fingerprint does not match its fields")
        }
        match self.provider.as_str() {
            "local"
                if EmbeddingModel::from_id(&self.model).is_some()
                    && self.dimensions == LOCAL_EMBEDDING_DIMENSIONS
                    && self.schema_version == LOCAL_EMBEDDING_SCHEMA_VERSION =>
            {
                Ok(())
            }
            "voyage"
                if !self.model.trim().is_empty()
                    && voyage_dimensions_supported(self.dimensions)
                    && self.schema_version == VOYAGE_EMBEDDING_SCHEMA_VERSION =>
            {
                Ok(())
            }
            _ => bail!("unsupported embedding profile"),
        }
    }

    /// A Voyage raw-text embedding space using the current document/query contract.
    pub fn voyage(model: impl Into<String>, dimensions: usize) -> Result<Self> {
        let model = model.into();
        if model.trim().is_empty() {
            bail!("Voyage embedding model must not be empty")
        }
        if !voyage_dimensions_supported(dimensions) {
            bail!("Voyage embedding dimensions must be one of 256, 512, 1024, or 2048")
        }
        Ok(Self::new(
            "voyage",
            model.trim(),
            dimensions,
            VOYAGE_EMBEDDING_SCHEMA_VERSION,
        ))
    }
}

fn profile_fingerprint(profile: &EmbeddingProfile) -> String {
    let contract = format!(
        "provider={}\nmodel={}\ndimensions={}\nschema_version={}\ndocument_input=document\nquery_input=query\nnormalization=l2\nmetric=l2",
        profile.provider, profile.model, profile.dimensions, profile.schema_version
    );
    hex::encode(Sha256::digest(contract.as_bytes()))
}

fn env_value(name: &str) -> Result<Option<String>> {
    match env::var(name) {
        Ok(value) => Ok(Some(value)),
        Err(env::VarError::NotPresent) => Ok(None),
        Err(env::VarError::NotUnicode(_)) => bail!("{name} is not valid UTF-8"),
    }
}

fn profile_from_values(
    provider: Option<String>,
    voyage_model: Option<String>,
    voyage_dimensions: Option<String>,
    voyage_schema_version: Option<String>,
) -> Result<EmbeddingProfile> {
    let provider = provider.unwrap_or_else(|| "local".to_string());
    match provider.as_str() {
        "local" => Ok(EmbeddingProfile::local()),
        "voyage" => {
            let model = voyage_model.unwrap_or_else(|| VOYAGE_EMBEDDING_MODEL.to_string());
            if model.trim().is_empty() {
                bail!("FUNES_EMBEDDING_MODEL must not be empty")
            }
            let dimensions = voyage_dimensions
                .map(|value| {
                    value
                        .parse::<usize>()
                        .context("FUNES_EMBEDDING_DIMENSIONS must be one of 256, 512, 1024, or 2048")
                })
                .transpose()?
                .unwrap_or(VOYAGE_EMBEDDING_DIMENSIONS);
            let schema_version = voyage_schema_version
                .as_deref()
                .unwrap_or(VOYAGE_EMBEDDING_SCHEMA_VERSION);
            if schema_version != VOYAGE_EMBEDDING_SCHEMA_VERSION {
                bail!("unsupported FUNES_EMBEDDING_SCHEMA_VERSION `{schema_version}`; expected `2`")
            }
            EmbeddingProfile::voyage(model, dimensions)
        }
        other => Err(anyhow!(
            "unsupported FUNES_EMBEDDING_PROVIDER `{other}`; expected `local` or `voyage`"
        )),
    }
}

/// Resolve the runtime embedding-space contract without loading a model or reading an API key.
pub fn embedding_profile() -> Result<EmbeddingProfile> {
    profile_from_values(
        env_value("FUNES_EMBEDDING_PROVIDER")?,
        env_value("FUNES_EMBEDDING_MODEL")?,
        env_value("FUNES_EMBEDDING_DIMENSIONS")?,
        env_value("FUNES_EMBEDDING_SCHEMA_VERSION")?,
    )
}

/// Embed each text into a dense vector, in input order.
pub trait Embedder: Send {
    fn model(&self) -> Option<EmbeddingModel> {
        None
    }

    /// Encode each text as given, in input order.
    fn encode(&mut self, _texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        bail!("this provider has no local raw encoder")
    }

    /// How many texts a call should carry. A caller reports progress and checks its budget between
    /// calls, so this is about a few seconds' work; a GPU wants many texts per call to stay busy.
    fn batch_size(&self) -> usize {
        EMBED_BATCH
    }

    fn embed(&mut self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        let prefix = self.model().map_or("", |model| model.prefixes().1);
        let prefixed: Vec<String> = texts.iter().map(|t| format!("{prefix}{t}")).collect();
        self.encode(&prefixed.iter().map(String::as_str).collect::<Vec<_>>())
    }

    fn embed_query(&mut self, query: &str) -> Result<Vec<f32>> {
        let prefixed = format!("{}{query}", self.model().map_or("", |model| model.prefixes().0));
        self.encode(&[prefixed.as_str()])?.pop().context("empty embedding")
    }

    /// Embed persisted material in the provider's document input space.
    fn embed_documents(&mut self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        self.embed(texts)
    }
}

/// The numeric contract returned by a [`Reranker`]. Local cross-encoders return classifier logits;
/// hosted rerank APIs may already return a normalized relevance score.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RerankScoreKind {
    Logit,
    Relevance,
}

/// Score each doc against the query; one score per doc, in input order (higher = more relevant).
pub trait Reranker: Send {
    fn rerank(&mut self, query: &str, docs: &[&str]) -> Result<Vec<f32>>;

    fn score_kind(&self) -> RerankScoreKind {
        RerankScoreKind::Logit
    }
}

/// Build the embedder for `model` on the compiled-in backend. Call sites use this instead of
/// naming a concrete type, so the backend is decided only by the `Default*` alias above.
pub fn embedder(model: EmbeddingModel) -> Result<Box<dyn Embedder>> {
    #[cfg(all(feature = "metal", target_os = "macos"))]
    return Ok(Box::new(metal::Hybrid::new(model)?));
    #[allow(unreachable_code)]
    Ok(Box::new(DefaultEmbedder::new(model)?))
}

/// Build an embedder for an already-resolved profile.
pub fn embedder_for(profile: &EmbeddingProfile) -> Result<Box<dyn Embedder>> {
    profile.validate()?;
    match profile.provider.as_str() {
        "local" => embedder(EmbeddingModel::from_id(&profile.model).context("unsupported local embedding model")?),
        "voyage" => Ok(Box::new(VoyageEmbedder::new(
            profile.model.clone(),
            profile.dimensions,
        )?)),
        other => Err(anyhow!("unsupported embedding provider `{other}`")),
    }
}

/// Build the reranker for the compiled-in backend. See [`embedder`].
pub fn reranker() -> Result<Option<Box<dyn Reranker>>> {
    let provider = env_value("FUNES_RERANK_PROVIDER")?.unwrap_or_else(|| "local".to_string());
    match provider.as_str() {
        "none" => Ok(None),
        "local" => {
            #[cfg(all(feature = "metal", target_os = "macos"))]
            return Ok(Some(Box::new(metal::HybridReranker::new()?)));
            #[allow(unreachable_code)]
            Ok(Some(Box::new(DefaultReranker::new()?)))
        }
        "voyage" => Ok(Some(Box::new(VoyageReranker::new()?))),
        other => Err(anyhow!(
            "unsupported FUNES_RERANK_PROVIDER `{other}`; expected `none`, `local`, or `voyage`"
        )),
    }
}

/// Resolve a snapshot dir for `repo` (e.g. "BAAI/bge-small-en-v1.5") in the standard HF cache,
/// downloading `files`, paths within the repo, on first use.
fn snapshot(repo: &str, files: &[&str]) -> Result<PathBuf> {
    if let Some(dir) = local_snapshot(repo, files) {
        return Ok(dir);
    }
    eprintln!("downloading {repo} (first use)…");
    // A dedicated thread: the blocking hf-hub client drives its own runtime with block_on,
    // which would panic if called from a thread already inside a tokio runtime (the MCP server).
    let owned = repo.to_string();
    let files: Vec<String> = files.iter().map(|f| f.to_string()).collect();
    std::thread::spawn(move || -> Result<PathBuf> {
        let (owner, name) = owned
            .split_once('/')
            .ok_or_else(|| anyhow!("model repo {owned} is not owner/name"))?;
        let client = hf_hub::HFClientSync::new().map_err(|e| anyhow!("hf-hub client: {e}"))?;
        let r = client.model(owner, name);
        let mut dir = PathBuf::new();
        for file in &files {
            let path = r
                .download_file()
                .filename(file)
                .send()
                .with_context(|| format!("downloading {owned}/{file}"))?;
            // The snapshot holds the file as deep as its path in the repo goes: onnx/model.onnx
            // is two levels down.
            dir = path
                .ancestors()
                .nth(Path::new(file).components().count())
                .ok_or_else(|| anyhow!("{} is not in a snapshot dir", path.display()))?
                .to_path_buf();
        }
        Ok(dir)
    })
    .join()
    .map_err(|_| anyhow!("model download thread panicked"))?
}

/// A cached snapshot of `repo` holding each of `files`, if any, in the cache hf-hub downloads into.
/// With several snapshots cached the pick is arbitrary, which is fine here: any complete copy of
/// these frozen model repos is interchangeable.
fn local_snapshot(repo: &str, files: &[&str]) -> Option<PathBuf> {
    let snaps = hf_hub::resolve_cache_dir()
        .join(format!("models--{}", repo.replace('/', "--")))
        .join("snapshots");
    for e in std::fs::read_dir(&snaps).ok()?.flatten() {
        let p = e.path();
        if files.iter().all(|f| p.join(f).exists()) {
            return Some(p);
        }
    }
    None
}

/// fastembed/ort embedder on the ONNX Runtime CPU EP.
#[cfg(feature = "onnx")]
pub struct OnnxEmbedder {
    model: EmbeddingModel,
    inner: fastembed::TextEmbedding,
}

#[cfg(feature = "onnx")]
impl OnnxEmbedder {
    pub fn new(model: EmbeddingModel) -> Result<Self> {
        use fastembed::{InitOptions, TextEmbedding};
        let fastembed_model = match model {
            EmbeddingModel::BgeSmallEn => fastembed::EmbeddingModel::BGESmallENV15,
            EmbeddingModel::MultilingualE5Small => fastembed::EmbeddingModel::MultilingualE5Small,
        };
        Ok(Self {
            model,
            inner: TextEmbedding::try_new(InitOptions::new(fastembed_model).with_max_length(MAX_LEN))?,
        })
    }
}

#[cfg(feature = "onnx")]
impl Embedder for OnnxEmbedder {
    fn model(&self) -> Option<EmbeddingModel> {
        Some(self.model)
    }

    fn encode(&mut self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        self.inner.embed(texts, None)
    }
}

/// fastembed/ort reranker: the gte-reranker-modernbert-base cross-encoder, from the ONNX export its
/// repo ships, on the ONNX Runtime CPU EP.
#[cfg(feature = "onnx")]
pub struct OnnxReranker(fastembed::TextRerank);

#[cfg(feature = "onnx")]
impl OnnxReranker {
    pub fn new() -> Result<Self> {
        use fastembed::{RerankInitOptionsUserDefined, TextRerank, TokenizerFiles, UserDefinedRerankingModel};
        // fastembed does not list gte: funes fetches the export, and the four files fastembed
        // builds its tokenizer from.
        let dir = snapshot(
            "Alibaba-NLP/gte-reranker-modernbert-base",
            &[
                "onnx/model.onnx",
                "tokenizer.json",
                "config.json",
                "special_tokens_map.json",
                "tokenizer_config.json",
            ],
        )?;
        let read =
            |file: &str| std::fs::read(dir.join(file)).with_context(|| format!("reading {file} of {}", dir.display()));
        let tokenizer = TokenizerFiles {
            tokenizer_file: read("tokenizer.json")?,
            config_file: read("config.json")?,
            special_tokens_map_file: read("special_tokens_map.json")?,
            tokenizer_config_file: read("tokenizer_config.json")?,
        };
        let model = UserDefinedRerankingModel::new(dir.join("onnx/model.onnx"), tokenizer);
        let options = RerankInitOptionsUserDefined::new().with_max_length(MAX_LEN);
        Ok(Self(TextRerank::try_new_from_user_defined(model, options)?))
    }
}

#[cfg(feature = "onnx")]
impl Reranker for OnnxReranker {
    fn rerank(&mut self, query: &str, docs: &[&str]) -> Result<Vec<f32>> {
        // fastembed returns results carrying the original index; project back to input order.
        let mut scores = vec![0f32; docs.len()];
        for r in self.0.rerank(query, docs, false, None)? {
            scores[r.index] = r.score;
        }
        Ok(scores)
    }
}

/// Embed `texts` in batches of the embedder's [`Embedder::batch_size`], calling
/// `on_batch(embedded_so_far)` after each so a caller can report progress (or pass a no-op).
pub(crate) fn embed_batched(
    embedder: &mut dyn Embedder,
    texts: &[&str],
    mut on_batch: impl FnMut(usize),
) -> Result<Vec<Vec<f32>>> {
    let mut vectors: Vec<Vec<f32>> = Vec::with_capacity(texts.len());
    // Asked per call: the first call may bring up a faster backend that wants larger ones.
    while vectors.len() < texts.len() {
        let end = texts.len().min(vectors.len() + embedder.batch_size().max(1));
        let batch = embedder.embed_documents(&texts[vectors.len()..end])?;
        anyhow::ensure!(
            batch.len() == end - vectors.len(),
            "the embedder returned {} vector(s) for {} text(s)",
            batch.len(),
            end - vectors.len()
        );
        vectors.extend(batch);
        on_batch(vectors.len());
    }
    Ok(vectors)
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Recorder {
        model: EmbeddingModel,
        seen: Vec<String>,
    }

    impl Embedder for Recorder {
        fn model(&self) -> Option<EmbeddingModel> {
            Some(self.model)
        }

        fn encode(&mut self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
            self.seen.extend(texts.iter().map(|t| t.to_string()));
            Ok(vec![vec![0.0]; texts.len()])
        }
    }

    #[test]
    fn a_model_is_found_by_its_id_and_only_by_it() {
        for m in EmbeddingModel::ALL {
            assert_eq!(EmbeddingModel::from_id(m.id()), Some(m));
        }
        assert_eq!(EmbeddingModel::from_id("BAAI/bge-small-zh-v1.5"), None);
    }

    #[test]
    fn e5_encodes_prefixed_queries_and_passages() {
        let mut e = Recorder {
            model: EmbeddingModel::MultilingualE5Small,
            seen: Vec::new(),
        };
        e.embed(&["一个段落"]).unwrap();
        e.embed_query("一个问题").unwrap();
        assert_eq!(e.seen, ["passage: 一个段落", "query: 一个问题"]);
    }

    #[test]
    fn bge_encodes_queries_and_passages_as_given() {
        let mut e = Recorder {
            model: EmbeddingModel::BgeSmallEn,
            seen: Vec::new(),
        };
        e.embed(&["a passage"]).unwrap();
        e.embed_query("a question").unwrap();
        assert_eq!(e.seen, ["a passage", "a question"]);
    }

    #[test]
    fn voyage_profile_defaults_and_fingerprint_are_stable() {
        let profile = profile_from_values(Some("voyage".to_string()), None, None, None).unwrap();
        assert_eq!(profile.provider, "voyage");
        assert_eq!(profile.model, "voyage-4-lite");
        assert_eq!(profile.dimensions, 1024);
        assert_eq!(profile.schema_version, "2");
        assert_eq!(
            profile.fingerprint,
            "cef392023b4419f9d7850cc1fdeb841b40d1d648ac13431ff850f833d70b9a55"
        );
    }

    #[test]
    fn local_profile_ignores_voyage_overrides_and_stays_384_dimensions() {
        let profile = profile_from_values(
            Some("local".to_string()),
            Some("not-a-local-model".to_string()),
            Some("2048".to_string()),
            Some("99".to_string()),
        )
        .unwrap();
        assert_eq!(profile.model, "BAAI/bge-small-en-v1.5");
        assert_eq!(profile.dimensions, 384);
        assert_eq!(profile.schema_version, "1");
    }

    #[test]
    fn profile_rejects_unknown_provider_and_invalid_dimensions() {
        assert!(profile_from_values(Some("unknown".to_string()), None, None, None).is_err());
        assert!(profile_from_values(Some("voyage".to_string()), None, Some("384".to_string()), None).is_err());
        assert!(profile_from_values(Some("voyage".to_string()), None, Some("wide".to_string()), None).is_err());
        assert!(profile_from_values(Some("voyage".to_string()), None, None, Some("3".to_string())).is_err());
    }

    #[test]
    fn stored_profiles_validate_the_entire_embedding_contract() {
        for model in EmbeddingModel::ALL {
            EmbeddingProfile::local_model(model).validate().unwrap();
        }
        let mut profile = EmbeddingProfile::voyage("voyage-4-lite", 1024).unwrap();
        profile.validate().unwrap();
        profile.model = "voyage-4".into();
        assert!(
            profile.validate().is_err(),
            "same dimensions do not imply the same space"
        );
        assert!(EmbeddingProfile::new("local", "unknown", 384, "1").validate().is_err());
        assert!(EmbeddingProfile::new("voyage", "voyage-4-lite", 384, "2")
            .validate()
            .is_err());
        assert!(EmbeddingProfile::new("voyage", "voyage-4-lite", 1024, "1")
            .validate()
            .is_err());
    }
}
