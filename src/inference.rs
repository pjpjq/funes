//! The inference backend behind funes' two model operations — embedding and reranking. The rest of
//! funes talks to these traits via the [`embedder`]/[`reranker`] factories, never a concrete ML
//! stack. `FUNES_EMBEDDING_PROVIDER` chooses local or Voyage at runtime; the local implementation
//! is selected at build time by the `Default*` aliases below: default build → BLAS (a from-scratch
//! forward on Accelerate/faer); `--no-default-features --features onnx` → fastembed/ort.

#[cfg(feature = "blas")]
pub mod blas;
pub mod voyage;

use std::env;

use anyhow::{anyhow, bail, Context, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use self::voyage::{VoyageEmbedder, VoyageReranker};

// The single backend-selection point. One of these alias pairs is compiled; the factories below
// box whichever it names. When both backends are compiled in (the backend benchmark builds that
// way to use ONNX as its reference), BLAS is the one funes runs.
#[cfg(all(feature = "onnx", not(feature = "blas")))]
use self::{OnnxEmbedder as DefaultEmbedder, OnnxReranker as DefaultReranker};
#[cfg(feature = "blas")]
use blas::{BlasEmbedder as DefaultEmbedder, BlasReranker as DefaultReranker};
#[cfg(not(any(feature = "blas", feature = "onnx")))]
compile_error!("funes needs an inference backend: feature `blas` (default) or `onnx`");

/// How many texts one `embed` call takes.
const EMBED_BATCH: usize = 256;

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
                bail!(
                    "unsupported FUNES_EMBEDDING_SCHEMA_VERSION `{schema_version}`; expected `2`"
                )
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
    fn embed(&mut self, texts: &[&str]) -> Result<Vec<Vec<f32>>>;

    /// Embed persisted material. Existing local implementations retain their old `embed` path.
    fn embed_documents(&mut self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        self.embed(texts)
    }

    /// Embed a retrieval query in the provider's query input space.
    fn embed_query(&mut self, text: &str) -> Result<Vec<f32>> {
        self.embed(&[text])?
            .into_iter()
            .next()
            .context("embedding provider returned no query vector")
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

/// Build the embedder for the configured runtime profile.
pub fn embedder() -> Result<Box<dyn Embedder>> {
    embedder_for(&embedding_profile()?)
}

/// Build an embedder for an already-resolved profile.
pub fn embedder_for(profile: &EmbeddingProfile) -> Result<Box<dyn Embedder>> {
    if profile.fingerprint != profile_fingerprint(profile) {
        bail!("embedding profile fingerprint does not match its fields")
    }
    match profile.provider.as_str() {
        "local" => {
            if profile.model != LOCAL_EMBEDDING_MODEL
                || profile.dimensions != LOCAL_EMBEDDING_DIMENSIONS
                || profile.schema_version != LOCAL_EMBEDDING_SCHEMA_VERSION
            {
                bail!("unsupported local embedding profile")
            }
            Ok(Box::new(DefaultEmbedder::new()?))
        }
        "voyage" => {
            if profile.model.trim().is_empty()
                || profile.dimensions == 0
                || profile.schema_version != VOYAGE_EMBEDDING_SCHEMA_VERSION
            {
                bail!("unsupported Voyage embedding profile")
            }
            Ok(Box::new(VoyageEmbedder::new(
                profile.model.clone(),
                profile.dimensions,
            )?))
        }
        other => Err(anyhow!("unsupported embedding provider `{other}`")),
    }
}

/// Build the reranker for the compiled-in backend. See [`embedder`].
pub fn reranker() -> Result<Option<Box<dyn Reranker>>> {
    let provider = env_value("FUNES_RERANK_PROVIDER")?.unwrap_or_else(|| "none".to_string());
    match provider.as_str() {
        "none" => Ok(None),
        "local" => Ok(Some(Box::new(DefaultReranker::new()?))),
        "voyage" => Ok(Some(Box::new(VoyageReranker::new()?))),
        other => Err(anyhow!(
            "unsupported FUNES_RERANK_PROVIDER `{other}`; expected `none`, `local`, or `voyage`"
        )),
    }
}

/// fastembed/ort embedder: BAAI/bge-small-en-v1.5 on the ONNX Runtime CPU EP.
#[cfg(feature = "onnx")]
pub struct OnnxEmbedder(fastembed::TextEmbedding);

#[cfg(feature = "onnx")]
impl OnnxEmbedder {
    pub fn new() -> Result<Self> {
        use fastembed::{EmbeddingModel, InitOptions, TextEmbedding};
        Ok(Self(TextEmbedding::try_new(InitOptions::new(
            EmbeddingModel::BGESmallENV15,
        ))?))
    }
}

#[cfg(feature = "onnx")]
impl Embedder for OnnxEmbedder {
    fn embed(&mut self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        self.0.embed(texts, None)
    }
}

/// fastembed/ort reranker: BAAI/bge-reranker-base cross-encoder on the ONNX Runtime CPU EP.
#[cfg(feature = "onnx")]
pub struct OnnxReranker(fastembed::TextRerank);

#[cfg(feature = "onnx")]
impl OnnxReranker {
    pub fn new() -> Result<Self> {
        use fastembed::{RerankInitOptions, RerankerModel, TextRerank};
        Ok(Self(TextRerank::try_new(RerankInitOptions::new(
            RerankerModel::BGERerankerBase,
        ))?))
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

/// Embed `texts` in batches of [`EMBED_BATCH`], calling `on_batch(embedded_so_far)` after each so a
/// caller can report progress (or pass a no-op).
pub(crate) fn embed_batched(
    embedder: &mut dyn Embedder,
    texts: &[&str],
    mut on_batch: impl FnMut(usize),
) -> Result<Vec<Vec<f32>>> {
    let mut vectors: Vec<Vec<f32>> = Vec::with_capacity(texts.len());
    for group in texts.chunks(EMBED_BATCH) {
        vectors.extend(embedder.embed_documents(group)?);
        on_batch(vectors.len());
    }
    Ok(vectors)
}

#[cfg(test)]
mod tests {
    use super::*;

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
        assert!(profile_from_values(
            Some("voyage".to_string()),
            None,
            Some("384".to_string()),
            None
        )
        .is_err());
        assert!(profile_from_values(
            Some("voyage".to_string()),
            None,
            Some("wide".to_string()),
            None
        )
        .is_err());
        assert!(profile_from_values(
            Some("voyage".to_string()),
            None,
            None,
            Some("3".to_string())
        )
        .is_err());
    }
}
