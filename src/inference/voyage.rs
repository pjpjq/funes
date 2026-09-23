//! Voyage AI's native HTTP inference backend.
//!
//! The API key stays inside request headers and errors deliberately report only status/error
//! classes, never the response body. Document embeddings retry a small bounded set of transient
//! responses; query-time operations use one request with a short total timeout.

use std::env;
use std::thread;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use anyhow::{anyhow, bail, Result};
use reqwest::blocking::Client;
use reqwest::header::HeaderMap;
use reqwest::redirect::Policy;
use reqwest::StatusCode;
use serde::{Deserialize, Serialize};

use super::{voyage_dimensions_supported, Embedder, RerankScoreKind, Reranker};

const EMBEDDINGS_URL: &str = "https://api.voyageai.com/v1/embeddings";
const RERANK_URL: &str = "https://api.voyageai.com/v1/rerank";
const RERANK_MODEL: &str = "rerank-3-lite";
const DOCUMENT_TIMEOUT: Duration = Duration::from_secs(30);
const QUERY_TIMEOUT: Duration = Duration::from_secs(3);
const DOCUMENT_ATTEMPTS: usize = 5;
const DOCUMENT_RETRY_DELAY: Duration = Duration::from_millis(200);
const DOCUMENT_RATE_LIMIT_DELAY: Duration = Duration::from_secs(60);
// Voyage's embeddings endpoint accepts at most 128 input strings per request.
// Keep this provider-specific guard here so callers may continue batching local
// backends more aggressively without ever sending an invalid Voyage payload.
const MAX_EMBEDDING_INPUTS: usize = 128;
// Four capped sleeps plus five 30-second requests stay below the Space's
// 900-second canonical-ingest subprocess deadline.
const DOCUMENT_MAX_RATE_LIMIT_DELAY: Duration = Duration::from_secs(120);

#[derive(Serialize)]
struct EmbeddingsRequest<'a> {
    input: Vec<&'a str>,
    model: &'a str,
    input_type: &'static str,
    output_dimension: usize,
    truncation: bool,
}

#[derive(Deserialize)]
struct EmbeddingsResponse {
    data: Vec<EmbeddingData>,
}

#[derive(Deserialize)]
struct EmbeddingData {
    embedding: Vec<f32>,
    index: usize,
}

#[derive(Serialize)]
struct RerankRequest<'a> {
    query: &'a str,
    documents: Vec<&'a str>,
    model: &'a str,
    return_documents: bool,
    truncation: bool,
}

#[derive(Deserialize)]
struct RerankResponse {
    data: Vec<RerankData>,
}

#[derive(Deserialize)]
struct RerankData {
    index: usize,
    relevance_score: f32,
}

fn api_key_from(value: Option<String>) -> Result<String> {
    let key = value.ok_or_else(|| anyhow!("VOYAGE_API_KEY is required for Voyage inference"))?;
    let key = key.trim();
    if key.is_empty() {
        bail!("VOYAGE_API_KEY is required for Voyage inference")
    }
    Ok(key.to_string())
}

fn api_key() -> Result<String> {
    let value = match env::var("VOYAGE_API_KEY") {
        Ok(value) => Some(value),
        Err(env::VarError::NotPresent) => None,
        Err(env::VarError::NotUnicode(_)) => {
            bail!("VOYAGE_API_KEY is not valid UTF-8")
        }
    };
    api_key_from(value)
}

fn rerank_model_from(value: Option<String>) -> Result<String> {
    let model = value.unwrap_or_else(|| RERANK_MODEL.to_string());
    if model.trim().is_empty() {
        bail!("FUNES_RERANK_MODEL must not be empty")
    }
    Ok(model.trim().to_string())
}

fn rerank_model() -> Result<String> {
    let value = match env::var("FUNES_RERANK_MODEL") {
        Ok(value) => Some(value),
        Err(env::VarError::NotPresent) => None,
        Err(env::VarError::NotUnicode(_)) => {
            bail!("FUNES_RERANK_MODEL is not valid UTF-8")
        }
    };
    rerank_model_from(value)
}

fn http_client() -> Result<Client> {
    Client::builder()
        .redirect(Policy::none())
        .connect_timeout(QUERY_TIMEOUT)
        .build()
        .map_err(|_| anyhow!("failed to build Voyage HTTP client"))
}

/// Keep reqwest's blocking runtime entirely outside a Tokio worker. The inference traits are
/// synchronous, so provider construction and each request run on a scoped OS thread. The retained
/// client owns its connection pool across calls instead of rebuilding TLS state for every request.
fn run_blocking_http<T: Send>(operation: &str, work: impl FnOnce() -> Result<T> + Send) -> Result<T> {
    thread::scope(|scope| {
        scope
            .spawn(work)
            .join()
            .map_err(|_| anyhow!("Voyage {operation} HTTP worker panicked"))?
    })
}

fn request_error(operation: &str, error: &reqwest::Error) -> anyhow::Error {
    if error.is_timeout() {
        anyhow!("Voyage {operation} request timed out")
    } else if error.is_connect() {
        anyhow!("Voyage {operation} connection failed")
    } else {
        anyhow!("Voyage {operation} request failed")
    }
}

fn retryable(status: StatusCode) -> bool {
    status == StatusCode::TOO_MANY_REQUESTS || status.is_server_error()
}

fn retryable_request_error(error: &reqwest::Error) -> bool {
    error.is_timeout() || error.is_connect() || error.is_body()
}

fn retry_after_delay(value: &str, now: SystemTime) -> Option<Duration> {
    if let Ok(seconds) = value.trim().parse::<u64>() {
        return Some(Duration::from_secs(seconds));
    }
    let reset = chrono::DateTime::parse_from_rfc2822(value.trim()).ok()?.timestamp();
    let now = now.duration_since(UNIX_EPOCH).unwrap_or_default().as_secs();
    Some(Duration::from_secs(reset.saturating_sub(now as i64).max(0) as u64))
}

fn epoch_reset_delay(value: &str, now: SystemTime) -> Option<Duration> {
    let reset = value.trim().parse::<u64>().ok()?;
    let now = now.duration_since(UNIX_EPOCH).unwrap_or_default().as_secs();
    Some(Duration::from_secs(reset.saturating_sub(now)))
}

fn exponential_rate_limit_delay(base: Duration, attempt: usize) -> Duration {
    let multiplier = 1u32.checked_shl(attempt as u32).unwrap_or(u32::MAX);
    base.saturating_mul(multiplier).min(DOCUMENT_MAX_RATE_LIMIT_DELAY)
}

fn rate_limit_delay_from_headers(headers: &HeaderMap, now: SystemTime, fallback: Duration) -> Duration {
    headers
        .get(reqwest::header::RETRY_AFTER)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| retry_after_delay(value, now))
        .or_else(|| {
            headers
                .get("x-ratelimit-reset")
                .and_then(|value| value.to_str().ok())
                .and_then(|value| epoch_reset_delay(value, now))
        })
        .unwrap_or(fallback)
        .min(DOCUMENT_MAX_RATE_LIMIT_DELAY)
}

fn normalize(vector: &mut [f32]) -> Result<()> {
    if vector.iter().any(|value| !value.is_finite()) {
        bail!("Voyage embeddings returned a non-finite vector")
    }
    let norm = vector
        .iter()
        .map(|value| f64::from(*value) * f64::from(*value))
        .sum::<f64>()
        .sqrt();
    if !norm.is_finite() || norm == 0.0 {
        bail!("Voyage embeddings returned a zero-length vector")
    }
    for value in vector {
        *value = (f64::from(*value) / norm) as f32;
    }
    Ok(())
}

fn decode_embeddings(response: EmbeddingsResponse, expected: usize, dimensions: usize) -> Result<Vec<Vec<f32>>> {
    if response.data.len() != expected {
        bail!(
            "Voyage embeddings returned {} vectors for {expected} inputs",
            response.data.len()
        )
    }

    let mut vectors = vec![None; expected];
    for mut item in response.data {
        if item.index >= expected || vectors[item.index].is_some() {
            bail!("Voyage embeddings returned invalid vector indexes")
        }
        if item.embedding.len() != dimensions {
            bail!(
                "Voyage embeddings returned dimension {}, expected {dimensions}",
                item.embedding.len()
            )
        }
        normalize(&mut item.embedding)?;
        vectors[item.index] = Some(item.embedding);
    }
    vectors
        .into_iter()
        .map(|vector| vector.ok_or_else(|| anyhow!("Voyage embeddings omitted a vector")))
        .collect()
}

/// Voyage embeddings with explicit document/query modes.
pub struct VoyageEmbedder {
    client: Client,
    api_key: String,
    endpoint: String,
    model: String,
    dimensions: usize,
    document_timeout: Duration,
    query_timeout: Duration,
    document_attempts: usize,
    document_retry_delay: Duration,
    document_rate_limit_delay: Duration,
}

impl VoyageEmbedder {
    pub fn new(model: String, dimensions: usize) -> Result<Self> {
        Self::with_config(
            api_key()?,
            EMBEDDINGS_URL.to_string(),
            model,
            dimensions,
            DOCUMENT_TIMEOUT,
            QUERY_TIMEOUT,
            DOCUMENT_ATTEMPTS,
            DOCUMENT_RETRY_DELAY,
            DOCUMENT_RATE_LIMIT_DELAY,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn with_config(
        api_key: String,
        endpoint: String,
        model: String,
        dimensions: usize,
        document_timeout: Duration,
        query_timeout: Duration,
        document_attempts: usize,
        document_retry_delay: Duration,
        document_rate_limit_delay: Duration,
    ) -> Result<Self> {
        if model.trim().is_empty() {
            bail!("Voyage embedding model must not be empty")
        }
        if !voyage_dimensions_supported(dimensions) {
            bail!("Voyage embedding dimensions must be one of 256, 512, 1024, or 2048")
        }
        if document_attempts == 0 {
            bail!("Voyage document attempts must be positive")
        }
        let client = run_blocking_http("client setup", http_client)?;
        Ok(Self {
            client,
            api_key,
            endpoint,
            model,
            dimensions,
            document_timeout,
            query_timeout,
            document_attempts,
            document_retry_delay,
            document_rate_limit_delay,
        })
    }

    fn rate_limit_delay(&self, response: &reqwest::blocking::Response, attempt: usize) -> Duration {
        rate_limit_delay_from_headers(
            response.headers(),
            SystemTime::now(),
            exponential_rate_limit_delay(self.document_rate_limit_delay, attempt),
        )
    }

    fn request(&self, texts: &[&str], input_type: &'static str, documents: bool) -> Result<Vec<Vec<f32>>> {
        run_blocking_http("embeddings", || self.request_blocking(texts, input_type, documents))
    }

    fn request_blocking(&self, texts: &[&str], input_type: &'static str, documents: bool) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        let request = EmbeddingsRequest {
            input: texts.to_vec(),
            model: &self.model,
            input_type,
            output_dimension: self.dimensions,
            truncation: true,
        };
        let attempts = if documents { self.document_attempts } else { 1 };
        let timeout = if documents {
            self.document_timeout
        } else {
            self.query_timeout
        };

        for attempt in 0..attempts {
            let response = self
                .client
                .post(&self.endpoint)
                .timeout(timeout)
                .bearer_auth(&self.api_key)
                .json(&request)
                .send();
            let response = match response {
                Ok(response) => response,
                Err(error) if documents && retryable_request_error(&error) && attempt + 1 < attempts => {
                    thread::sleep(self.document_retry_delay.saturating_mul((attempt + 1) as u32));
                    continue;
                }
                Err(error) => return Err(request_error("embeddings", &error)),
            };
            let status = response.status();
            if status.is_success() {
                let response = match response.json::<EmbeddingsResponse>() {
                    Ok(response) => response,
                    Err(error) if documents && retryable_request_error(&error) && attempt + 1 < attempts => {
                        thread::sleep(self.document_retry_delay.saturating_mul((attempt + 1) as u32));
                        continue;
                    }
                    Err(error) if retryable_request_error(&error) => return Err(request_error("embeddings", &error)),
                    Err(_) => bail!("Voyage embeddings returned invalid JSON"),
                };
                return decode_embeddings(response, texts.len(), self.dimensions);
            }
            if documents && retryable(status) && attempt + 1 < attempts {
                let delay = if status == StatusCode::TOO_MANY_REQUESTS {
                    self.rate_limit_delay(&response, attempt)
                } else {
                    self.document_retry_delay.saturating_mul((attempt + 1) as u32)
                };
                thread::sleep(delay);
                continue;
            }
            bail!("Voyage embeddings request failed with HTTP {}", status.as_u16())
        }
        unreachable!("positive attempt count always returns")
    }
}

impl Embedder for VoyageEmbedder {
    /// Compatibility means a legacy call remains a document embedding, never an ambiguous query.
    fn embed(&mut self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        self.embed_documents(texts)
    }

    fn embed_documents(&mut self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        let mut vectors = Vec::with_capacity(texts.len());
        for group in texts.chunks(MAX_EMBEDDING_INPUTS) {
            vectors.extend(self.request(group, "document", true)?);
        }
        Ok(vectors)
    }

    fn embed_query(&mut self, text: &str) -> Result<Vec<f32>> {
        self.request(&[text], "query", false)?
            .into_iter()
            .next()
            .ok_or_else(|| anyhow!("Voyage embeddings returned no query vector"))
    }
}

fn decode_rerank(response: RerankResponse, expected: usize) -> Result<Vec<f32>> {
    if response.data.len() != expected {
        bail!(
            "Voyage rerank returned {} scores for {expected} documents",
            response.data.len()
        )
    }

    let mut scores = vec![None; expected];
    for item in response.data {
        if item.index >= expected || scores[item.index].is_some() {
            bail!("Voyage rerank returned invalid document indexes")
        }
        if !item.relevance_score.is_finite() {
            bail!("Voyage rerank returned a non-finite score")
        }
        scores[item.index] = Some(item.relevance_score);
    }
    scores
        .into_iter()
        .map(|score| score.ok_or_else(|| anyhow!("Voyage rerank omitted a score")))
        .collect()
}

/// Voyage `rerank-3-lite`, bounded to one query-time request.
pub struct VoyageReranker {
    client: Client,
    api_key: String,
    endpoint: String,
    model: String,
    query_timeout: Duration,
}

impl VoyageReranker {
    pub fn new() -> Result<Self> {
        Self::with_config(api_key()?, RERANK_URL.to_string(), rerank_model()?, QUERY_TIMEOUT)
    }

    fn with_config(api_key: String, endpoint: String, model: String, query_timeout: Duration) -> Result<Self> {
        if model.trim().is_empty() {
            bail!("Voyage rerank model must not be empty")
        }
        let client = run_blocking_http("client setup", http_client)?;
        Ok(Self {
            client,
            api_key,
            endpoint,
            model,
            query_timeout,
        })
    }
}

impl Reranker for VoyageReranker {
    fn rerank(&mut self, query: &str, docs: &[&str]) -> Result<Vec<f32>> {
        run_blocking_http("rerank", || self.rerank_blocking(query, docs))
    }

    fn score_kind(&self) -> RerankScoreKind {
        RerankScoreKind::Relevance
    }
}

impl VoyageReranker {
    fn rerank_blocking(&self, query: &str, docs: &[&str]) -> Result<Vec<f32>> {
        if docs.is_empty() {
            return Ok(Vec::new());
        }
        let request = RerankRequest {
            query,
            documents: docs.to_vec(),
            model: &self.model,
            return_documents: false,
            truncation: true,
        };
        let response = self
            .client
            .post(&self.endpoint)
            .timeout(self.query_timeout)
            .bearer_auth(&self.api_key)
            .json(&request)
            .send()
            .map_err(|error| request_error("rerank", &error))?;
        let status = response.status();
        if !status.is_success() {
            bail!("Voyage rerank request failed with HTTP {}", status.as_u16())
        }
        let response = response.json::<RerankResponse>().map_err(|error| {
            if retryable_request_error(&error) {
                request_error("rerank", &error)
            } else {
                anyhow!("Voyage rerank returned invalid JSON")
            }
        })?;
        decode_rerank(response, docs.len())
    }
}

#[cfg(test)]
mod tests {
    use std::io::{Read, Write};
    use std::net::{TcpListener, TcpStream};
    use std::sync::{Arc, Mutex};
    use std::thread::JoinHandle;
    use std::time::Instant;

    use serde_json::{json, Value};

    use super::*;

    const TEST_DIMENSIONS: usize = 256;

    struct MockResponse {
        status: u16,
        body: String,
        delay: Duration,
        headers: Vec<(&'static str, &'static str)>,
    }

    impl MockResponse {
        fn json(status: u16, body: Value) -> Self {
            Self {
                status,
                body: body.to_string(),
                delay: Duration::ZERO,
                headers: Vec::new(),
            }
        }

        fn delayed(status: u16, body: Value, delay: Duration) -> Self {
            Self {
                status,
                body: body.to_string(),
                delay,
                headers: Vec::new(),
            }
        }

        fn with_header(mut self, name: &'static str, value: &'static str) -> Self {
            self.headers.push((name, value));
            self
        }
    }

    #[derive(Clone)]
    struct CapturedRequest {
        headers: String,
        body: Value,
    }

    struct MockServer {
        url: String,
        requests: Arc<Mutex<Vec<CapturedRequest>>>,
        handle: Option<JoinHandle<()>>,
    }

    impl MockServer {
        fn start(responses: Vec<MockResponse>) -> Self {
            let listener = TcpListener::bind("127.0.0.1:0").unwrap();
            let url = format!("http://{}", listener.local_addr().unwrap());
            let requests = Arc::new(Mutex::new(Vec::new()));
            let captured = Arc::clone(&requests);
            let handle = std::thread::spawn(move || {
                let mut handlers = Vec::new();
                for response in responses {
                    let (mut stream, _) = listener.accept().unwrap();
                    let captured = Arc::clone(&captured);
                    handlers.push(std::thread::spawn(move || {
                        let request = read_request(&mut stream);
                        captured.lock().unwrap().push(request);
                        std::thread::sleep(response.delay);
                        let reason = if response.status == 200 { "OK" } else { "Error" };
                        let headers = response
                            .headers
                            .iter()
                            .map(|(name, value)| format!("{name}: {value}\r\n"))
                            .collect::<String>();
                        let wire = format!(
                            "HTTP/1.1 {} {}\r\nContent-Type: application/json\r\n{}Content-Length: {}\r\nConnection: close\r\n\r\n{}",
                            response.status,
                            reason,
                            headers,
                            response.body.len(),
                            response.body
                        );
                        let _ = stream.write_all(wire.as_bytes());
                    }));
                }
                for handler in handlers {
                    handler.join().unwrap();
                }
            });
            Self {
                url,
                requests,
                handle: Some(handle),
            }
        }

        fn finish(mut self) -> Vec<CapturedRequest> {
            self.handle.take().unwrap().join().unwrap();
            self.requests.lock().unwrap().clone()
        }
    }

    fn read_request(stream: &mut TcpStream) -> CapturedRequest {
        stream.set_read_timeout(Some(Duration::from_secs(2))).unwrap();
        let mut wire = Vec::new();
        let mut buffer = [0u8; 4096];
        let (header_end, content_length) = loop {
            let read = stream.read(&mut buffer).unwrap();
            assert!(read > 0, "request ended before headers");
            wire.extend_from_slice(&buffer[..read]);
            if let Some(header_end) = wire.windows(4).position(|window| window == b"\r\n\r\n") {
                let headers = String::from_utf8(wire[..header_end].to_vec()).unwrap();
                let content_length = headers
                    .lines()
                    .find_map(|line| {
                        let (name, value) = line.split_once(':')?;
                        name.eq_ignore_ascii_case("content-length")
                            .then(|| value.trim().parse::<usize>().unwrap())
                    })
                    .unwrap();
                break (header_end + 4, content_length);
            }
        };
        while wire.len() < header_end + content_length {
            let read = stream.read(&mut buffer).unwrap();
            assert!(read > 0, "request ended before body");
            wire.extend_from_slice(&buffer[..read]);
        }
        let headers = String::from_utf8(wire[..header_end - 4].to_vec()).unwrap();
        let body = serde_json::from_slice(&wire[header_end..header_end + content_length]).unwrap();
        CapturedRequest { headers, body }
    }

    fn test_embedder(server: &MockServer, dimensions: usize) -> VoyageEmbedder {
        VoyageEmbedder::with_config(
            "test-api-key".to_string(),
            server.url.clone(),
            "test-embedding-model".to_string(),
            dimensions,
            Duration::from_secs(1),
            Duration::from_millis(100),
            DOCUMENT_ATTEMPTS,
            Duration::ZERO,
            Duration::ZERO,
        )
        .unwrap()
    }

    fn unit_vector(position: usize) -> Vec<f32> {
        let mut vector = vec![0.0; TEST_DIMENSIONS];
        vector[position] = 1.0;
        vector
    }

    fn embedding_response(vectors: Vec<(usize, Vec<f32>)>) -> Value {
        json!({
            "data": vectors
                .into_iter()
                .map(|(index, embedding)| json!({ "index": index, "embedding": embedding }))
                .collect::<Vec<_>>()
        })
    }

    #[test]
    fn retained_clients_are_send_and_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<VoyageEmbedder>();
        assert_send_sync::<VoyageReranker>();
    }

    #[tokio::test]
    async fn retained_blocking_client_lifecycle_is_safe_inside_tokio() {
        let server = MockServer::start(vec![MockResponse::json(
            200,
            embedding_response(vec![(0, unit_vector(0))]),
        )]);

        let mut embedder = test_embedder(&server, TEST_DIMENSIONS);
        assert_eq!(embedder.embed_query("needle").unwrap(), unit_vector(0));
        drop(embedder);

        assert_eq!(server.finish().len(), 1);
    }

    #[test]
    fn document_and_query_payloads_are_distinct_and_dimensioned() {
        let server = MockServer::start(vec![
            MockResponse::json(200, embedding_response(vec![(1, unit_vector(1)), (0, unit_vector(0))])),
            MockResponse::json(200, embedding_response(vec![(0, unit_vector(2))])),
        ]);
        let mut embedder = test_embedder(&server, TEST_DIMENSIONS);

        let documents = embedder.embed_documents(&["first", "second"]).unwrap();
        let query = embedder.embed_query("needle").unwrap();
        assert_eq!(documents, vec![unit_vector(0), unit_vector(1)]);
        assert_eq!(query, unit_vector(2));

        let requests = server.finish();
        assert_eq!(requests.len(), 2);
        assert!(requests[0]
            .headers
            .to_ascii_lowercase()
            .contains("authorization: bearer test-api-key"));
        assert_eq!(requests[0].body["input"], json!(["first", "second"]));
        assert_eq!(requests[0].body["model"], "test-embedding-model");
        assert_eq!(requests[0].body["input_type"], "document");
        assert_eq!(requests[0].body["output_dimension"], TEST_DIMENSIONS);
        assert_eq!(requests[0].body["truncation"], true);
        assert_eq!(requests[1].body["input"], json!(["needle"]));
        assert_eq!(requests[1].body["input_type"], "query");
        assert_eq!(requests[1].body["output_dimension"], TEST_DIMENSIONS);
        assert_eq!(requests[1].body["truncation"], true);
    }

    #[test]
    fn document_batches_are_split_at_voyage_input_limit() {
        let first = (0..MAX_EMBEDDING_INPUTS)
            .map(|index| (index, unit_vector(index % TEST_DIMENSIONS)))
            .collect();
        let server = MockServer::start(vec![
            MockResponse::json(200, embedding_response(first)),
            MockResponse::json(200, embedding_response(vec![(0, unit_vector(0))])),
        ]);
        let mut embedder = test_embedder(&server, TEST_DIMENSIONS);
        let texts: Vec<String> = (0..MAX_EMBEDDING_INPUTS + 1)
            .map(|index| format!("document-{index}"))
            .collect();
        let refs: Vec<&str> = texts.iter().map(String::as_str).collect();

        let vectors = embedder.embed_documents(&refs).unwrap();
        assert_eq!(vectors.len(), MAX_EMBEDDING_INPUTS + 1);

        let requests = server.finish();
        assert_eq!(requests.len(), 2);
        assert_eq!(requests[0].body["input"].as_array().unwrap().len(), MAX_EMBEDDING_INPUTS);
        assert_eq!(requests[1].body["input"].as_array().unwrap().len(), 1);
    }

    #[test]
    fn missing_key_and_wrong_response_dimension_are_rejected() {
        let missing = api_key_from(None).unwrap_err().to_string();
        assert!(missing.contains("VOYAGE_API_KEY"));

        let server = MockServer::start(vec![MockResponse::json(
            200,
            embedding_response(vec![(0, vec![1.0; TEST_DIMENSIONS - 1])]),
        )]);
        let mut embedder = test_embedder(&server, TEST_DIMENSIONS);
        let error = embedder.embed_documents(&["document"]).unwrap_err().to_string();
        assert!(error.contains("dimension 255, expected 256"));
        assert_eq!(server.finish().len(), 1);
    }

    #[test]
    fn documents_retry_only_429_and_5xx_with_a_bounded_attempt_count() {
        let server = MockServer::start(vec![
            MockResponse::json(429, json!({ "private": "response-one" })),
            MockResponse::json(503, json!({ "private": "response-two" })),
            MockResponse::json(200, embedding_response(vec![(0, unit_vector(0))])),
        ]);
        let mut embedder = test_embedder(&server, TEST_DIMENSIONS);
        assert_eq!(embedder.embed_documents(&["document"]).unwrap(), vec![unit_vector(0)]);
        assert_eq!(server.finish().len(), 3);

        let exhausted = MockServer::start(vec![
            MockResponse::json(500, json!({ "private": "first-body" })),
            MockResponse::json(502, json!({ "private": "second-body" })),
            MockResponse::json(503, json!({ "private": "third-body" })),
            MockResponse::json(504, json!({ "private": "fourth-body" })),
            MockResponse::json(599, json!({ "private": "last-body" })),
        ]);
        let mut embedder = test_embedder(&exhausted, TEST_DIMENSIONS);
        let error = embedder.embed_documents(&["document"]).unwrap_err().to_string();
        assert!(error.contains("HTTP 599"));
        assert!(!error.contains("last-body"));
        assert_eq!(exhausted.finish().len(), DOCUMENT_ATTEMPTS);
    }

    #[test]
    fn retry_delay_accepts_http_date_and_rate_limit_reset() {
        let now = UNIX_EPOCH + Duration::from_secs(1_600_000_000);
        let mut headers = HeaderMap::new();
        headers.insert(
            reqwest::header::RETRY_AFTER,
            "Sun, 13 Sep 2020 12:27:00 GMT".parse().unwrap(),
        );
        assert_eq!(
            rate_limit_delay_from_headers(&headers, now, Duration::from_secs(60)),
            Duration::from_secs(20)
        );

        headers.remove(reqwest::header::RETRY_AFTER);
        headers.insert("x-ratelimit-reset", "1600000060".parse().unwrap());
        assert_eq!(
            rate_limit_delay_from_headers(&headers, now, Duration::from_secs(120)),
            Duration::from_secs(60)
        );
    }

    #[test]
    fn retry_delay_uses_capped_exponential_fallback() {
        assert_eq!(
            exponential_rate_limit_delay(Duration::from_secs(10), 0),
            Duration::from_secs(10)
        );
        assert_eq!(
            exponential_rate_limit_delay(Duration::from_secs(10), 2),
            Duration::from_secs(40)
        );
        assert_eq!(
            exponential_rate_limit_delay(Duration::from_secs(60), 4),
            DOCUMENT_MAX_RATE_LIMIT_DELAY
        );
    }

    #[test]
    fn document_429_honors_retry_after_without_using_the_fallback_delay() {
        let server = MockServer::start(vec![
            MockResponse::json(429, json!({ "private": "rate-limited" })).with_header("Retry-After", "0"),
            MockResponse::json(200, embedding_response(vec![(0, unit_vector(0))])),
        ]);
        let mut embedder = test_embedder(&server, TEST_DIMENSIONS);
        embedder.document_rate_limit_delay = Duration::from_millis(250);
        let started = Instant::now();
        assert_eq!(embedder.embed_documents(&["document"]).unwrap(), vec![unit_vector(0)]);
        assert!(started.elapsed() < Duration::from_millis(200));
        assert_eq!(server.finish().len(), 2);
    }

    #[test]
    fn document_429_without_retry_after_uses_the_rate_limit_delay() {
        let server = MockServer::start(vec![
            MockResponse::json(429, json!({ "private": "rate-limited" })),
            MockResponse::json(200, embedding_response(vec![(0, unit_vector(0))])),
        ]);
        let mut embedder = test_embedder(&server, TEST_DIMENSIONS);
        embedder.document_rate_limit_delay = Duration::from_millis(40);
        let started = Instant::now();
        assert_eq!(embedder.embed_documents(&["document"]).unwrap(), vec![unit_vector(0)]);
        assert!(started.elapsed() >= Duration::from_millis(35));
        assert!(started.elapsed() < Duration::from_millis(500));
        assert_eq!(server.finish().len(), 2);
    }

    #[test]
    fn documents_retry_a_transport_timeout_but_queries_do_not() {
        let server = MockServer::start(vec![
            MockResponse::delayed(
                200,
                embedding_response(vec![(0, unit_vector(0))]),
                Duration::from_millis(200),
            ),
            MockResponse::json(200, embedding_response(vec![(0, unit_vector(1))])),
        ]);
        let mut embedder = test_embedder(&server, TEST_DIMENSIONS);
        embedder.document_timeout = Duration::from_millis(40);
        assert_eq!(embedder.embed_documents(&["document"]).unwrap(), vec![unit_vector(1)]);
        assert_eq!(server.finish().len(), 2);
    }

    #[test]
    fn query_does_not_retry_and_honors_a_short_total_timeout() {
        assert!(QUERY_TIMEOUT <= Duration::from_millis(3500));

        let failed = MockServer::start(vec![MockResponse::json(
            503,
            json!({ "private": "do-not-return-this-body" }),
        )]);
        let mut embedder = test_embedder(&failed, TEST_DIMENSIONS);
        let error = embedder.embed_query("needle").unwrap_err().to_string();
        assert!(error.contains("HTTP 503"));
        assert!(!error.contains("do-not-return-this-body"));
        assert_eq!(failed.finish().len(), 1);

        let rate_limited = MockServer::start(vec![MockResponse::json(
            429,
            json!({ "private": "do-not-return-this-body" }),
        )]);
        let mut embedder = test_embedder(&rate_limited, TEST_DIMENSIONS);
        let started = Instant::now();
        let error = embedder.embed_query("needle").unwrap_err().to_string();
        assert!(error.contains("HTTP 429"));
        assert!(started.elapsed() < Duration::from_millis(200));
        assert_eq!(rate_limited.finish().len(), 1);

        let slow = MockServer::start(vec![MockResponse::delayed(
            200,
            embedding_response(vec![(0, unit_vector(0))]),
            Duration::from_millis(750),
        )]);
        let mut embedder = test_embedder(&slow, TEST_DIMENSIONS);
        embedder.query_timeout = Duration::from_millis(40);
        let started = Instant::now();
        let error = embedder.embed_query("needle").unwrap_err().to_string();
        assert!(error.contains("timed out"));
        assert!(started.elapsed() < Duration::from_millis(400));
        assert_eq!(slow.finish().len(), 1);
    }

    #[test]
    fn rerank_model_is_configurable_and_scores_restore_input_order() {
        assert_eq!(rerank_model_from(None).unwrap(), "rerank-3-lite");
        assert_eq!(
            rerank_model_from(Some(" custom-rerank ".to_string())).unwrap(),
            "custom-rerank"
        );
        let server = MockServer::start(vec![MockResponse::json(
            200,
            json!({
                "data": [
                    { "index": 1, "relevance_score": 0.9 },
                    { "index": 0, "relevance_score": 0.1 }
                ]
            }),
        )]);
        let mut reranker = VoyageReranker::with_config(
            "test-api-key".to_string(),
            server.url.clone(),
            "custom-rerank".to_string(),
            Duration::from_millis(100),
        )
        .unwrap();
        assert_eq!(reranker.score_kind(), RerankScoreKind::Relevance);
        assert_eq!(reranker.rerank("needle", &["first", "second"]).unwrap(), [0.1, 0.9]);

        let requests = server.finish();
        assert_eq!(requests.len(), 1);
        assert_eq!(requests[0].body["query"], "needle");
        assert_eq!(requests[0].body["documents"], json!(["first", "second"]));
        assert_eq!(requests[0].body["model"], "custom-rerank");
        assert_eq!(requests[0].body["return_documents"], false);
        assert_eq!(requests[0].body["truncation"], true);
    }
}
