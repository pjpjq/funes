//! Token-free metadata and strict-query regression tests; every HTTP request stays on loopback.

use std::collections::BTreeMap;
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::Path;
use std::process::{Command, Output, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use arrow_array::types::Float32Type;
use arrow_array::{FixedSizeListArray, RecordBatch, RecordBatchIterator};
use arrow_schema::{DataType, Field, Schema};
use funes::memory::{dataset, Memory};
use lance::dataset::{Dataset, WriteParams};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

const OLD_SHA: &str = "1111111111111111111111111111111111111111";
const HEAD_SHA: &str = "2222222222222222222222222222222222222222";
const BAD_SHA: &str = "3333333333333333333333333333333333333333";

type Files = BTreeMap<String, Vec<u8>>;

fn batch(dimension: i32) -> RecordBatchIterator<Vec<Result<RecordBatch, arrow_schema::ArrowError>>> {
    let schema = Arc::new(Schema::new(vec![Field::new(
        "vector",
        DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, true)), dimension),
        true,
    )]));
    let vectors = FixedSizeListArray::from_iter_primitive::<Float32Type, _, _>(
        [Some(vec![Some(0.0); dimension as usize])],
        dimension,
    );
    let rows = RecordBatch::try_new(schema.clone(), vec![Arc::new(vectors)]).unwrap();
    RecordBatchIterator::new(vec![Ok(rows)], schema)
}

fn files(root: &Path, version: u64) -> Files {
    let mut pending = vec![root.to_path_buf()];
    let mut files = BTreeMap::new();
    while let Some(dir) = pending.pop() {
        for entry in std::fs::read_dir(dir).unwrap() {
            let path = entry.unwrap().path();
            if path.is_dir() {
                pending.push(path);
            } else {
                let logical = path.strip_prefix(root).unwrap().to_str().unwrap().to_string();
                files.insert(logical, std::fs::read(path).unwrap());
            }
        }
    }
    files.insert(
        "chunks.lance/_versions/latest_version_hint.json".to_string(),
        serde_json::to_vec(&json!({"version": version})).unwrap(),
    );
    // The read wrapper deliberately ignores flat hints, so use production's physical layout.
    files
        .into_iter()
        .map(|(logical, body)| {
            let hash = hex::encode(Sha256::digest(logical.as_bytes()));
            (format!("__funes_shards__/v1/{}/{logical}", &hash[..2]), body)
        })
        .collect()
}

struct Hub {
    endpoint: String,
    requests: Arc<Mutex<Vec<String>>>,
    stop: Arc<AtomicBool>,
    server: Option<thread::JoinHandle<()>>,
}

impl Hub {
    fn new(snapshots: BTreeMap<String, Files>) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        listener.set_nonblocking(true).unwrap();
        let endpoint = format!("http://{}", listener.local_addr().unwrap());
        let requests = Arc::new(Mutex::new(Vec::new()));
        let log = Arc::clone(&requests);
        let stop = Arc::new(AtomicBool::new(false));
        let stopped = Arc::clone(&stop);
        let server = thread::spawn(move || {
            while !stopped.load(Ordering::Relaxed) {
                match listener.accept() {
                    Ok((stream, _)) => serve(stream, &snapshots, &log),
                    Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                        thread::sleep(Duration::from_millis(5));
                    }
                    Err(error) => panic!("loopback accept: {error}"),
                }
            }
        });
        Self {
            endpoint,
            requests,
            stop,
            server: Some(server),
        }
    }

    fn command(&self, home: &Path, args: &[&str]) -> Command {
        let mut command = isolated(home);
        command.args(args).env("HF_ENDPOINT", &self.endpoint);
        command
    }

    fn assert_metadata_only(&self, revision: Option<&str>) {
        let requests = self.requests.lock().unwrap();
        assert!(!requests.is_empty(), "fixture must exercise real metadata I/O");
        for request in requests.iter() {
            let first = request.lines().next().unwrap();
            let read_only_stat = first.starts_with("POST /api/datasets/test/snapshot/paths-info/");
            assert!(
                first.starts_with("GET ") || first.starts_with("HEAD ") || read_only_stat,
                "write: {first}"
            );
            assert!(
                !request.to_ascii_lowercase().replace("%2f", "/").contains("/data/"),
                "vector download: {first}"
            );
            assert!(
                !request.to_ascii_lowercase().contains("authorization:"),
                "unexpected token"
            );
            if let Some(revision) = revision {
                assert!(first.contains(revision), "request escaped fixed SHA: {first}");
                assert!(!first.contains("/refs"), "query worker resolved live HEAD");
            }
        }
    }
}

impl Drop for Hub {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        self.server.take().unwrap().join().unwrap();
    }
}

fn serve(mut stream: TcpStream, snapshots: &BTreeMap<String, Files>, log: &Mutex<Vec<String>>) {
    stream.set_read_timeout(Some(Duration::from_secs(2))).unwrap();
    stream.set_write_timeout(Some(Duration::from_secs(2))).unwrap();
    let mut request = Vec::new();
    let mut buffer = [0; 4096];
    while !request.windows(4).any(|bytes| bytes == b"\r\n\r\n") {
        let count = stream.read(&mut buffer).unwrap();
        if count == 0 {
            return;
        }
        request.extend_from_slice(&buffer[..count]);
        assert!(request.len() < 65536);
    }
    let header_end = request.windows(4).position(|bytes| bytes == b"\r\n\r\n").unwrap() + 4;
    let headers = String::from_utf8_lossy(&request[..header_end]);
    let content_length = headers
        .lines()
        .find_map(|line| {
            let (key, value) = line.split_once(':')?;
            key.eq_ignore_ascii_case("content-length")
                .then(|| value.trim().parse::<usize>().unwrap())
        })
        .unwrap_or(0);
    assert!(content_length < 65536);
    while request.len() < header_end + content_length {
        let count = stream.read(&mut buffer).unwrap();
        if count == 0 {
            return;
        }
        request.extend_from_slice(&buffer[..count]);
    }
    let request = String::from_utf8(request).unwrap();
    let mut parts = request.lines().next().unwrap().split_whitespace();
    let method = parts.next().unwrap();
    let target = parts.next().unwrap();
    let route = target.split('?').next().unwrap();
    log.lock().unwrap().push(request.clone());
    let mut sha = HEAD_SHA;
    let mut status = 200;
    let body = if route == "/api/datasets/test/snapshot/refs" {
        serde_json::to_vec(&json!({
            "branches": [{"name": "main", "ref": "refs/heads/main", "targetCommit": HEAD_SHA}],
            "tags": []
        }))
        .unwrap()
    } else if let Some(rest) = route.strip_prefix("/datasets/test/snapshot/resolve/") {
        let (revision, filename) = rest.split_once('/').unwrap();
        sha = revision;
        match snapshots.get(revision).and_then(|files| files.get(filename)) {
            Some(body) => body.clone(),
            None => {
                status = 404;
                b"{\"error\":\"fixture missing\"}".to_vec()
            }
        }
    } else if let Some(revision) = route.strip_prefix("/api/datasets/test/snapshot/paths-info/") {
        sha = revision;
        let form = request.split_once("\r\n\r\n").unwrap().1;
        let encoded = form
            .split('&')
            .find_map(|field| field.strip_prefix("paths="))
            .unwrap_or("");
        let mut decoded = Vec::new();
        let mut bytes = encoded.bytes();
        while let Some(byte) = bytes.next() {
            decoded.push(if byte == b'%' {
                let hex = [bytes.next().unwrap(), bytes.next().unwrap()];
                u8::from_str_radix(std::str::from_utf8(&hex).unwrap(), 16).unwrap()
            } else {
                byte
            });
        }
        let filename = String::from_utf8(decoded).unwrap();
        let entry = snapshots
            .get(revision)
            .and_then(|files| files.get(&filename))
            .map(|body| {
                json!({
                    "type": "file", "path": filename, "size": body.len(),
                    "oid": hex::encode(Sha256::digest(body)),
                    "lastCommit": {"id": revision, "date": "2026-11-13T00:00:00Z"}
                })
            });
        serde_json::to_vec(&entry.into_iter().collect::<Vec<_>>()).unwrap()
    } else if let Some(rest) = route.strip_prefix("/api/datasets/test/snapshot/tree/") {
        let (revision, prefix) = rest.split_once('/').unwrap_or((rest, ""));
        sha = revision;
        let entries: Vec<Value> = snapshots
            .get(revision)
            .into_iter()
            .flat_map(|files| files.iter())
            .filter(|(filename, _)| filename.starts_with(prefix))
            .map(|(filename, body)| {
                json!({
                    "type": "file", "path": filename, "size": body.len(),
                    "oid": hex::encode(Sha256::digest(body)),
                    "lastCommit": {"id": revision, "date": "2026-11-13T00:00:00Z"}
                })
            })
            .collect();
        serde_json::to_vec(&entries).unwrap()
    } else {
        status = 404;
        b"{\"error\":\"unexpected fixture route\"}".to_vec()
    };
    let error_header = if status == 404 {
        "X-Error-Code: EntryNotFound\r\n"
    } else {
        ""
    };
    let etag = hex::encode(Sha256::digest(&body));
    let _ = write!(stream,
        "HTTP/1.1 {status} Fixture\r\nContent-Type: application/json\r\nContent-Length: {}\r\nETag: \"{etag}\"\r\nX-Repo-Commit: {sha}\r\n{error_header}Connection: close\r\n\r\n",
        body.len()
    );
    if method != "HEAD" {
        let _ = stream.write_all(&body);
    }
}

fn isolated(home: &Path) -> Command {
    let mut command = Command::new(env!("CARGO_BIN_EXE_funes"));
    command
        .env_clear()
        .env("HOME", home)
        .env("FUNES_HOME", home.join("funes"))
        .env("HF_HOME", home.join("hf"))
        .env("HF_HUB_CACHE", home.join("hf/hub"))
        .env("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
        .env("HF_ENDPOINT", "http://127.0.0.1:1")
        .env("NO_PROXY", "*")
        .env("DO_NOT_TRACK", "1")
        .env("FUNES_EMBEDDING_PROVIDER", "local");
    command
}

fn run(mut command: Command) -> Output {
    let mut child = command.stdout(Stdio::piped()).stderr(Stdio::piped()).spawn().unwrap();
    let deadline = Instant::now() + Duration::from_secs(25);
    let mut expired = false;
    while child.try_wait().unwrap().is_none() {
        if Instant::now() >= deadline {
            expired = true;
            child.kill().unwrap();
            break;
        }
        thread::sleep(Duration::from_millis(10));
    }
    let output = child.wait_with_output().unwrap();
    assert!(
        !expired,
        "snapshot subprocess deadline: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    output
}

fn failure(output: Output, expected: &str) {
    assert!(!output.status.success());
    assert!(output.stdout.is_empty(), "failure emitted success output");
    let error = String::from_utf8_lossy(&output.stderr);
    assert!(error.contains(expected), "expected {expected:?}, got {error}");
}

#[test]
fn query_rejects_native_writes_before_any_io() {
    let home = tempfile::tempdir().unwrap();
    for args in [
        vec!["ingest-docs", "/missing.jsonl"],
        vec!["index", "/missing.jsonl"],
        vec!["push", "test/snapshot", "--yes"],
        vec!["compact", "test/snapshot"],
        vec!["optimize-index", "test/snapshot"],
        vec!["scrub"],
        vec!["reindex", "--all"],
        vec!["bind", "test/snapshot"],
        vec!["add", "codex"],
        vec!["remove", "codex"],
        vec!["update"],
        vec!["sync", "run"],
        vec!["sync", "backfill"],
        vec!["sync", "install"],
        vec!["doctor"],
        vec!["sources"],
    ] {
        let mut command = isolated(home.path());
        command.args(args).env("FUNES_SERVICE_ROLE", "query");
        failure(run(command), "refuses native write and maintenance commands");
    }
    assert_eq!(
        std::fs::read_dir(home.path()).unwrap().count(),
        0,
        "rejected command wrote files"
    );
}

#[test]
fn query_requires_both_valid_pin_variables() {
    let home = tempfile::tempdir().unwrap();
    for (sha, version, expected) in [
        (None, None, "FUNES_QUERY_HF_REVISION"),
        (Some(OLD_SHA), None, "FUNES_QUERY_LANCE_VERSION"),
        (Some("main"), Some("1"), "40-hex"),
        (Some(OLD_SHA), Some("0"), "positive u64"),
        (Some(OLD_SHA), Some("18446744073709551616"), "positive u64"),
    ] {
        let mut command = isolated(home.path());
        command
            .args(["mcp", "test/snapshot"])
            .env("FUNES_SERVICE_ROLE", "query");
        if let Some(sha) = sha {
            command.env("FUNES_QUERY_HF_REVISION", sha);
        }
        if let Some(version) = version {
            command.env("FUNES_QUERY_LANCE_VERSION", version);
        }
        failure(run(command), expected);
    }
}

#[test]
fn invalid_roles_and_snapshot_arguments_fail_closed() {
    let home = tempfile::tempdir().unwrap();
    let mut command = isolated(home.path());
    command
        .args(["query-snapshot", "test/snapshot"])
        .env("FUNES_SERVICE_ROLE", "reader");
    failure(run(command), "FUNES_SERVICE_ROLE");
    for args in [
        vec!["query-snapshot", "local"],
        vec!["query-snapshot", "test/snapshot/prefix"],
        vec!["query-snapshot", "test/snapshot", "--revision", "main"],
    ] {
        let mut command = isolated(home.path());
        command.args(args).env("FUNES_SERVICE_ROLE", "query");
        let output = run(command);
        assert!(!output.status.success());
        assert!(output.stdout.is_empty());
        assert!(!String::from_utf8_lossy(&output.stderr).contains("requires FUNES_QUERY"));
    }
}

#[test]
fn pinned_open_child() {
    let Ok(mode) = std::env::var("FUNES_SNAPSHOT_TEST_CHILD") else {
        return;
    };
    let runtime = tokio::runtime::Runtime::new().unwrap();
    runtime.block_on(async {
        let memory = if mode == "local" {
            Memory::local()
        } else {
            Memory::parse("test/snapshot")
        };
        let result = memory.open().await;
        if mode == "success" {
            assert_eq!(result.unwrap().version().version, 1);
            assert!(matches!(
                memory.state().await.unwrap(),
                funes::memory::MemoryState::Ready(_)
            ));
        } else {
            let error = match result {
                Ok(_) => panic!("must reject snapshot"),
                Err(error) => format!("{error:#}"),
            };
            let expected = match mode.as_str() {
                "local" => "requires a remote memory",
                "mismatch" => "Lance version mismatch",
                _ => "opening the wrapped dataset",
            };
            assert!(error.contains(expected), "{error}");
        }
    });
}

#[tokio::test]
async fn metadata_and_query_open_share_one_compatible_immutable_snapshot() {
    let source = tempfile::tempdir().unwrap();
    let uri = dataset::table_uri(source.path());
    let mut dataset = Dataset::write(
        batch(dataset::DIM),
        &uri,
        Some(WriteParams {
            enable_v2_manifest_paths: false,
            ..Default::default()
        }),
    )
    .await
    .unwrap();
    let old = files(source.path(), 1);
    dataset.append(batch(dataset::DIM), None).await.unwrap();
    let head = files(source.path(), 2);
    let bad_source = tempfile::tempdir().unwrap();
    Dataset::write(
        batch(3),
        &dataset::table_uri(bad_source.path()),
        Some(WriteParams {
            enable_v2_manifest_paths: false,
            ..Default::default()
        }),
    )
    .await
    .unwrap();
    let hub = Hub::new(BTreeMap::from([
        (OLD_SHA.to_string(), old),
        (HEAD_SHA.to_string(), head),
        (BAD_SHA.to_string(), files(bad_source.path(), 1)),
    ]));

    for (revision, expected_sha, expected_version) in [(None, HEAD_SHA, 2), (Some(OLD_SHA), OLD_SHA, 1)] {
        let home = tempfile::tempdir().unwrap();
        let mut command = hub.command(home.path(), &["query-snapshot", "test/snapshot"]);
        // Metadata discovery is exempt even when inherited worker pin values are invalid.
        command
            .env("FUNES_SERVICE_ROLE", "query")
            .env("FUNES_QUERY_HF_REVISION", "invalid")
            .env("FUNES_QUERY_LANCE_VERSION", "0");
        if let Some(revision) = revision {
            command.args(["--revision", revision]);
        }
        let output = run(command);
        assert!(output.status.success(), "{}", String::from_utf8_lossy(&output.stderr));
        let text = String::from_utf8(output.stdout).unwrap();
        assert_eq!(text.lines().count(), 1);
        assert_eq!(
            serde_json::from_str::<Value>(&text).unwrap(),
            json!({
                "schema_version": 1, "memory": "test/snapshot", "hf_sha": expected_sha,
                "lance_version": expected_version
            })
        );
        assert!(!home.path().join(".funes/install-id").exists());
    }
    hub.assert_metadata_only(None);

    let home = tempfile::tempdir().unwrap();
    failure(
        run(hub.command(home.path(), &["query-snapshot", "test/snapshot", "--revision", BAD_SHA])),
        "memory vector dimension 3 != stored 384",
    );
    hub.assert_metadata_only(None);

    for (mode, revision, version) in [
        ("success", OLD_SHA, "1"),
        ("mismatch", OLD_SHA, "2"),
        ("local", OLD_SHA, "1"),
        ("unavailable", "4444444444444444444444444444444444444444", "1"),
    ] {
        hub.requests.lock().unwrap().clear();
        let home = tempfile::tempdir().unwrap();
        let mut command = isolated(home.path());
        command = {
            let mut child = Command::new(std::env::current_exe().unwrap());
            child.env_clear().envs(
                command
                    .get_envs()
                    .filter_map(|(key, value)| value.map(|value| (key, value))),
            );
            child
        };
        command
            .args(["--exact", "pinned_open_child", "--nocapture"])
            .env("HF_ENDPOINT", &hub.endpoint)
            .env("FUNES_SERVICE_ROLE", "query")
            .env("FUNES_QUERY_HF_REVISION", revision)
            .env("FUNES_QUERY_LANCE_VERSION", version)
            .env("FUNES_SNAPSHOT_TEST_CHILD", mode);
        let output = run(command);
        assert!(
            output.status.success(),
            "{mode}: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        if mode != "local" {
            hub.assert_metadata_only(Some(revision));
        } else {
            assert!(hub.requests.lock().unwrap().is_empty());
        }
    }
}
