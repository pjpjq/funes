//! Exercise the real Hub request in a child process, without changing other tests' environment.

use funes::memory::{remote_reachability, Reachability};
use std::io::{Read, Write};
use std::net::TcpListener;
use std::process::{Command, Stdio};
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};
use std::thread;
use std::time::{Duration, Instant};

fn check_probe(status: u16, delay: Duration, expected: &str) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    listener.set_nonblocking(true).unwrap();
    let address = listener.local_addr().unwrap();
    let stop = Arc::new(AtomicBool::new(false));
    let server_stop = Arc::clone(&stop);
    let server = thread::spawn(move || {
        let mut requests = Vec::new();
        let deadline = Instant::now() + Duration::from_secs(15);
        while !server_stop.load(Ordering::Relaxed) && Instant::now() < deadline {
            let (mut stream, _) = match listener.accept() {
                Ok(connection) => connection,
                Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                    thread::sleep(Duration::from_millis(10));
                    continue;
                }
                Err(error) => panic!("mock accept: {error}"),
            };
            stream.set_read_timeout(Some(Duration::from_secs(2))).unwrap();
            stream.set_write_timeout(Some(Duration::from_secs(2))).unwrap();
            let mut request = Vec::new();
            let mut buffer = [0; 1024];
            while !request.windows(4).any(|bytes| bytes == b"\r\n\r\n") {
                let count = stream.read(&mut buffer).unwrap();
                assert!(count > 0 && request.len() < 16384, "incomplete mock request");
                request.extend_from_slice(&buffer[..count]);
            }
            requests.push(String::from_utf8(request).unwrap());
            thread::sleep(delay);
            let body = if status == 200 {
                r#"{"id":"test/reachability","sha":"0123456789012345678901234567890123456789"}"#
            } else {
                r#"{"error":"fixture response"}"#
            };
            let error_header = if status == 404 {
                "X-Error-Code: RepoNotFound\r\n"
            } else {
                ""
            };
            // A timed-out probe closes the socket before the deliberately slow response.
            let _ = write!(
                stream,
                "HTTP/1.1 {status} Fixture\r\nContent-Type: application/json\r\n{error_header}Content-Length: {}\r\nConnection: close\r\n\r\n{body}",
                body.len()
            );
        }
        requests
    });
    let home = tempfile::tempdir().unwrap();
    let mut child = Command::new(std::env::current_exe().unwrap())
        .args(["--exact", "probe_child", "--nocapture"])
        .env_clear()
        .env("HOME", home.path())
        .env("HF_HOME", home.path())
        .env("HF_ENDPOINT", format!("http://{address}"))
        .env("HF_TOKEN", "hf_reachability_fixture")
        .env("NO_PROXY", "*")
        .env("FUNES_PROBE_EXPECTED", expected)
        .env("FUNES_PROBE_DELAYED", if delay.is_zero() { "0" } else { "1" })
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let deadline = Instant::now() + Duration::from_secs(12);
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
    stop.store(true, Ordering::Relaxed);
    let requests = server.join().unwrap();
    assert!(!expired, "probe child exceeded its bounded deadline");
    assert!(
        output.status.success(),
        "child failed: {}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(requests.len(), 1, "reachability must not retry");
    assert_eq!(
        requests[0].lines().next().unwrap(),
        "GET /api/datasets/test/reachability?expand=sha HTTP/1.1"
    );
}

#[test]
fn probe_child() {
    let Ok(expected) = std::env::var("FUNES_PROBE_EXPECTED") else {
        return;
    };
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let began = Instant::now();
    let actual = runtime.block_on(remote_reachability("hf://datasets/test/reachability"));
    let actual = match actual {
        Reachability::Ok => "ok",
        Reachability::Offline => "offline",
        Reachability::Missing => "missing",
    };
    assert_eq!(actual, expected);
    if std::env::var("FUNES_PROBE_DELAYED").unwrap() == "1" {
        assert!(began.elapsed() >= Duration::from_secs(5));
        assert!(
            began.elapsed() < Duration::from_secs(6),
            "five-second probe deadline was extended"
        );
    }
}

#[test]
fn minimal_info_is_reachable() {
    check_probe(200, Duration::ZERO, "ok");
}

#[test]
fn auth_errors_remain_reachable() {
    for status in [401, 403] {
        check_probe(status, Duration::ZERO, "ok");
    }
}

#[test]
fn missing_repo_stays_missing() {
    check_probe(404, Duration::ZERO, "missing");
}

#[test]
fn server_failure_is_offline_without_retry() {
    check_probe(503, Duration::ZERO, "offline");
}

#[test]
fn slow_probe_keeps_five_second_deadline() {
    check_probe(200, Duration::from_secs(6), "offline");
}
