use std::fs;
use std::os::unix::fs::PermissionsExt;
use std::path::Path;
use std::process::{Command, Output};

fn fake_sync(root: &Path, exit_code: i32) -> std::path::PathBuf {
    let path = root.join("funes-sync");
    fs::write(
        &path,
        format!("#!/bin/sh\nprintf '%s\\n' \"$*\" >> \"$FUNES_TEST_CLI_LOG\"\nexit {exit_code}\n"),
    )
    .unwrap();
    let mut permissions = fs::metadata(&path).unwrap().permissions();
    permissions.set_mode(0o755);
    fs::set_permissions(&path, permissions).unwrap();
    path
}

fn run(root: &Path, args: &[&str], exit_code: i32) -> (Output, std::path::PathBuf) {
    let log = root.join("argv.log");
    let output = Command::new(env!("CARGO_BIN_EXE_funes"))
        .args(args)
        .env("FUNES_SYNC_BIN", fake_sync(root, exit_code))
        .env("FUNES_TEST_CLI_LOG", &log)
        .output()
        .unwrap();
    (output, log)
}

#[test]
fn reindex_forwards_exact_scope() {
    for (flag, expected) in [
        ("--retrieval-text", "reindex --retrieval-text\n"),
        ("--all", "reindex --all\n"),
    ] {
        let root = tempfile::tempdir().unwrap();
        let (output, log) = run(root.path(), &["reindex", flag], 0);
        assert!(output.status.success(), "{}", String::from_utf8_lossy(&output.stderr));
        assert_eq!(fs::read_to_string(log).unwrap(), expected);
    }
}

#[test]
fn reindex_requires_exactly_one_scope() {
    for args in [vec!["reindex"], vec!["reindex", "--retrieval-text", "--all"]] {
        let root = tempfile::tempdir().unwrap();
        let (output, log) = run(root.path(), &args, 0);
        assert!(!output.status.success());
        assert!(!log.exists(), "invalid arguments must not start funes-sync");
    }
}

#[test]
fn reindex_propagates_sync_failure() {
    let root = tempfile::tempdir().unwrap();
    let (output, log) = run(root.path(), &["reindex", "--all"], 23);
    assert!(!output.status.success());
    assert_eq!(fs::read_to_string(log).unwrap(), "reindex --all\n");
}
