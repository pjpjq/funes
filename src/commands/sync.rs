use anyhow::{anyhow, Context, Result};

/// Delegate the long-lived bridge to the separately testable Python package.  Keeping
/// discovery/queue state out of the Rust Lance process means a Space restart or a
/// client upgrade cannot strand a pending upload.
pub fn run(args: &[&str]) -> Result<()> {
    let mut child = std::process::Command::new(std::env::var("FUNES_SYNC_BIN").unwrap_or_else(|_| "funes-sync".into()));
    let status = child
        .args(args)
        .status()
        .context("running funes-sync (set FUNES_SYNC_BIN to its absolute path)")?;
    if status.success() {
        Ok(())
    } else {
        Err(anyhow!("funes-sync exited with {:?}", status.code()))
    }
}
