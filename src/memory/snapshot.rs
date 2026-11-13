//! Describe a compatible remote snapshot by reading only Hub and Lance metadata.

use anyhow::{ensure, Context, Result};
use serde::Serialize;

use super::{remote, Memory, READ_BRANCH};
use crate::hub::{client, hf_token, parse_hf};
use crate::service_role::validate_revision;

#[derive(Debug, Serialize)]
pub struct SnapshotDescription {
    pub schema_version: u32,
    pub memory: String,
    pub hf_sha: String,
    pub lance_version: u64,
}

pub async fn describe(spec: &str, revision: Option<&str>) -> Result<SnapshotDescription> {
    let memory = Memory::parse(spec);
    let Memory::Remote { uri } = &memory else {
        anyhow::bail!("query-snapshot requires a remote owner/name memory");
    };
    let (owner, name, prefix) = parse_hf(uri)?;
    ensure!(prefix.is_empty(), "query-snapshot requires a repository-root memory");
    let sha = match revision {
        Some(revision) => {
            validate_revision(revision)?;
            revision.to_ascii_lowercase()
        }
        None => {
            let repo = client(hf_token().as_deref(), false)?.dataset(&owner, &name);
            let sha = remote::head_oid(&repo, READ_BRANCH)
                .await
                .context("resolving query snapshot head")?;
            validate_revision(&sha)?;
            sha.to_ascii_lowercase()
        }
    };
    // Bypass the worker's environment requirements, but reuse the same compatible pinned open.
    let ds = memory.open_remote_revision(&sha).await?;
    let lance_version = ds.version().version;
    ensure!(lance_version > 0, "query snapshot Lance version must be positive");
    Ok(SnapshotDescription {
        schema_version: 1,
        memory: format!("{owner}/{name}"),
        hf_sha: sha,
        lance_version,
    })
}
