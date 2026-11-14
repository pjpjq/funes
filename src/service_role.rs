//! Process-local service policy. Query children must inherit one immutable snapshot pair.

use anyhow::{bail, ensure, Context, Result};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ServiceRole {
    Combined,
    Query,
    Writer,
}

impl ServiceRole {
    pub fn from_env() -> Result<Self> {
        match std::env::var("FUNES_SERVICE_ROLE") {
            Ok(value) => Self::parse(&value),
            Err(std::env::VarError::NotPresent) => Ok(Self::Combined),
            Err(error) => Err(error).context("invalid FUNES_SERVICE_ROLE"),
        }
    }

    fn parse(value: &str) -> Result<Self> {
        match value {
            "combined" => Ok(Self::Combined),
            "query" => Ok(Self::Query),
            "writer" => Ok(Self::Writer),
            _ => bail!("FUNES_SERVICE_ROLE must be combined, query, or writer"),
        }
    }
}

#[derive(Clone, Debug)]
pub(crate) struct QuerySnapshot {
    pub revision: String,
    pub lance_version: u64,
}

impl QuerySnapshot {
    pub(crate) fn from_env() -> Result<Self> {
        let revision = std::env::var("FUNES_QUERY_HF_REVISION")
            .context("query role requires FUNES_QUERY_HF_REVISION (full 40-hex SHA)")?;
        let version = std::env::var("FUNES_QUERY_LANCE_VERSION")
            .context("query role requires FUNES_QUERY_LANCE_VERSION (positive u64)")?;
        Self::parse(&revision, &version)
    }

    fn parse(revision: &str, version: &str) -> Result<Self> {
        validate_revision(revision)?;
        ensure!(
            !version.is_empty() && version.bytes().all(|byte| byte.is_ascii_digit()),
            "FUNES_QUERY_LANCE_VERSION must be a positive u64"
        );
        let lance_version = version
            .parse::<u64>()
            .context("FUNES_QUERY_LANCE_VERSION must be a positive u64")?;
        ensure!(lance_version > 0, "FUNES_QUERY_LANCE_VERSION must be a positive u64");
        Ok(Self {
            revision: revision.to_ascii_lowercase(),
            lance_version,
        })
    }
}

pub(crate) fn validate_revision(revision: &str) -> Result<()> {
    ensure!(
        revision.len() == 40 && revision.bytes().all(|byte| byte.is_ascii_hexdigit()),
        "HF revision must be a full 40-hex commit SHA"
    );
    Ok(())
}

/// Validate the worker pin without opening a dataset. Metadata discovery does not call this.
pub fn require_query_snapshot() -> Result<()> {
    QuerySnapshot::from_env().map(|_| ())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roles_are_explicit_and_fail_closed() {
        assert_eq!(ServiceRole::parse("combined").unwrap(), ServiceRole::Combined);
        assert_eq!(ServiceRole::parse("query").unwrap(), ServiceRole::Query);
        assert_eq!(ServiceRole::parse("writer").unwrap(), ServiceRole::Writer);
        for invalid in ["", "Query", " query", "reader", "combined "] {
            assert!(ServiceRole::parse(invalid).is_err());
        }
    }

    #[test]
    fn query_snapshot_requires_immutable_sha_and_positive_u64() {
        let sha = "0123456789abcdef0123456789abcdef01234567";
        for invalid in ["", "main", "d0926cb", "0123456789abcdef0123456789abcdef0123456g"] {
            assert!(QuerySnapshot::parse(invalid, "1").is_err());
        }
        for invalid in ["", "0", "-1", "+1", " 1", "1.0", "18446744073709551616"] {
            assert!(QuerySnapshot::parse(sha, invalid).is_err());
        }
        let snapshot = QuerySnapshot::parse(sha, "18446744073709551615").unwrap();
        assert_eq!(snapshot.revision, sha);
        assert_eq!(snapshot.lance_version, u64::MAX);
    }
}
