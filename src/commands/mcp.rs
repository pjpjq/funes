//! `funes mcp`: expose recall over the Model Context Protocol (stdio transport),
//! so any MCP client (Claude Code, Cursor, …) can call funes as a first-class tool.
//! stdout is the JSON-RPC channel — logs must go to stderr.

use super::recall;
use super::target::Target;
use crate::agents;
use anyhow::Result;
use rmcp::handler::server::router::tool::ToolRouter;
use rmcp::handler::server::wrapper::Parameters;
use rmcp::model::{CallToolResult, ContentBlock, Implementation, ServerCapabilities, ServerConfig};
use rmcp::transport::stdio;
use rmcp::{schemars, tool, tool_handler, tool_router, ServerHandler, ServiceExt};
use std::sync::Arc;
use tokio::sync::OnceCell;

#[derive(serde::Serialize)]
struct StructuredRecall {
    hits: Vec<StructuredHit>,
}

#[derive(serde::Serialize)]
struct StructuredHit {
    raw_text: String,
    session_id: String,
    seq: i64,
    timestamp: String,
    block_type: String,
    harness: String,
    score: f64,
    neighbors: Vec<StructuredNeighbor>,
}

#[derive(serde::Serialize)]
struct StructuredNeighbor {
    raw_text: String,
    seq: i64,
    role: String,
    block_type: String,
}

fn recall_tool_result(result: recall::RecallResult) -> CallToolResult {
    let hits = result
        .hits
        .into_iter()
        .map(|(hit, score)| StructuredHit {
            raw_text: hit.text,
            session_id: hit.session_id,
            seq: hit.seq,
            timestamp: hit.ts,
            block_type: hit.block_type,
            harness: hit.harness,
            score,
            neighbors: hit
                .neighbors
                .into_iter()
                .map(|neighbor| StructuredNeighbor {
                    raw_text: neighbor.text,
                    seq: neighbor.seq,
                    role: neighbor.role,
                    block_type: neighbor.block_type,
                })
                .collect(),
        })
        .collect();
    let mut response = CallToolResult::success(vec![ContentBlock::text(result.text)]);
    response.structured_content =
        Some(serde_json::to_value(StructuredRecall { hits }).expect("recall hits contain only JSON-compatible values"));
    response
}

fn text_tool_result(text: String) -> CallToolResult {
    CallToolResult::success(vec![ContentBlock::text(text)])
}

#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
pub struct RecallRequest {
    #[schemars(
        description = "Natural-language description of what to recall from past sessions. A query matches best in the language they are written in, which `status` reports."
    )]
    pub query: String,
    #[schemars(
        description = "Hits returned (default 8). Each is a verbatim passage of a few hundred tokens, so `k` is what a call costs to read."
    )]
    pub k: Option<usize>,
    #[schemars(description = "Adjacent chunks attached to each hit for context; 0 returns the hits alone.")]
    pub neighbors: Option<i64>,
    #[schemars(
        description = "Size of the fused pool the hits are cut from (default 30): how far the search looks. Larger than `k`, the pool is reranked by the cross-encoder before the cut, which costs seconds and grows with the pool, so `k` can stay small while the search looks deep. A pool of `k` rows or fewer is read whole, in the fused order."
    )]
    pub candidates: Option<usize>,
    #[schemars(description = "Restrict to a block type: text | thinking | tool_use | tool_result")]
    pub block_type: Option<String>,
    #[schemars(
        description = "Restrict to a role, as the turns carry it (an agent's turns say user | assistant | tool | system)"
    )]
    pub role: Option<String>,
    #[schemars(
        description = "Restrict to a harness facet, as the turns carry it (`claude` also matches the older `claude_code`)"
    )]
    pub harness: Option<String>,
    #[schemars(description = "Restrict to turns whose checkout resolved to this repo, as `owner/name`")]
    pub repo: Option<String>,
    #[schemars(description = "Restrict to turns on or after this date, `YYYY-MM-DD`")]
    pub since: Option<String>,
    #[schemars(description = "Restrict to turns on or before this date, `YYYY-MM-DD`")]
    pub until: Option<String>,
    #[schemars(
        description = "Memory to read for this call, as it stands — `<org>/<repo>`, an `hf://…` URI, a local path, or `local`. Omitted, the server's memory, with this host's turns not yet pushed to it when the host is bound to it."
    )]
    pub memory: Option<String>,
    #[schemars(description = "Restrict to exact canonical source identity")]
    pub source_identity: Option<String>,
    #[schemars(description = "Restrict to exact canonical source version")]
    pub source_version: Option<String>,
    #[schemars(description = "Restrict to exact canonical content hash")]
    pub content_hash: Option<String>,
    #[schemars(description = "Restrict to exact canonical updated at")]
    pub updated_at: Option<String>,
    #[schemars(description = "Restrict to exact canonical source agent")]
    pub source_agent: Option<String>,
    #[schemars(description = "Restrict to exact canonical source type")]
    pub source_type: Option<String>,
    #[schemars(description = "Restrict to exact canonical project")]
    pub project: Option<String>,
    #[schemars(description = "Restrict to exact canonical device id")]
    pub device_id: Option<String>,
    #[schemars(description = "Restrict to exact canonical content type")]
    pub content_type: Option<String>,
    #[schemars(description = "Restrict to present (false) or soft-missing (true) canonical sources")]
    pub source_missing: Option<bool>,
}

#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
pub struct GetRequest {
    #[schemars(description = "Session id from a recall hit's `→ get` line")]
    pub session_id: String,
    #[schemars(
        description = "First turn to read, as the session's own seq, which a hit's `→ get` line gives you. Defaults to the session's start."
    )]
    pub from: Option<i64>,
    #[schemars(
        description = "Last turn to read, as the session's own seq. Omit it to read a fixed span from `from`; every reply names the range it covered."
    )]
    pub to: Option<i64>,
    #[schemars(
        description = "Memory to read for this call, as it stands — the one the hit's `→ get` line carries, and none when it carries none, so the drill-down reads what the search read. Omitted, the server's memory, where a session this host has not finished pushing is read from the local memory."
    )]
    pub memory: Option<String>,
}

#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
pub struct SessionsRequest {
    #[schemars(description = "Keep only sessions whose checkout resolved to this repo, as `owner/name`")]
    pub repo: Option<String>,
    #[schemars(description = "Keep only sessions that started on or after this date, `YYYY-MM-DD`")]
    pub since: Option<String>,
    #[schemars(description = "Keep only sessions that started on or before this date, `YYYY-MM-DD`")]
    pub until: Option<String>,
    #[schemars(
        description = "Rows to list, keeping the most recent. More than the maximum cannot fit in one reply — walk with `offset` instead. Zero is an error, not every match."
    )]
    pub limit: Option<usize>,
    #[schemars(
        description = "Skip this many of the most recent matches before taking `limit`, to walk a listing back through time. The closing line names the offset that continues, and a given offset always names the same session."
    )]
    pub offset: Option<usize>,
    #[schemars(
        description = "Memory to list, as it stands — `<org>/<repo>`, an `hf://…` URI, a local path, or `local`. Omitted, the server's memory, with the sessions this host has not finished pushing listed from the local memory when the host is bound to it."
    )]
    pub memory: Option<String>,
}

#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
pub struct ScanRequest {
    #[schemars(
        description = "The literal string to find. Not regex — a pattern that silently matched nothing would read as a clean result."
    )]
    pub needle: String,
    #[schemars(description = "Session to scan — the id from a `sessions` row or a recall hit's `→ get` line")]
    pub session_id: String,
    #[schemars(
        description = "First turn to scan, as the session's own seq. Omit to start at the session's beginning; pass the seq a capped scan told you to continue from."
    )]
    pub from: Option<i64>,
    #[schemars(description = "Last turn to scan, as the session's own seq. Omit to scan to the end of the session.")]
    pub to: Option<i64>,
    #[schemars(description = "Match regardless of case")]
    pub ignore_case: Option<bool>,
    #[schemars(description = "Characters of surrounding text shown on each side of a match")]
    pub context: Option<usize>,
    #[schemars(
        description = "Memory to scan, as it stands — `<org>/<repo>`, an `hf://…` URI, a local path, or `local`. Omitted, the server's memory, where a session this host has not finished pushing is read from the local memory."
    )]
    pub memory: Option<String>,
}

#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
pub struct SketchRequest {
    #[schemars(description = "Session to digest — the id from a `sessions` row or a hit's `→ get` line")]
    pub session_id: String,
    #[schemars(
        description = "How many distinct places to show — breadth (default 8, maximum 40). Also held to what `max_chars` can render at 240 characters apiece, and any narrowing is reported in the reply."
    )]
    pub units: Option<usize>,
    #[schemars(
        description = "Total characters to render — cost (default 8000, maximum 40000). More than that is a reply no caller receives."
    )]
    pub max_chars: Option<usize>,
    #[schemars(
        description = "First turn to digest, as the session's own seq. Digest a long session in windows to get coverage proportional to its length rather than a fixed sample of it."
    )]
    pub from: Option<i64>,
    #[schemars(description = "Last turn to digest, as the session's own seq.")]
    pub to: Option<i64>,
    #[schemars(
        description = "Memory to read for this call, as it stands — `<org>/<repo>`, an `hf://…` URI, a local path, or `local`. Omitted, the server's memory, where a session this host has not finished pushing is read from the local memory."
    )]
    pub memory: Option<String>,
}

#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
pub struct StatusRequest {
    #[schemars(
        description = "Memory to inspect — `<org>/<repo>`, an `hf://…` URI, a local path, or `local`. Omitted, the server's memory."
    )]
    pub memory: Option<String>,
}

/// A tool's text, led by the stale-install note when there is one. A tool result is the one channel
/// every client hands to the model, so this is where an install that stopped capturing gets said;
/// the CLI prints the same line on stderr and keeps stdout byte-identical to this text.
fn noted(text: String) -> String {
    match agents::stale_install_notice() {
        Some(note) => format!("{note}\n{text}"),
        None => text,
    }
}

fn pin_memory_enabled() -> bool {
    let value = std::env::var("FUNES_MCP_PIN_MEMORY").ok();
    pin_memory_value(value.as_deref())
}

fn pin_memory_value(value: Option<&str>) -> bool {
    matches!(
        value.map(str::trim).map(str::to_ascii_lowercase).as_deref(),
        Some("1" | "true" | "yes" | "on")
    )
}

#[derive(Clone)]
pub(crate) struct Funes {
    /// The memory the server was started on (`funes mcp <memory>`), read by every call naming
    /// none; `None` leaves those calls to the host's binding, as on the CLI.
    memory: Option<String>,
    /// Set only for an explicit server memory plus `FUNES_MCP_PIN_MEMORY=true`. Tool-call memory
    /// overrides never touch this cache and retain the normal resolve-current-head behavior.
    pinned_recall: Option<Arc<OnceCell<recall::PinnedRead>>>,
    #[allow(dead_code)]
    tool_router: ToolRouter<Funes>,
}

#[tool_router]
impl Funes {
    fn new(memory: Option<String>) -> Self {
        Self::new_with_pin(memory, pin_memory_enabled())
    }

    fn new_with_pin(memory: Option<String>, pin_memory: bool) -> Self {
        let memory = memory
            .map(|value| value.trim().to_string())
            .filter(|value| !value.is_empty());
        Self {
            pinned_recall: (pin_memory && memory.is_some()).then(|| Arc::new(OnceCell::new())),
            memory,
            tool_router: Self::tool_router(),
        }
    }

    /// What a call reads: the memory it names, as it stands; naming none, the server's.
    fn target(&self, spec: Option<String>) -> Result<Target> {
        match spec.filter(|s| !s.trim().is_empty()) {
            Some(spec) => Target::resolve(Some(&spec)),
            None => Target::served(self.memory.as_deref()),
        }
    }

    /// Only an omitted (or blank) call-level memory is the server default. Even an override that
    /// names the same URI deliberately takes the fresh path: spelling an override is an explicit
    /// request to retain ordinary per-call resolution semantics.
    fn pins_recall(&self, spec: &Option<String>) -> bool {
        self.pinned_recall.is_some() && spec.as_deref().is_none_or(|value| value.trim().is_empty())
    }

    #[allow(clippy::too_many_arguments)]
    async fn recall_with_memory(
        &self,
        memory: Option<String>,
        query: String,
        k: usize,
        candidates: usize,
        neighbors: i64,
        filter: recall::RecallFilter,
    ) -> Result<recall::RecallResult> {
        if self.pins_recall(&memory) {
            let cache = self
                .pinned_recall
                .as_ref()
                .expect("pins_recall requires a configured cache");
            let server_memory =
                crate::memory::Memory::parse(self.memory.as_deref().expect("pinned memory is explicit"));
            let pinned = cache.get_or_try_init(|| recall::pin_read(server_memory)).await?;
            recall::recall_filtered_pinned(pinned, query, k, candidates, neighbors, filter).await
        } else {
            recall::recall_filtered_result(self.target(memory)?, query, k, candidates, neighbors, filter).await
        }
    }

    #[tool(
        description = "Semantic search over the user's past AI agent sessions: describe what you are after, get back the verbatim passages — what was decided, tried, measured or investigated. Call it when they refer to earlier work, or when you are about to re-derive something a session may already have settled. Call it too before claiming that something was never built, was dropped, or was never discussed: the code cannot show that, only the sessions can. Ranked top-k, so it gives you a foothold on a topic, not every session touching it. Ranking weighs relevance, not age: for the latest word on a topic, or one stretch of time, bound it with `since`/`until`. Each hit is a passage you will read: `k` is the token cost, `candidates` the reach, and a pool larger than `k` is reranked, at seconds of latency that grow with the pool, so the reach can grow without the reading."
    )]
    async fn recall(
        &self,
        Parameters(RecallRequest {
            query,
            k,
            neighbors,
            candidates,
            block_type,
            role,
            harness,
            repo,
            since,
            until,
            source_identity,
            source_version,
            content_hash,
            updated_at,
            source_agent,
            source_type,
            project,
            device_id,
            content_type,
            source_missing,
            memory,
        }): Parameters<RecallRequest>,
    ) -> CallToolResult {
        match self
            .recall_with_memory(
                memory,
                query,
                k.unwrap_or(recall::DEFAULT_K),
                candidates.unwrap_or(recall::DEFAULT_CANDIDATES),
                neighbors.unwrap_or(recall::DEFAULT_NEIGHBORS),
                recall::RecallFilter {
                    block_type,
                    role,
                    since,
                    until,
                    harness,
                    source_identity,
                    source_version,
                    content_hash,
                    updated_at,
                    source_agent,
                    source_type,
                    project,
                    repo,
                    device_id,
                    content_type,
                    source_missing,
                },
            )
            .await
        {
            Ok(mut result) => {
                result.text = noted(result.text);
                recall_tool_result(result)
            }
            Err(e) => text_tool_result(noted(format!("recall error: {e}"))),
        }
    }

    #[tool(
        description = "Read one session's turns as they were written — whole blocks, splits reassembled, nothing ranked or summarized. Typically the follow-up to a search verb: expand a result into the turns around it. Sessions run to thousands of turns: read the stretch you need, not the session."
    )]
    async fn get(
        &self,
        Parameters(GetRequest {
            session_id,
            from,
            to,
            memory,
        }): Parameters<GetRequest>,
    ) -> String {
        let range = recall::TurnRange { from, to };
        noted(
            match async { recall::get(self.target(memory)?, session_id, range).await }.await {
                Ok(s) if !s.is_empty() => s,
                Ok(_) => "no results".to_string(),
                Err(e) => format!("get error: {e}"),
            },
        )
    }

    #[tool(
        description = "Enumerate a memory's sessions, oldest first. Metadata only — when it started and when its last indexed turn was, harness, repo, turn count, id, and the prompt each opened with, which is the ask a session started from, not what it became. Filter by repo or date to get the session ids the other verbs take; a whole memory can run to thousands of rows."
    )]
    async fn sessions(
        &self,
        Parameters(SessionsRequest {
            repo,
            since,
            until,
            limit,
            offset,
            memory,
        }): Parameters<SessionsRequest>,
    ) -> String {
        let filter = recall::SessionFilter {
            repo,
            since,
            until,
            limit,
            offset: offset.unwrap_or(0),
        };
        noted(
            match async { recall::sessions(self.target(memory)?, filter).await }.await {
                Ok(s) if !s.is_empty() => s,
                Ok(_) => "no results".to_string(),
                Err(e) => format!("sessions error: {e}"),
            },
        )
    }

    #[tool(
        description = "Find every occurrence of a literal string in one session — exhaustive, unranked, in reading order. For when you already know the wording and want where it occurs, rather than what a topic is about. Scope is one session and one spelling: a zero result clears only that."
    )]
    async fn scan(
        &self,
        Parameters(ScanRequest {
            needle,
            session_id,
            from,
            to,
            ignore_case,
            context,
            memory,
        }): Parameters<ScanRequest>,
    ) -> String {
        noted(
            match async {
                recall::scan(
                    self.target(memory)?,
                    needle,
                    session_id,
                    from,
                    to,
                    ignore_case.unwrap_or(false),
                    context.unwrap_or(recall::DEFAULT_CONTEXT),
                )
                .await
            }
            .await
            {
                Ok(s) if !s.is_empty() => s,
                Ok(_) => "no results".to_string(),
                Err(e) => format!("scan error: {e}"),
            },
        )
    }

    #[tool(
        description = "What one session was about, worked on, and ended up at: the passages most distinctive within it, verbatim, always including its opening ask and its last word. Takes a session id; there is no query. Enough to tell the user what a session was, in one call; not a way to find a particular thing in it."
    )]
    async fn sketch(
        &self,
        Parameters(SketchRequest {
            session_id,
            units,
            max_chars,
            from,
            to,
            memory,
        }): Parameters<SketchRequest>,
    ) -> String {
        noted(
            match async { super::sketch::run(self.target(memory)?, session_id, from, to, units, max_chars).await }.await
            {
                Ok(s) if !s.is_empty() => s,
                Ok(_) => "no results".to_string(),
                Err(e) => format!("sketch error: {e}"),
            },
        )
    }

    #[tool(
        description = "Health and size of a memory: indexed chunks, the languages its sessions are written in, local chunks awaiting embedding, source sessions awaiting indexing, and for a remote this host pushes to what it has yet to push. Call it when a read comes back empty or thinner than expected — it says whether the memory is the problem rather than the call."
    )]
    async fn status(&self, Parameters(StatusRequest { memory }): Parameters<StatusRequest>) -> String {
        // No update check here: it needs the network, and the "update available" notice belongs
        // on the human-facing CLI `funes status`, not on this hot, otherwise-local tool path.
        noted(
            async { recall::status(self.target(memory)?).await }
                .await
                .unwrap_or_else(|e| format!("status error: {e}")),
        )
    }
}

#[tool_handler]
impl ServerHandler for Funes {
    fn get_info(&self) -> ServerConfig {
        let mut server_info = Implementation::default();
        server_info.name = "funes".to_string();
        server_info.version = env!("CARGO_PKG_VERSION").to_string();
        let mut instructions = "Persistent memory over the user's past AI coding sessions: their transcripts, \
                                indexed automatically as they work and read-only here — nothing has to be saved. \
                                When earlier work matters, this is the memory to consult."
            .to_string();
        if let Some(note) = agents::stale_install_notice() {
            instructions.push_str("\n\n");
            instructions.push_str(&note);
        }
        ServerConfig::new(ServerCapabilities::builder().enable_tools().build())
            .with_server_info(server_info)
            .with_instructions(instructions)
    }
}

pub async fn run(memory: Option<String>) -> Result<()> {
    let service = Funes::new(memory).serve(stdio()).await?;
    service.waiting().await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::types::Float32Type;
    use arrow_array::{ArrayRef, FixedSizeListArray, RecordBatch, RecordBatchIterator};
    use arrow_schema::{DataType, Field, Schema};
    use lance::dataset::Dataset;

    async fn legacy_memory() -> tempfile::TempDir {
        let memory = tempfile::tempdir().unwrap();
        let dimension = crate::memory::dataset::DIM;
        let schema = Arc::new(Schema::new(vec![Field::new(
            "vector",
            DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, true)), dimension),
            true,
        )]));
        let vectors: ArrayRef = Arc::new(FixedSizeListArray::from_iter_primitive::<Float32Type, _, _>(
            [Some(vec![Some(0.0); dimension as usize])],
            dimension,
        ));
        let batch = RecordBatch::try_new(Arc::clone(&schema), vec![vectors]).unwrap();
        let reader = RecordBatchIterator::new(vec![Ok(batch)], schema);
        let dataset = Dataset::write(
            reader,
            crate::memory::dataset::table_uri(memory.path().to_str().unwrap()).as_str(),
            None,
        )
        .await
        .unwrap();
        drop(dataset);
        memory
    }

    fn absent_facet() -> recall::RecallFilter {
        recall::RecallFilter {
            source_identity: Some("not-in-legacy-schema".to_string()),
            ..recall::RecallFilter::default()
        }
    }

    #[tokio::test]
    async fn concurrent_and_sequential_default_recalls_open_once_but_an_override_opens_fresh() {
        let memory = legacy_memory().await;
        let spec = memory.path().to_string_lossy().into_owned();
        let server = Funes::new_with_pin(Some(spec.clone()), true);
        let clone = server.clone();

        let (first, concurrent) = tokio::join!(
            server.recall_with_memory(None, "unused".to_string(), 1, 1, 0, absent_facet()),
            clone.recall_with_memory(None, "unused".to_string(), 1, 1, 0, absent_facet())
        );
        let first = first.unwrap();
        let concurrent = concurrent.unwrap();
        assert_eq!(first.text, recall::rendered("", &[]));
        assert_eq!(concurrent.text, recall::rendered("", &[]));
        let pinned_json = serde_json::to_value(recall_tool_result(first)).unwrap();
        assert_eq!(pinned_json["content"][0]["text"], recall::rendered("", &[]));
        assert_eq!(pinned_json["structuredContent"]["hits"], serde_json::json!([]));

        // Make a second Dataset::open impossible. The same server still recalls from its cached
        // handle, while an explicit call-level override must take the ordinary fresh-open path.
        std::fs::remove_dir_all(memory.path().join("chunks.lance")).unwrap();
        let second = server
            .recall_with_memory(None, "unused".to_string(), 1, 1, 0, absent_facet())
            .await
            .unwrap();
        assert_eq!(second.text, recall::rendered("", &[]));

        let override_error = server
            .recall_with_memory(Some(spec.clone()), "unused".to_string(), 1, 1, 0, absent_facet())
            .await
            .err()
            .expect("an explicit override must open afresh");
        assert_eq!(
            override_error.to_string(),
            crate::memory::Memory::parse(&spec).empty_error().to_string()
        );
    }

    #[test]
    fn pin_is_opt_in_and_only_for_the_server_default_memory() {
        let pinned = Funes::new_with_pin(Some("acme/memory".to_string()), true);
        assert!(pinned.pins_recall(&None));
        assert!(pinned.pins_recall(&Some("  ".to_string())));
        assert!(!pinned.pins_recall(&Some("acme/memory".to_string())));
        assert!(!pinned.pins_recall(&Some("local".to_string())));

        let opted_out = Funes::new_with_pin(Some("acme/memory".to_string()), false);
        assert!(!opted_out.pins_recall(&None));

        let no_explicit_server_memory = Funes::new_with_pin(None, true);
        assert!(!no_explicit_server_memory.pins_recall(&None));
    }

    #[test]
    fn pin_env_requires_an_explicit_truthy_value() {
        for value in [None, Some(""), Some("0"), Some("false"), Some("maybe")] {
            assert!(!pin_memory_value(value));
        }
        for value in [Some("1"), Some(" true "), Some("YES"), Some("on")] {
            assert!(pin_memory_value(value));
        }
    }

    #[test]
    fn structured_recall_adds_raw_hits_without_changing_text_content() {
        let hit = recall::Hit {
            id: "raw-hit".into(),
            memory: String::new(),
            fused: 0.875,
            text: "complete raw hit\nwith a second line".to_string(),
            session_id: "session-123".to_string(),
            workdir: "/tmp/project".to_string(),
            turn_uuid: "turn-1".to_string(),
            seq: 17,
            ts: "2026-09-14T01:02:03Z".to_string(),
            block_type: "text".to_string(),
            harness: "codex".to_string(),
            neighbors: vec![recall::Neighbor {
                seq: 16,
                role: "user".to_string(),
                block_type: "text".to_string(),
                text: "raw neighboring turn".to_string(),
            }],
        };
        let hits = vec![(hit, 0.875)];
        let text = recall::rendered("", &hits);

        let response = recall_tool_result(recall::RecallResult {
            text: text.clone(),
            hits,
        });
        let json = serde_json::to_value(response).unwrap();

        assert_eq!(json["content"][0]["text"], text);
        assert_eq!(
            json["structuredContent"]["hits"][0]["raw_text"],
            "complete raw hit\nwith a second line"
        );
        assert_eq!(json["structuredContent"]["hits"][0]["session_id"], "session-123");
        assert_eq!(json["structuredContent"]["hits"][0]["seq"], 17);
        assert_eq!(
            json["structuredContent"]["hits"][0]["timestamp"],
            "2026-09-14T01:02:03Z"
        );
        assert_eq!(json["structuredContent"]["hits"][0]["block_type"], "text");
        assert_eq!(json["structuredContent"]["hits"][0]["harness"], "codex");
        assert_eq!(json["structuredContent"]["hits"][0]["score"], 0.875);
        assert_eq!(
            json["structuredContent"]["hits"][0]["neighbors"][0],
            serde_json::json!({
                "raw_text": "raw neighboring turn",
                "seq": 16,
                "role": "user",
                "block_type": "text"
            })
        );
    }

    #[test]
    fn recall_errors_keep_text_and_omit_structured_content() {
        let json = serde_json::to_value(text_tool_result("recall error: broken".to_string())).unwrap();
        assert_eq!(json["content"][0]["text"], "recall error: broken");
        assert!(json.get("structuredContent").is_none());
    }
}
