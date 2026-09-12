#!/usr/bin/env python3
"""Run the provider-backed Chinese retrieval benchmark against native Funes.

The benchmark keeps credentials in process memory only.  It reads a Hugging Face
token from an environment variable, calls the OpenAI-compatible HF Router to
build English retrieval shadows, and compares two isolated native Funes indexes:

* before: canonical documents whose retrieval text is raw Chinese, queried with
  raw Chinese queries;
* after: canonical documents whose retrieval text is only the provider-generated
  English shadow, queried with provider-generated shadows.

Both arms use the same checked-in 60-memory/20-query fixture and the same native
Lance hybrid retrieval settings.  No remote memory or long-lived service is
modified.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = Path(__file__).resolve().parent
for path in (ROOT, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from benchmark_chinese_retrieval import Memory, Query, _memories, _queries
from service.server import PROMPT_VERSION, RETRIEVAL_PROMPT, normalize_text
from space.server import NativeMcpWorker


DEFAULT_ENDPOINT = "https://router.huggingface.co/v1/chat/completions"
DEFAULT_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
DEFAULT_OUTPUT = ROOT / "docs" / "chinese-retrieval-provider-results.json"
DEFAULT_BINARY = ROOT / "target" / "release" / "funes"
TOKEN_ENV_NAMES = (
    "HF_TOKEN",
    "FUNES_HF_TOKEN",
    "HUGGING_FACE_HUB_TOKEN",
    "HUGGINGFACE_TOKEN",
)
GET_LINE_RE = re.compile(r"^\s*→ get bench-([^\s]+)", re.MULTILINE)
ENV_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class ProviderError(RuntimeError):
    """A provider failure whose message never includes request headers or bodies."""


@dataclass(frozen=True)
class ProviderStats:
    calls: int
    seconds: float


class ProviderNormalizer:
    """Small standard-library OpenAI-compatible client with bounded retries."""

    def __init__(
        self,
        *,
        endpoint: str,
        model: str,
        token: str,
        timeout: float,
        retries: int,
    ) -> None:
        self.endpoint = endpoint
        self.model = model
        self._token = token
        self.timeout = timeout
        self.retries = retries
        self._calls = 0
        self._seconds = 0.0
        self._lock = threading.Lock()

    @property
    def stats(self) -> ProviderStats:
        with self._lock:
            return ProviderStats(calls=self._calls, seconds=round(self._seconds, 3))

    def normalize(self, raw: str) -> str:
        payload = json.dumps(
            {
                "model": self.model,
                "temperature": 0,
                "max_tokens": 256,
                "messages": [
                    {"role": "system", "content": RETRIEVAL_PROMPT},
                    {"role": "user", "content": raw},
                ],
            },
            ensure_ascii=False,
        ).encode("utf-8")
        last_error = "provider request failed"
        for attempt in range(self.retries):
            request = urllib.request.Request(
                self.endpoint,
                data=payload,
                headers={
                    "Authorization": "Bearer " + self._token,
                    "Content-Type": "application/json",
                },
                method="POST",
            )
            started = time.monotonic()
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    data = json.load(response)
                content = data["choices"][0]["message"]["content"]
                if not isinstance(content, str) or not normalize_text(content):
                    raise ProviderError("provider returned empty completion content")
                return normalize_text(content)
            except urllib.error.HTTPError as exc:
                last_error = f"provider HTTP {exc.code} ({exc.reason})"
                retryable = exc.code in (408, 425, 429) or exc.code >= 500
                if not retryable:
                    raise ProviderError(last_error) from None
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                last_error = f"provider transport failure ({type(exc).__name__})"
            except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError):
                last_error = "provider returned a malformed chat-completion response"
            finally:
                elapsed = time.monotonic() - started
                with self._lock:
                    self._calls += 1
                    self._seconds += elapsed
            if attempt + 1 < self.retries:
                time.sleep(0.5 * (2**attempt))
        raise ProviderError(f"{last_error} after {self.retries} attempts")


def _token_from_environment(requested: str | None) -> tuple[str, str]:
    if requested and not ENV_NAME_RE.fullmatch(requested):
        raise ProviderError("invalid token environment variable name")
    names = (requested,) if requested else TOKEN_ENV_NAMES
    for name in names:
        if name and os.environ.get(name):
            return os.environ[name], name
    expected = requested or ", ".join(TOKEN_ENV_NAMES)
    raise ProviderError(f"no provider token in environment ({expected})")


def _normalize_unique(
    provider: ProviderNormalizer,
    texts: list[str],
    concurrency: int,
) -> dict[str, str]:
    unique = list(dict.fromkeys(texts))
    completed = 0
    results: dict[str, str] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {pool.submit(provider.normalize, text): text for text in unique}
        for future in concurrent.futures.as_completed(futures):
            text = futures[future]
            results[text] = future.result()
            completed += 1
            if completed == len(unique) or completed % 10 == 0:
                print(f"provider normalization: {completed}/{len(unique)}", file=sys.stderr, flush=True)
    return results


def _atomic_write_json(path: Path, value: dict[str, object]) -> None:
    """Replace a result/checkpoint only after a complete JSON document is on disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _provider_outputs(inputs: list[str], translations: dict[str, str]) -> list[dict[str, str]]:
    return [
        {
            "input_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "input": text,
            "output": translations[text],
        }
        for text in inputs
    ]


def _load_provider_checkpoint(
    path: Path,
    *,
    endpoint: str,
    model: str,
    inputs: list[str],
) -> tuple[dict[str, str], dict[str, object]] | None:
    if not path.is_file():
        return None
    try:
        checkpoint = json.loads(path.read_text(encoding="utf-8"))
        provider = checkpoint["provider"]
        rows = checkpoint["provider_outputs"]
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None
    prompt_sha256 = hashlib.sha256(RETRIEVAL_PROMPT.encode("utf-8")).hexdigest()
    if (
        not isinstance(provider, dict)
        or provider.get("endpoint") != endpoint
        or provider.get("model") != model
        or provider.get("prompt_sha256") != prompt_sha256
        or not isinstance(rows, list)
    ):
        return None
    translations = {
        row["input"]: row["output"]
        for row in rows
        if isinstance(row, dict)
        and isinstance(row.get("input"), str)
        and isinstance(row.get("output"), str)
        and normalize_text(row["output"])
    }
    if any(text not in translations for text in inputs):
        return None
    return translations, dict(provider)


def _write_canonical_source(
    source: Path,
    memories: list[Memory],
    shadows: dict[str, str] | None,
) -> None:
    source.parent.mkdir(parents=True, exist_ok=True)
    with source.open("w", encoding="utf-8") as handle:
        for index, memory in enumerate(memories):
            retrieval_text = memory.raw if shadows is None else shadows[memory.raw]
            content_hash = hashlib.sha256(retrieval_text.encode("utf-8")).hexdigest()
            timestamp = f"2026-09-12T00:00:{index % 60:02d}Z"
            record = {
                "source_identity": f"bench-{memory.ident}",
                "source_version": f"sha256:{content_hash}",
                "retrieval_text": retrieval_text,
                "content_hash": content_hash,
                "updated_at": timestamp,
                "metadata": {
                    "benchmark": "chinese-retrieval-provider",
                    "fixture_id": memory.ident,
                    "source_agent": memory.source_agent,
                    "source_type": memory.source_type,
                },
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _ingest_docs(
    binary: Path,
    home: Path,
    memory: Path,
    source: Path,
    timeout: int,
    token: str,
) -> float:
    env = os.environ.copy()
    env["FUNES_HOME"] = str(home)
    started = time.monotonic()
    completed = subprocess.run(
        [str(binary), "ingest-docs", str(source), "--memory", str(memory)],
        capture_output=True,
        text=True,
        env=env,
        timeout=timeout,
        check=False,
    )
    elapsed = time.monotonic() - started
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout)[-1200:]
        if token:
            detail = detail.replace(token, "[REDACTED]")
        raise RuntimeError(f"native Funes ingest-docs failed with exit {completed.returncode}: {detail}")
    return round(elapsed, 3)


def _ranked_idents(output: str) -> list[str]:
    return GET_LINE_RE.findall(output)


def _metrics(ranks: list[int | None]) -> dict[str, float]:
    return {
        f"recall@{k}": round(sum(rank is not None and rank <= k for rank in ranks) / len(ranks), 4)
        for k in (1, 3, 5)
    }


def _run_arm(
    *,
    name: str,
    binary: Path,
    home: Path,
    memory: Path,
    source: Path,
    queries: list[Query],
    query_shadows: dict[str, str] | None,
    candidates: int,
    index_timeout: int,
    recall_timeout: float,
    token: str,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    print(f"{name}: ingesting {source.name}", file=sys.stderr, flush=True)
    index_seconds = _ingest_docs(binary, home, memory, source, index_timeout, token)
    print(
        f"{name}: ingest complete in {index_seconds:.3f}s; recalling {len(queries)} queries",
        file=sys.stderr,
        flush=True,
    )
    worker = NativeMcpWorker(str(binary), str(memory), home, timeout=recall_timeout)
    rows: list[dict[str, object]] = []
    ranks: list[int | None] = []
    query_seconds: list[float] = []
    try:
        for index, query in enumerate(queries, start=1):
            retrieval_query = query.text if query_shadows is None else query_shadows[query.text]
            started = time.monotonic()
            output = worker.recall(
                retrieval_query,
                k=5,
                candidates=candidates,
                half_life=0,
                neighbors=0,
            )
            elapsed = time.monotonic() - started
            if output.startswith("recall error:"):
                raise RuntimeError("native Funes recall returned an error")
            ranked = _ranked_idents(output)
            rank = ranked.index(query.expected) + 1 if query.expected in ranked else None
            ranks.append(rank)
            query_seconds.append(elapsed)
            rows.append(
                {
                    "query": query.text,
                    "retrieval_query": retrieval_query,
                    "expected": query.expected,
                    "rank": rank,
                    "seconds": round(elapsed, 3),
                }
            )
            if index % 5 == 0:
                print(f"{name}: recall {index}/{len(queries)}", file=sys.stderr, flush=True)
    finally:
        worker.close()
    summary: dict[str, object] = {
        **_metrics(ranks),
        "index_seconds": index_seconds,
        "mean_query_seconds": round(sum(query_seconds) / len(query_seconds), 3),
    }
    return summary, rows


def _self_check() -> dict[str, object]:
    memories = _memories()
    queries = _queries(memories)
    assert len(memories) >= 50
    assert len(queries) == 20
    assert len({memory.ident for memory in memories}) == len(memories)
    memory_ids = {memory.ident for memory in memories}
    assert all(query.expected in memory_ids for query in queries)
    assert _ranked_idents("→ get bench-first --from 0\n  → get bench-second --from 0\n") == [
        "first",
        "second",
    ]
    assert _metrics([1, 3, 5, None]) == {
        "recall@1": 0.25,
        "recall@3": 0.5,
        "recall@5": 0.75,
    }
    assert len(set([m.raw for m in memories] + [q.text for q in queries])) == 44
    return {
        "status": "ok",
        "memories": len(memories),
        "queries": len(queries),
        "unique_provider_inputs": len(set([m.raw for m in memories] + [q.text for q in queries])),
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    memories = _memories()
    queries = _queries(memories)
    if len(memories) < 50 or len(queries) < 20:
        raise RuntimeError("fixture must contain at least 50 memories and 20 queries")
    if not args.binary.is_file():
        raise RuntimeError(f"native Funes binary not found: {args.binary}")
    inputs = list(dict.fromkeys([memory.raw for memory in memories] + [query.text for query in queries]))
    cached = _load_provider_checkpoint(
        args.output,
        endpoint=args.endpoint,
        model=args.model,
        inputs=inputs,
    )
    token = ""
    checkpoint_reused = cached is not None
    if cached is not None:
        translations, provider_metadata = cached
        print(f"provider normalization: reused {len(inputs)} outputs from {args.output}", file=sys.stderr)
    else:
        token, token_env = _token_from_environment(args.token_env)
        provider = ProviderNormalizer(
            endpoint=args.endpoint,
            model=args.model,
            token=token,
            timeout=args.request_timeout,
            retries=args.retries,
        )
        translations = _normalize_unique(provider, inputs, args.concurrency)
        stats = provider.stats
        provider_metadata = {
            "protocol": "OpenAI-compatible chat completions",
            "endpoint": args.endpoint,
            "model": args.model,
            "credential_source": f"environment variable {token_env}",
            "credential_value_recorded": False,
            "prompt_version": PROMPT_VERSION,
            "prompt_sha256": hashlib.sha256(RETRIEVAL_PROMPT.encode("utf-8")).hexdigest(),
            "request_attempts": stats.calls,
            "request_seconds": stats.seconds,
        }
        checkpoint = {
            "status": "provider_normalization_completed",
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            "benchmark_kind": "provider_backed_native_funes_chinese_retrieval_v1",
            "provider": provider_metadata,
            "dataset": {
                "fixture": "scripts/benchmark_chinese_retrieval.py",
                "memories": len(memories),
                "queries": len(queries),
                "unique_provider_inputs": len(translations),
            },
            "provider_outputs": _provider_outputs(inputs, translations),
        }
        _atomic_write_json(args.output, checkpoint)
        print(f"provider normalization: checkpoint written to {args.output}", file=sys.stderr)
    if args.provider_only:
        return {
            "status": "provider_normalization_completed",
            "output": str(args.output),
            "provider_outputs": len(translations),
            "checkpoint_reused": checkpoint_reused,
        }

    with tempfile.TemporaryDirectory(prefix="funes-provider-bench.") as temp:
        root = Path(temp)
        before_root = root / "before"
        after_root = root / "after"
        before_source = before_root / "canonical.jsonl"
        after_source = after_root / "canonical.jsonl"
        _write_canonical_source(before_source, memories, None)
        _write_canonical_source(after_source, memories, translations)
        before, before_rows = _run_arm(
            name="before",
            binary=args.binary,
            home=before_root / "home",
            memory=before_root / "memory",
            source=before_source,
            queries=queries,
            query_shadows=None,
            candidates=args.candidates,
            index_timeout=args.index_timeout,
            recall_timeout=args.recall_timeout,
            token=token,
        )
        after, after_rows = _run_arm(
            name="after",
            binary=args.binary,
            home=after_root / "home",
            memory=after_root / "memory",
            source=after_source,
            queries=queries,
            query_shadows=translations,
            candidates=args.candidates,
            index_timeout=args.index_timeout,
            recall_timeout=args.recall_timeout,
            token=token,
        )

    combined_rows = []
    for before_row, after_row in zip(before_rows, after_rows):
        combined_rows.append(
            {
                "query": before_row["query"],
                "provider_query_shadow": after_row["retrieval_query"],
                "expected": before_row["expected"],
                "before_rank": before_row["rank"],
                "after_rank": after_row["rank"],
                "before_seconds": before_row["seconds"],
                "after_seconds": after_row["seconds"],
            }
        )
    result: dict[str, object] = {
        "status": "completed",
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "benchmark_kind": "provider_backed_native_funes_chinese_retrieval_v1",
        "provider": {**provider_metadata, "checkpoint_reused_for_retrieval": checkpoint_reused},
        "dataset": {
            "fixture": "scripts/benchmark_chinese_retrieval.py",
            "memories": len(memories),
            "queries": len(queries),
            "unique_provider_inputs": len(translations),
        },
        "retrieval": {
            "backend": "native Funes Lance vector + BM25 + cross-encoder rerank",
            "embedding_model": "BAAI/bge-small-en-v1.5",
            "k": 5,
            "candidates": args.candidates,
            "neighbors": 0,
            "half_life": 0,
        },
        "method": {
            "before": "canonical retrieval_text contains raw Chinese and queries use raw Chinese",
            "after": "canonical retrieval_text contains only the provider-generated English shadow and queries use provider-generated shadows",
            "ingestion": "both arms use funes ingest-docs with explicit local --memory paths; no transcript or raw_text sidecar is indexed",
            "isolation": "each arm has a separate temporary FUNES_HOME and local memory path; temporary files are removed after the run",
        },
        "before": before,
        "after": after,
        "delta": {
            key: round(float(after[key]) - float(before[key]), 4)
            for key in ("recall@1", "recall@3", "recall@5")
        },
        "rows": combined_rows,
        "document_shadows": [
            {"id": memory.ident, "provider_shadow": translations[memory.raw]} for memory in memories
        ],
        "provider_outputs": _provider_outputs(inputs, translations),
    }
    _atomic_write_json(args.output, result)
    return result


def parser() -> argparse.ArgumentParser:
    out = argparse.ArgumentParser(description=__doc__)
    out.add_argument("--self-check", action="store_true", help="validate the fixture and metric/parser invariants without network access")
    out.add_argument("--provider-only", action="store_true", help="stop after atomically checkpointing provider outputs")
    out.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    out.add_argument("--model", default=DEFAULT_MODEL)
    out.add_argument("--token-env", help="read the provider token from only this environment variable")
    out.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    out.add_argument("--binary", type=Path, default=DEFAULT_BINARY)
    out.add_argument("--concurrency", type=int, default=4)
    out.add_argument("--request-timeout", type=float, default=30.0)
    out.add_argument("--retries", type=int, default=3)
    out.add_argument("--index-timeout", type=int, default=900)
    out.add_argument("--recall-timeout", type=float, default=180.0)
    out.add_argument("--candidates", type=int, default=12)
    return out


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.concurrency < 1 or args.retries < 1 or args.candidates < 5:
        raise SystemExit("concurrency/retries must be positive and candidates must be at least 5")
    if args.self_check:
        print(json.dumps(_self_check(), ensure_ascii=False, indent=2))
        return 0
    try:
        result = run(args)
    except (ProviderError, RuntimeError, subprocess.TimeoutExpired) as exc:
        print(f"benchmark failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    if result["status"] == "provider_normalization_completed":
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    print(
        json.dumps(
            {
                "status": result["status"],
                "output": str(args.output),
                "before": result["before"],
                "after": result["after"],
                "delta": result["delta"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
