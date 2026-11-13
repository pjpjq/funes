// Funes remote recall for pi. This file contains no credential or
// machine-specific value; settings come from env/config/Keychain at runtime.
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { execFileSync } from "node:child_process";
import { readFileSync, statSync } from "node:fs";
import { homedir } from "node:os";

export function configuredRemoteUrl(): string {
  if (process.env.FUNES_REMOTE_URL) return process.env.FUNES_REMOTE_URL;
  const path = process.env.FUNES_CONFIG || `${process.env.HOME || homedir()}/.config/funes/config.toml`;
  try {
    let section = "";
    let remoteUrl = "";
    let legacySyncUrl = "";
    let legacyTopLevelUrl = "";
    for (const sourceLine of readFileSync(path, "utf8").split(/\r?\n/)) {
      const line = sourceLine.trim();
      const sectionLine = line.match(/^\[([^\]]+)\]$/);
      if (sectionLine) {
        section = sectionLine[1];
        continue;
      }
      const value = line.match(/^(url|remote_url)\s*=\s*["']([^"']+)["']/);
      if (!value) continue;
      if (section === "remote" && value[1] === "url") remoteUrl = value[2];
      else if (section === "sync" && value[1] === "remote_url") legacySyncUrl = value[2];
      else if (!section && value[1] === "remote_url") legacyTopLevelUrl = value[2];
    }
    return remoteUrl || legacySyncUrl || legacyTopLevelUrl;
  } catch {}
  return "";
}

function configuredRemoteNumber(keys: string[]): number | undefined {
  const path = process.env.FUNES_CONFIG || `${process.env.HOME || homedir()}/.config/funes/config.toml`;
  try {
    let section = "";
    for (const sourceLine of readFileSync(path, "utf8").split(/\r?\n/)) {
      const line = sourceLine.trim();
      const sectionLine = line.match(/^\[([^\]]+)\]$/);
      if (sectionLine) {
        section = sectionLine[1];
        continue;
      }
      if (section !== "remote" && section !== "sync") continue;
      const match = line.match(/^([A-Za-z0-9_.-]+)\s*=\s*(?:["']([^"']+)["']|([0-9]+(?:\.[0-9]+)?))/);
      if (!match) continue;
      const [, key, quotedVal, numVal] = match;
      if (keys.includes(key)) {
        const val = Number(quotedVal ?? numVal);
        if (!Number.isNaN(val)) return val;
      }
    }
  } catch {}
  return undefined;
}

const base = configuredRemoteUrl().replace(/\/+$/, "");

function keychain(service: string): string {
  if (process.platform !== "darwin") return "";
  try {
    return execFileSync("/usr/bin/security", [
      "find-generic-password", "-a", process.env.USER || "", "-s", service, "-w",
    ], { encoding: "utf8", stdio: ["ignore", "pipe", "ignore"], timeout: 5_000 }).trim();
  } catch { return ""; }
}

function zshrcEnv(name: "FUNES_API_TOKEN" | "FUNES_HF_TOKEN" | "HF_TOKEN"): string {
  if (process.platform !== "darwin") return "";
  const expressions = {
    FUNES_API_TOKEN: '"${FUNES_API_TOKEN:-}"',
    FUNES_HF_TOKEN: '"${FUNES_HF_TOKEN:-}"',
    HF_TOKEN: '"${HF_TOKEN:-}"',
  };
  try {
    return execFileSync(
      "/bin/zsh",
      ["-lc", `source "$HOME/.zshrc" >/dev/null 2>&1; printf %s ${expressions[name]}`],
      { encoding: "utf8", stdio: ["ignore", "pipe", "ignore"], timeout: 5_000 },
    ).trim();
  } catch { return ""; }
}

function secretFileEnv(name: "FUNES_API_TOKEN" | "FUNES_HF_TOKEN" | "HF_TOKEN"): string {
  if (process.platform !== "linux") return "";
  const path = process.env.FUNES_ENV_FILE || `${process.env.HOME || homedir()}/.config/funes/env`;
  try {
    // This file is a process-environment fallback, not general shell syntax.
    // Reject group/world-readable files and only accept literal assignments so
    // starting pi can never evaluate commands from it.
    const stat = statSync(path);
    if (!stat.isFile()) return "";
    if ((stat.mode & 0o077) !== 0) return "";
    if (typeof process.getuid === "function" && stat.uid !== process.getuid()) return "";
    for (const sourceLine of readFileSync(path, "utf8").split(/\r?\n/)) {
      const line = sourceLine.trim().replace(/^export\s+/, "");
      const value = line.match(/^([A-Za-z_][A-Za-z0-9_]*)=(.*)$/);
      if (!value || value[1] !== name) continue;
      const raw = value[2].trim();
      if (
        raw.length >= 2
        && ((raw.startsWith("'") && raw.endsWith("'"))
          || (raw.startsWith('"') && raw.endsWith('"')))
      ) return raw.slice(1, -1);
      return raw;
    }
  } catch {}
  return "";
}

const token = process.env.FUNES_API_TOKEN || secretFileEnv("FUNES_API_TOKEN") || keychain("funes-api-token") || zshrcEnv("FUNES_API_TOKEN");
const hubToken = process.env.FUNES_HF_TOKEN || process.env.HF_TOKEN || secretFileEnv("FUNES_HF_TOKEN") || secretFileEnv("HF_TOKEN") || keychain("funes-hf-token") || zshrcEnv("FUNES_HF_TOKEN") || zshrcEnv("HF_TOKEN");

function envNumber(name: string): number | undefined {
  const raw = process.env[name];
  if (raw === undefined || raw.trim() === "") return undefined;
  const value = Number(raw);
  return Number.isFinite(value) ? value : undefined;
}

function firstEnvNumber(names: string[]): number | undefined {
  for (const name of names) {
    const value = envNumber(name);
    if (value !== undefined) return value;
  }
  return undefined;
}

function bounded(value: number | undefined, fallback: number, minimum: number, maximum: number): number {
  return Math.min(maximum, Math.max(minimum, value ?? fallback));
}

// Keep one deadline across readiness checks, network attempts, and backoff. The
// *_MS spelling was used by an earlier release; prefer it when present while
// also accepting the seconds-based names shared with the other integrations.
const remoteTimeoutMs = (() => {
  const milliseconds = envNumber("FUNES_REMOTE_TIMEOUT_MS");
  if (milliseconds !== undefined) return bounded(milliseconds, 180_000, 5_000, 300_000);
  return bounded((envNumber("FUNES_REMOTE_TIMEOUT") ?? 180) * 1_000, 180_000, 5_000, 300_000);
})();
const remoteAttempts = Math.floor(bounded(envNumber("FUNES_REMOTE_ATTEMPTS"), 5, 1, 5));
const remoteAttemptTimeoutMs = bounded(
  (envNumber("FUNES_REMOTE_ATTEMPT_TIMEOUT_MS") ?? (envNumber("FUNES_REMOTE_ATTEMPT_TIMEOUT") ?? 50) * 1_000),
  50_000,
  1_000,
  55_000,
);
const remoteReadyTimeoutMs = bounded((envNumber("FUNES_REMOTE_READY_TIMEOUT") ?? 8) * 1_000, 8_000, 1_000, 15_000);
const remoteReadyPolls = Math.floor(bounded(envNumber("FUNES_REMOTE_READY_POLLS"), 8, 1, 30));

type FetchResponse = Awaited<ReturnType<typeof fetch>>;
type FetchResult = {
  response: FetchResponse;
  value: unknown;
  validJson: boolean;
};

type CallBudget = {
  timeoutMs: number;
  attempts: number;
  attemptTimeoutMs: number;
  readyTimeoutMs: number;
  readyPolls: number;
};

const manualBudget: CallBudget = {
  timeoutMs: remoteTimeoutMs,
  attempts: remoteAttempts,
  attemptTimeoutMs: remoteAttemptTimeoutMs,
  readyTimeoutMs: remoteReadyTimeoutMs,
  readyPolls: remoteReadyPolls,
};

const recallTimeoutMs = (() => {
  const ms = firstEnvNumber(["FUNES_REMOTE_RECALL_TIMEOUT_MS", "FUNES_REMOTE_SEARCH_TIMEOUT_MS"]);
  if (ms !== undefined) return bounded(ms, 18_000, 100, 60_000);
  const sec = firstEnvNumber(["FUNES_REMOTE_RECALL_TIMEOUT", "FUNES_REMOTE_SEARCH_TIMEOUT"]);
  if (sec !== undefined) return bounded(sec * 1_000, 18_000, 100, 60_000);
  const cfgMs = configuredRemoteNumber(["recall_timeout_ms", "search_timeout_ms"]);
  if (cfgMs !== undefined) return bounded(cfgMs, 18_000, 100, 60_000);
  const cfgSec = configuredRemoteNumber(["recall_timeout", "search_timeout"]);
  if (cfgSec !== undefined) return bounded(cfgSec * 1_000, 18_000, 100, 60_000);
  return 18_000;
})();

const recallAttempts = Math.floor(
  bounded(
    firstEnvNumber([
      "FUNES_REMOTE_RECALL_ATTEMPTS",
      "FUNES_REMOTE_SEARCH_ATTEMPTS",
    ]),
    2,
    1,
    5,
  ),
);

const recallAttemptTimeoutMs = (() => {
  const ms = firstEnvNumber([
    "FUNES_REMOTE_RECALL_ATTEMPT_TIMEOUT_MS",
    "FUNES_REMOTE_SEARCH_ATTEMPT_TIMEOUT_MS",
  ]);
  if (ms !== undefined) return bounded(ms, 8_000, 100, 30_000);
  const sec = firstEnvNumber([
    "FUNES_REMOTE_RECALL_ATTEMPT_TIMEOUT",
    "FUNES_REMOTE_SEARCH_ATTEMPT_TIMEOUT",
  ]);
  if (sec !== undefined) return bounded(sec * 1_000, 8_000, 100, 30_000);
  return 8_000;
})();

const manualRecallBudget: CallBudget = {
  timeoutMs: recallTimeoutMs,
  attempts: recallAttempts,
  attemptTimeoutMs: recallAttemptTimeoutMs,
  readyTimeoutMs: 2_000,
  readyPolls: 1,
};

const recall503DelayMs = (() => {
  const ms = firstEnvNumber([
    "FUNES_REMOTE_RECALL_503_DELAY_MS",
    "FUNES_REMOTE_SEARCH_503_DELAY_MS",
  ]);
  if (ms !== undefined) return bounded(ms, 3_000, 100, 15_000);
  const sec = firstEnvNumber([
    "FUNES_REMOTE_RECALL_503_DELAY",
    "FUNES_REMOTE_SEARCH_503_DELAY",
  ]);
  if (sec !== undefined) return bounded(sec * 1_000, 3_000, 100, 15_000);
  const cfgMs = configuredRemoteNumber([
    "recall_503_delay_ms",
    "search_503_delay_ms",
  ]);
  if (cfgMs !== undefined) return bounded(cfgMs, 3_000, 100, 15_000);
  const cfgSec = configuredRemoteNumber([
    "recall_503_delay",
    "search_503_delay",
  ]);
  if (cfgSec !== undefined) return bounded(cfgSec * 1_000, 3_000, 100, 15_000);
  return 3_000;
})();

const autoRecallTimeoutMs = (() => {
  const ms = firstEnvNumber([
    "FUNES_REMOTE_AUTO_RECALL_TIMEOUT_MS",
    "FUNES_REMOTE_AUTO_SEARCH_TIMEOUT_MS",
  ]);
  if (ms !== undefined) return ms <= 0 ? 0 : bounded(ms, 2_500, 100, 10_000);
  const sec = firstEnvNumber([
    "FUNES_REMOTE_AUTO_RECALL_TIMEOUT",
    "FUNES_REMOTE_AUTO_SEARCH_TIMEOUT",
  ]);
  if (sec !== undefined) return sec <= 0 ? 0 : bounded(sec * 1_000, 2_500, 100, 10_000);
  const cfgMs = configuredRemoteNumber([
    "auto_recall_timeout_ms",
    "auto_search_timeout_ms",
  ]);
  if (cfgMs !== undefined) return cfgMs <= 0 ? 0 : bounded(cfgMs, 2_500, 100, 10_000);
  const cfgSec = configuredRemoteNumber([
    "auto_recall_timeout",
    "auto_search_timeout",
  ]);
  if (cfgSec !== undefined) return cfgSec <= 0 ? 0 : bounded(cfgSec * 1_000, 2_500, 100, 10_000);
  return 2_500;
})();

const autoRecallAttemptTimeoutMs = (() => {
  if (autoRecallTimeoutMs <= 0) return 0;
  const ms = firstEnvNumber([
    "FUNES_REMOTE_AUTO_RECALL_ATTEMPT_TIMEOUT_MS",
    "FUNES_REMOTE_AUTO_SEARCH_ATTEMPT_TIMEOUT_MS",
  ]);
  if (ms !== undefined) return ms <= 0 ? 0 : bounded(ms, autoRecallTimeoutMs, 100, 10_000);
  const sec = firstEnvNumber([
    "FUNES_REMOTE_AUTO_RECALL_ATTEMPT_TIMEOUT",
    "FUNES_REMOTE_AUTO_SEARCH_ATTEMPT_TIMEOUT",
  ]);
  if (sec !== undefined) return sec <= 0 ? 0 : bounded(sec * 1_000, autoRecallTimeoutMs, 100, 10_000);
  const cfgMs = configuredRemoteNumber([
    "auto_recall_attempt_timeout_ms",
    "auto_search_attempt_timeout_ms",
  ]);
  if (cfgMs !== undefined) return cfgMs <= 0 ? 0 : bounded(cfgMs, autoRecallTimeoutMs, 100, 10_000);
  const cfgSec = configuredRemoteNumber([
    "auto_recall_attempt_timeout",
    "auto_search_attempt_timeout",
  ]);
  if (cfgSec !== undefined) return cfgSec <= 0 ? 0 : bounded(cfgSec * 1_000, autoRecallTimeoutMs, 100, 10_000);
  return autoRecallTimeoutMs;
})();

const automaticRecallBudget: CallBudget = {
  timeoutMs: autoRecallTimeoutMs,
  attempts: 1,
  attemptTimeoutMs: autoRecallAttemptTimeoutMs,
  readyTimeoutMs: 1_750,
  readyPolls: 1,
};

class RequestTimeoutError extends Error {
  constructor() {
    super("funes remote request timed out");
    this.name = "RequestTimeoutError";
  }
}

function now(): number {
  return Date.now();
}

async function fetchWithTimeout(url: string, init: RequestInit, timeoutMs: number): Promise<FetchResult> {
  const controller = new AbortController();
  let timer: ReturnType<typeof setTimeout> | undefined;
  const request = fetch(url, { ...init, redirect: "manual", signal: controller.signal }).then(
    async (response): Promise<FetchResult> => {
      if (statusOfResponse(response) === 204) {
        return { response, value: undefined, validJson: false };
      }
      try {
        return { response, value: await response.json(), validJson: true };
      } catch {
        return { response, value: undefined, validJson: false };
      }
    },
  );
  const timeout = new Promise<never>((_, reject) => {
    timer = setTimeout(() => {
      controller.abort();
      reject(new RequestTimeoutError());
    }, Math.max(1, timeoutMs));
  });
  try {
    return await Promise.race([request, timeout]);
  } finally {
    if (timer !== undefined) clearTimeout(timer);
  }
}

function statusOf(response: FetchResult): number {
  return statusOfResponse(response.response);
}

function statusOfResponse(response: FetchResponse): number {
  const status = Number(response.status);
  if (Number.isFinite(status) && status > 0) return status;
  return response.ok ? 200 : 500;
}

function isSuccess(response: FetchResult): boolean {
  const status = statusOf(response);
  return response.response.ok || (status >= 200 && status < 300);
}

function isRetryableStatus(status: number): boolean {
  // 408/425 are transport-window responses rather than authentication or
  // malformed-request failures; keep the other 4xx statuses terminal.
  return status === 408 || status === 425 || status === 429 || status >= 500;
}

function retryDelay(response: FetchResult | undefined, attempt: number, recall: boolean = false): number {
  const exponential = Math.min(30_000, 1_000 * 2 ** (attempt + 1));
  const status = response ? statusOf(response) : 0;
  const retryAfter = response?.response.headers?.get("retry-after");
  if (retryAfter) {
    const seconds = Number(retryAfter);
    if (Number.isFinite(seconds) && seconds >= 0) {
      const ms = seconds * 1_000;
      return recall ? Math.min(30_000, ms) : Math.max(exponential, Math.min(30_000, ms));
    }
    const date = Date.parse(retryAfter);
    if (Number.isFinite(date)) {
      const ms = Math.max(0, date - now());
      return recall ? Math.min(30_000, ms) : Math.max(exponential, Math.min(30_000, ms));
    }
  }
  if (recall && status === 503) {
    return recall503DelayMs;
  }
  return exponential;
}

async function sleepUntil(ms: number, deadline: number): Promise<boolean> {
  const remaining = deadline - now();
  if (remaining <= 0) return false;
  const delay = Math.min(Math.max(0, ms), Math.max(0, remaining));
  if (delay <= 0) return true;
  await new Promise<void>((resolve) => setTimeout(resolve, delay));
  return now() < deadline;
}

type ReadyState = "ready" | "warming" | "transient" | "unknown" | "permanent";

type RemoteCall =
  | { ok: true; value: unknown }
  | { ok: false; message: string; status?: number };

async function probeReady(headers: Record<string, string>, timeoutMs: number): Promise<ReadyState> {
  try {
    const response = await fetchWithTimeout(`${base}/ready/search`, { method: "GET", headers }, timeoutMs);
    const status = statusOf(response);
    if (!isSuccess(response)) return isRetryableStatus(status) ? "transient" : "permanent";
    if (status === 204) return "ready";
    if (!response.validJson) return "unknown";
    const value = response.value;
    if (!value || typeof value !== "object") return "ready";
    const payload = value as Record<string, unknown>;
    const warm = payload.native_warm;
    const warmState = warm && typeof warm === "object" ? String((warm as Record<string, unknown>).state || "") : "";
    if (warmState === "warming" || payload.status === "warming") return "warming";
    if (payload.ok === false) return "warming";
    return "ready";
  } catch {
    // An unreachable readiness endpoint should not add fourteen seconds of
    // polling before the normal POST retry loop gets a chance to recover.
    return "unknown";
  }
}

async function waitUntilReady(headers: Record<string, string>, deadline: number, budget: CallBudget): Promise<ReadyState> {
  let state: ReadyState = "transient";
  for (let poll = 0; poll < budget.readyPolls; poll += 1) {
    const remaining = deadline - now();
    if (remaining <= 0) break;
    state = await probeReady(headers, Math.min(budget.readyTimeoutMs, remaining));
    if (state === "ready" || state === "unknown" || state === "permanent") return state;
    if (poll + 1 >= budget.readyPolls) break;
    // A warming response is expected during a cold Space start. Keep the poll
    // interval short, but let the shared deadline stop it deterministically.
    if (!(await sleepUntil(2_000, deadline))) break;
  }
  return state;
}

// Keep automatic recall useful without adding latency to every self-contained
// prompt.  Explicit memory language and historical/project-decision cues opt in;
// callers can still invoke funes_recall directly for any query.
function shouldRecall(prompt: string): boolean {
  if (prompt.length < 12) return false;
  return /(之前|上次|历史|做过|决定|决策|测试结果|偏好|已有实现|为什么放弃|以前|回忆|prior|previous|history|earlier|last time|we decided|decision|past work|preference|already implemented|old bug|regression)/i.test(prompt);
}

function remoteFailure(message: string, status?: number): RemoteCall {
  return { ok: false, message, ...(status === undefined ? {} : { status }) };
}

function failureForStatus(status: number): string {
  if (status === 401 || status === 403) return `Funes remote authentication failed (HTTP ${status})`;
  if (status === 429) return "Funes remote rate limited the request (HTTP 429)";
  if (status >= 500) return `Funes remote unavailable (HTTP ${status})`;
  return `Funes remote request failed (HTTP ${status})`;
}

async function call(path: string, body: Record<string, unknown>, budget?: CallBudget): Promise<RemoteCall> {
  const isRecall = path === "/search" || path === "/recall";
  const activeBudget = budget ?? (isRecall ? manualRecallBudget : manualBudget);
  if (!base) return remoteFailure("Funes remote URL is not configured");
  if (!token) return remoteFailure("Funes remote API token is not configured");
  let encodedBody: string;
  try {
    encodedBody = JSON.stringify(body);
  } catch {
    return remoteFailure("Funes remote request could not be encoded");
  }

  const headers: Record<string, string> = { "Content-Type": "application/json" };
  if (hubToken) {
    headers.Authorization = `Bearer ${hubToken}`;
    headers["X-Funes-Authorization"] = `Bearer ${token}`;
  } else {
    headers.Authorization = `Bearer ${token}`;
  }

  const deadline = now() + activeBudget.timeoutMs;
  // /search already owns its cold-restore/degraded behavior. Giving a
  // readiness probe inside the bounded automatic budget can suppress an
  // otherwise successful recall, so only the legacy /recall path preflights.
  if (path === "/recall") {
    const ready = await waitUntilReady(headers, deadline, activeBudget);
    // A permanent readiness response (most commonly 401/403) must not be
    // followed by a duplicate POST. A still-warming service can be queried if
    // the bounded poll count was reached before the overall deadline.
    if (ready === "permanent") return remoteFailure("Funes remote readiness check failed");
    if (ready === "warming" && now() >= deadline) return remoteFailure("Funes remote is still warming");
  }

  let lastResponse: FetchResult | undefined;
  let lastError: string | undefined;
  for (let attempt = 0; attempt < activeBudget.attempts; attempt += 1) {
    const remaining = deadline - now();
    if (remaining <= 0) break;
    lastResponse = undefined;
    try {
      const response = await fetchWithTimeout(
        `${base}${path}`,
        {
          method: "POST",
          headers,
          body: encodedBody,
        },
        Math.min(activeBudget.attemptTimeoutMs, remaining),
      );
      const status = statusOf(response);
      if (isSuccess(response)) {
        if (response.validJson) return { ok: true, value: response.value };
        // A truncated/invalid JSON body is transient; let the bounded retry
        // loop recover without changing the response shape for valid calls.
        lastError = "Funes remote returned invalid JSON";
      } else {
        // Never retry authentication or other caller errors.  Only provider /
        // gateway failures and explicit rate limits are transient here.
        if (!isRetryableStatus(status)) return remoteFailure(failureForStatus(status), status);
        lastResponse = response;
        lastError = failureForStatus(status);
      }
    } catch (error) {
      // Fetch failures include connection resets and per-attempt timeouts; all
      // are bounded by the shared deadline and safe to retry.
      lastError = error instanceof RequestTimeoutError
        ? "Funes remote request timed out"
        : "Funes remote request failed before receiving a response";
    }

    if (attempt + 1 >= activeBudget.attempts) break;
    const remainingAfterAttempt = deadline - now();
    if (remainingAfterAttempt <= 0) break;
    if (!(await sleepUntil(retryDelay(lastResponse, attempt, isRecall), deadline))) break;
  }
  const status = lastResponse ? statusOf(lastResponse) : undefined;
  return remoteFailure(lastError || "Funes remote request failed", status);
}

function toolResult(outcome: RemoteCall, name: string) {
  if (outcome.ok) {
    return { content: [{ type: "text", text: JSON.stringify(outcome.value, null, 2) }], details: {} };
  }
  const err = new Error(`${name} error: ${outcome.message}`);
  if (outcome.status !== undefined) (err as any).status = outcome.status;
  throw err;
}

export default function funesRemote(pi: ExtensionAPI) {
  pi.registerTool({
    name: "funes_recall",
    label: "funes recall",
    description: "Search unified Codex, Pi and Claude memory and return original raw context.",
    parameters: {
      type: "object",
      properties: {
        query: { type: "string" },
        limit: { type: "integer" },
        source_agent: { type: "string" },
        source_type: { type: "string" },
        project: { type: "string" },
        repo: { type: "string" },
        device_id: { type: "string" },
        role: { type: "string" },
        content_type: { type: "string" },
        source_missing: { type: "boolean" },
        since: { type: "string" },
        until: { type: "string" },
        facets: { type: "object" },
      },
      required: ["query"],
    },
    execute: async (_id: string, args: Record<string, unknown>) => toolResult(await call("/search", args, manualRecallBudget), "funes_recall"),
  });
  pi.registerTool({
    name: "funes_get",
    label: "funes get",
    description: "Read one original unified memory record.",
    parameters: { type: "object", properties: { record_id: { type: "string" } }, required: ["record_id"] },
    execute: async (_id: string, args: Record<string, unknown>) => toolResult(await call("/get", { id: args.record_id }, manualBudget), "funes_get"),
  });
  pi.on("before_agent_start", async (event) => {
    if (autoRecallTimeoutMs <= 0) return;
    const prompt = String(event.prompt || "").trim();
    if (!shouldRecall(prompt)) return;
    let result: any;
    try {
      const outcome = await call(
        "/search",
        { query: prompt, limit: 5 },
        automaticRecallBudget,
      );
      if (!outcome.ok) return;
      result = outcome.value;
    } catch {
      return;
    }
    const hits = Array.isArray(result?.results) ? result.results : [];
    if (!hits.length) return;
    const context = hits.map((hit: any, i: number) => `${i + 1}. ${String(hit.raw_text || "").slice(0, 1200)}`).join("\n");
    return { systemPrompt: `${event.systemPrompt}\n\n## Funes unified memory\n${context}` };
  });
}
