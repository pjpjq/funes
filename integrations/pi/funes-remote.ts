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

const automaticRecallBudget: CallBudget = {
  timeoutMs: 4_250,
  attempts: 1,
  attemptTimeoutMs: 4_250,
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

function retryDelay(response: FetchResult | undefined, attempt: number): number {
  const exponential = Math.min(30_000, 1_000 * 2 ** (attempt + 1));
  const retryAfter = response?.response.headers?.get("retry-after");
  if (!retryAfter) return exponential;
  const seconds = Number(retryAfter);
  if (Number.isFinite(seconds) && seconds >= 0) return Math.max(exponential, Math.min(30_000, seconds * 1_000));
  const date = Date.parse(retryAfter);
  if (Number.isFinite(date)) return Math.max(exponential, Math.min(30_000, Math.max(0, date - now())));
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

async function call(path: string, body: Record<string, unknown>, budget: CallBudget = manualBudget) {
  if (!base || !token) return null;
  let encodedBody: string;
  try {
    encodedBody = JSON.stringify(body);
  } catch {
    return null;
  }

  const headers: Record<string, string> = { "Content-Type": "application/json" };
  if (hubToken) {
    headers.Authorization = `Bearer ${hubToken}`;
    headers["X-Funes-Authorization"] = `Bearer ${token}`;
  } else {
    headers.Authorization = `Bearer ${token}`;
  }

  const deadline = now() + budget.timeoutMs;
  // /search already owns its cold-restore/degraded behavior. Giving a
  // readiness probe half of the automatic 4.25 second budget can suppress an
  // otherwise successful recall, so only the legacy /recall path preflights.
  if (path === "/recall") {
    const ready = await waitUntilReady(headers, deadline, budget);
    // A permanent readiness response (most commonly 401/403) must not be
    // followed by a duplicate POST. A still-warming service can be queried if
    // the bounded poll count was reached before the overall deadline.
    if (ready === "permanent" || (ready === "warming" && now() >= deadline)) return null;
  }

  let lastResponse: FetchResult | undefined;
  for (let attempt = 0; attempt < budget.attempts; attempt += 1) {
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
        Math.min(budget.attemptTimeoutMs, remaining),
      );
      const status = statusOf(response);
      if (isSuccess(response)) {
        if (response.validJson) return response.value;
        // A truncated/invalid JSON body is transient; let the bounded retry
        // loop recover without changing the response shape for valid calls.
      } else {
        // Never retry authentication or other caller errors.  Only provider /
        // gateway failures and explicit rate limits are transient here.
        if (!isRetryableStatus(status)) return null;
        lastResponse = response;
      }
    } catch {
      // Fetch failures include connection resets and per-attempt timeouts; all
      // are bounded by the shared deadline and safe to retry.
    }

    if (attempt + 1 >= budget.attempts) break;
    const remainingAfterAttempt = deadline - now();
    if (remainingAfterAttempt <= 0) break;
    if (!(await sleepUntil(retryDelay(lastResponse, attempt), deadline))) break;
  }
  return null;
}

export default function funesRemote(pi: ExtensionAPI) {
  pi.registerTool({
    name: "funes_recall",
    label: "funes recall",
    description: "Search unified Codex, Pi and Claude memory and return original raw context.",
    parameters: { type: "object", properties: { query: { type: "string" }, limit: { type: "integer" } }, required: ["query"] },
    execute: async (_id: string, args: Record<string, unknown>) => ({ content: [{ type: "text", text: JSON.stringify(await call("/search", args), null, 2) }], details: {} }),
  });
  pi.registerTool({
    name: "funes_get",
    label: "funes get",
    description: "Read one original unified memory record.",
    parameters: { type: "object", properties: { record_id: { type: "string" } }, required: ["record_id"] },
    execute: async (_id: string, args: Record<string, unknown>) => ({ content: [{ type: "text", text: JSON.stringify(await call("/get", { id: args.record_id }), null, 2) }], details: {} }),
  });
  pi.on("before_agent_start", async (event) => {
    const prompt = String(event.prompt || "").trim();
    if (!shouldRecall(prompt)) return;
    let result: any;
    try {
      result = await call(
        "/search",
        { query: prompt, limit: 5 },
        automaticRecallBudget,
      );
    } catch {
      return;
    }
    const hits = Array.isArray(result?.results) ? result.results : [];
    if (!hits.length) return;
    const context = hits.map((hit: any, i: number) => `${i + 1}. ${String(hit.raw_text || "").slice(0, 1200)}`).join("\n");
    return { systemPrompt: `${event.systemPrompt}\n\n## Funes unified memory\n${context}` };
  });
}
