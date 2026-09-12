// Funes remote recall for pi. The URL/token stay in the process environment;
// this file contains no credential or machine-specific value.
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { execFileSync } from "node:child_process";

const base = (process.env.FUNES_REMOTE_URL || "").replace(/\/+$/, "");

function keychain(service: string): string {
  if (process.platform !== "darwin") return "";
  try {
    return execFileSync("/usr/bin/security", [
      "find-generic-password", "-a", process.env.USER || "", "-s", service, "-w",
    ], { encoding: "utf8", stdio: ["ignore", "pipe", "ignore"] }).trim();
  } catch { return ""; }
}

const token = process.env.FUNES_API_TOKEN || keychain("funes-api-token");
const hubToken = process.env.FUNES_HF_TOKEN || process.env.HF_TOKEN || keychain("funes-hf-token");

// Keep automatic recall useful without adding latency to every self-contained
// prompt.  Explicit memory language and historical/project-decision cues opt in;
// callers can still invoke funes_recall directly for any query.
function shouldRecall(prompt: string): boolean {
  if (prompt.length < 12) return false;
  return /(之前|上次|历史|做过|决定|决策|测试结果|偏好|已有实现|为什么放弃|以前|回忆|prior|previous|history|earlier|last time|we decided|decision|past work|preference|already implemented|old bug|regression)/i.test(prompt);
}

async function call(path: string, body: Record<string, unknown>) {
  if (!base || !token) return null;
  try {
    const headers: Record<string, string> = { "Content-Type": "application/json" };
    if (hubToken) {
      headers.Authorization = `Bearer ${hubToken}`;
      headers["X-Funes-Authorization"] = `Bearer ${token}`;
    } else {
      headers.Authorization = `Bearer ${token}`;
    }
    const response = await fetch(`${base}${path}`, {
      method: "POST",
      headers,
      body: JSON.stringify(body),
      signal: AbortSignal.timeout(20_000),
    });
    return response.ok ? await response.json() : null;
  } catch { return null; }
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
    const result: any = await call("/search", { query: prompt, limit: 5 });
    const hits = Array.isArray(result?.results) ? result.results : [];
    if (!hits.length) return;
    const context = hits.map((hit: any, i: number) => `${i + 1}. ${String(hit.raw_text || "").slice(0, 1200)}`).join("\n");
    return { systemPrompt: `${event.systemPrompt}\n\n## Funes unified memory\n${context}` };
  });
}
