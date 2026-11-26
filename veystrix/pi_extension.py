"""The Veystrix provider extension for the pi coding agent, as installable TypeScript.

``python -m veystrix.harness_configure --apply`` writes ``~/.pi/agent/extensions/veystrix.ts``
from ``EXTENSION_SOURCE`` (idempotent: an unchanged file is left alone and a changed file gets
a timestamped backup). The source lives here, not in ``integrations/``, so the packaged app can
install it: the engine bundle ships only the ``veystrix`` and ``models`` packages.

The extension discovers models from the engine's ``GET /v1/models``: ``context_window`` is used
as-is, and ``max_output_tokens`` (the server's own output ceiling) becomes pi's ``maxTokens``.
pi then never asks for more output than fits, and the engine clamps whatever the client asks to
the room actually left in the window — uncapped generation in practice.
"""

EXTENSION_SOURCE = '''\
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";

// Installed by `python -m veystrix.harness_configure` (canonical source: veystrix/pi_extension.py).
// Override either default with environment variables when the engine runs elsewhere or needs a key.
const BASE = process.env.VEYSTRIX_BASE_URL ?? "http://127.0.0.1:8080/v1";
// The engine accepts any value on loopback unless it was started with a required key.
const API_KEY = process.env.VEYSTRIX_API_KEY ?? "local";

type ServerModel = {
  id: string;
  name?: string;
  family?: string;
  context_window?: number;
  max_output_tokens?: number;
};

function makeModelEntry(m: ServerModel, idOverride?: string) {
  // The engine clamps any requested output cap to the room left in the window, so the
  // server's own ceiling is "unlimited output": never the bogus 4096 default.
  const contextWindow = m.context_window ?? 16384;
  const maxTokens = Math.max(1, m.max_output_tokens ?? contextWindow);
  const id = idOverride ?? m.id;
  const qwen = m.family === "flashnext" || id.startsWith("veystrix-flashnext");
  const nemotron = m.family === "nemotron35-lightning" || id.startsWith("veystrix-nemotron35-lightning");
  const ornith = m.family === "ornith" || id.startsWith("veystrix-ornith");
  const inkling = m.family === "inkling2-mini" || id.startsWith("veystrix-inkling2-mini");

  return {
    id,
    name: m.name ?? id,
    reasoning: inkling || qwen || nemotron || ornith,
    thinkingLevelMap: nemotron
      ? { minimal: null, low: null, medium: "medium", high: null, xhigh: null, max: null }
      : { minimal: null, xhigh: "xhigh", max: null },
    compat: {
      supportsDeveloperRole: false,
      thinkingFormat: "qwen" as const,
      // Nemotron sends only medium, avoiding a Qwen-specific server default.
      supportsReasoningEffort: inkling || qwen || nemotron || ornith,
      supportsThinkingTokenBudget: false,
      sendSessionAffinityHeaders: true,
      // This Pi header format uses X-Session-Id, which the router forwards.
      sessionAffinityFormat: "openrouter" as const,
    },
    input: ["text"] as ("text" | "image")[],
    cost: {
      input: 0,
      output: 0,
      cacheRead: 0,
      cacheWrite: 0
    },
    contextWindow,
    maxTokens
  };
}

const FALLBACK_MODELS: ServerModel[] = [
  { id: "veystrix-inkling2-mini", name: "Veystrix Inkling 2 Mini", family: "inkling2-mini", context_window: 1048576 },
  { id: "veystrix-flashnext", name: "Veystrix Flash-Next", family: "flashnext", context_window: 262144 },
  { id: "veystrix-ornith", name: "Veystrix Ornith 1.5 35B-A3B", family: "ornith", context_window: 262144 },
  { id: "veystrix-nemotron35-lightning", name: "Veystrix Nemotron 3.5 Lightning", family: "nemotron35-lightning", context_window: 1048576 },
];

async function discover(signal?: AbortSignal) {
  const response = await fetch(`${BASE}/models`, {
    signal,
    headers: { Authorization: `Bearer ${API_KEY}` },
  });

  if (!response.ok) {
    throw new Error(`Veystrix returned HTTP ${response.status}`);
  }

  const payload = await response.json() as {
    data?: ServerModel[];
  };

  if (!Array.isArray(payload.data)) {
    throw new Error("Invalid Veystrix model catalog");
  }

  const seen = new Set<string>();
  const entries = [];
  for (const m of payload.data) {
    if (typeof m.id !== "string" || m.id.length === 0) continue;
    if (!seen.has(m.id)) {
      seen.add(m.id);
      entries.push(makeModelEntry(m));
    }
    if (typeof m.family === "string" && m.family.length > 0) {
      const alias = `veystrix-${m.family}`;
      if (!seen.has(alias)) {
        seen.add(alias);
        entries.push(makeModelEntry(m, alias));
      }
    }
  }
  return entries;
}

export default async function (pi: ExtensionAPI) {
  let models: Awaited<ReturnType<typeof discover>> = [];

  try {
    models = await discover(AbortSignal.timeout(3000));
  } catch {
    models = FALLBACK_MODELS.map(m => makeModelEntry(m));
  }
  if (models.length === 0) {
    models = FALLBACK_MODELS.map(m => makeModelEntry(m));
  }

  pi.registerProvider("veystrix", {
    name: "Veystrix Local",
    baseUrl: BASE,
    apiKey: API_KEY,
    api: "openai-completions",
    models,
    async refreshModels({ signal }) {
      return discover(signal);
    }
  });
}
'''
