/**
 * Runs the model off the main thread. The page sends a prompt; the worker
 * streams back one TokenEvent per token, carrying what the model considered
 * and where it looked.
 */

import { Model } from "./model.ts";
import type { FromWorker, ToWorker } from "./protocol.ts";
import { rng, sample, softmax, topCandidates } from "./sampler.ts";
import { Tokenizer } from "./tokenizer.ts";
import { parseWeights } from "./weights.ts";

let model: Model | null = null;
let tok: Tokenizer | null = null;
let generation = 0;

const post = (msg: FromWorker, transfer: Transferable[] = []) => self.postMessage(msg, { transfer });

async function fetchWithProgress(url: string): Promise<ArrayBuffer> {
  const res = await fetch(url);
  if (!res.ok || !res.body) throw new Error(`failed to fetch ${url}: ${res.status}`);
  const total = Number(res.headers.get("content-length")) || 0;
  const reader = res.body.getReader();
  const parts: Uint8Array[] = [];
  let loaded = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    parts.push(value);
    loaded += value.length;
    post({ type: "progress", loaded, total });
  }
  const buf = new Uint8Array(loaded);
  let o = 0;
  for (const p of parts) {
    buf.set(p, o);
    o += p.length;
  }
  return buf.buffer;
}

async function load(modelUrl: string, tokenizerUrl: string) {
  const [buf, tj] = await Promise.all([
    fetchWithProgress(modelUrl),
    fetch(tokenizerUrl).then((r) => r.json()),
  ]);
  tok = new Tokenizer(tj);
  const weights = parseWeights(buf);
  model = new Model(weights);
  const vocab = Array.from({ length: tok.vocabSize }, (_, i) => tok!.tokenText(i));
  post({ type: "ready", config: weights.config, meta: weights.meta, vocab, eot: tok.eot });
}

/** Feed `id` at the next position, and report it along with the step that predicted it. */
function emit(run: number, pos: number, id: number, prompt: boolean, prev: { probs: Float32Array; attn: Float32Array; ms: number }) {
  post(
    { type: "token", run, pos, id, prompt, p: prev.probs[id], top: topCandidates(prev.probs, 10), attn: prev.attn, ms: prev.ms },
    [prev.attn.buffer],
  );
}

function forward(id: number) {
  const t0 = performance.now();
  const { logits, attn } = model!.step(id);
  const ms = performance.now() - t0;
  const n = attn[0][0].length;
  const flat = new Float32Array(attn.length * attn[0].length * n);
  let o = 0;
  for (const layer of attn) for (const head of layer) flat.set(head, (o++) * n);
  return { logits, probs: softmax(logits), attn: flat, ms };
}

async function generate(msg: Extract<ToWorker, { type: "generate" }>) {
  const me = ++generation;
  const { run } = msg;
  const done = (reason: "eot" | "max" | "context" | "stopped") => post({ type: "done", run, reason });
  const m = model!;
  const t = tok!;
  const rand = rng(msg.seed);
  m.reset();

  const ids = [t.eot, ...t.encode(msg.prompt)].slice(0, m.cfg.ctx);
  let prev = forward(ids[0]);
  for (let i = 1; i < ids.length; i++) {
    emit(run, i, ids[i], true, prev);
    prev = forward(ids[i]);
  }

  let made = 0;
  for (;;) {
    if (me !== generation) return done("stopped");
    if (made >= msg.maxTokens) return done("max");
    if (m.pos >= m.cfg.ctx) return done("context");
    const id = sample(prev.logits, msg, rand);
    if (id === t.eot) return done("eot");
    emit(run, m.pos, id, false, prev);
    made++;
    prev = forward(id);
    // Yield so "stop" (and the next generate) can arrive; optionally slow down.
    await new Promise((r) => setTimeout(r, msg.delayMs));
  }
}

self.onmessage = async (e: MessageEvent<ToWorker>) => {
  const msg = e.data;
  try {
    if (msg.type === "load") await load(msg.modelUrl, msg.tokenizerUrl);
    else if (msg.type === "generate") await generate(msg);
    else if (msg.type === "stop") generation++;
  } catch (err) {
    post({ type: "error", message: err instanceof Error ? err.message : String(err) });
  }
};
