/** Turning logits into a chosen token, plus the numbers the UI shows about that choice. */

export interface Candidate {
  id: number;
  /** Probability under the model's raw (temperature 1) distribution. */
  p: number;
}

export interface SampleOptions {
  /** 0 means always take the most likely token. */
  temperature: number;
  /** Only sample among the k most likely tokens; 0 disables the cut. */
  topK: number;
}

export function softmax(logits: Float32Array, temperature = 1): Float32Array {
  const out = new Float32Array(logits.length);
  let max = -Infinity;
  for (let i = 0; i < logits.length; i++) if (logits[i] > max) max = logits[i];
  let sum = 0;
  for (let i = 0; i < logits.length; i++) {
    const e = Math.exp((logits[i] - max) / temperature);
    out[i] = e;
    sum += e;
  }
  for (let i = 0; i < out.length; i++) out[i] /= sum;
  return out;
}

export function topCandidates(probs: Float32Array, k: number): Candidate[] {
  const ids = Array.from(probs.keys());
  ids.sort((a, b) => probs[b] - probs[a]);
  return ids.slice(0, k).map((id) => ({ id, p: probs[id] }));
}

export function sample(logits: Float32Array, opts: SampleOptions, rand: () => number): number {
  if (opts.temperature <= 0) {
    let best = 0;
    for (let i = 1; i < logits.length; i++) if (logits[i] > logits[best]) best = i;
    return best;
  }
  const probs = softmax(logits, opts.temperature);
  const pool =
    opts.topK > 0 && opts.topK < probs.length
      ? topCandidates(probs, opts.topK)
      : Array.from(probs, (p, id) => ({ id, p }));
  let total = 0;
  for (const c of pool) total += c.p;
  let r = rand() * total;
  for (const c of pool) {
    r -= c.p;
    if (r <= 0) return c.id;
  }
  return pool[pool.length - 1].id;
}

/** Small seeded PRNG (mulberry32), so a seed reproduces a story exactly. */
export function rng(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
