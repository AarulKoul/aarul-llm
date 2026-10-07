/**
 * The AARUL forward pass in plain TypeScript: a port of train/model.py.
 *
 * Tokens are fed one at a time. Each layer keeps a key/value cache, so
 * position t only computes its own query/key/value and attends over the
 * cached keys, instead of re-running the whole prefix every step.
 *
 * Besides logits, every step returns the attention weights of the new token
 * over all previous positions, per layer and head: the "where it looked"
 * view in the UI.
 */

import type { Config, LayerWeights, Weights } from "./weights.ts";

export interface StepResult {
  logits: Float32Array;
  /** attn[layer][head][s]: how much the new token attended to position s. */
  attn: Float32Array[][];
}

export class Model {
  readonly cfg: Config;
  private readonly w: Weights;
  private readonly headDim: number;
  private readonly cos: Float32Array;
  private readonly sin: Float32Array;
  private readonly kCache: Float32Array[];
  private readonly vCache: Float32Array[];
  // Scratch buffers, allocated once.
  private readonly x: Float32Array;
  private readonly xn: Float32Array;
  private readonly q: Float32Array;
  private readonly att: Float32Array;
  private readonly proj: Float32Array;
  private readonly h1: Float32Array;
  private readonly h3: Float32Array;
  private readonly scores: Float32Array;
  /** Number of tokens already in the cache. */
  pos = 0;

  constructor(weights: Weights) {
    this.w = weights;
    const c = (this.cfg = weights.config);
    this.headDim = c.d_model / c.n_head;
    const half = this.headDim / 2;
    this.cos = new Float32Array(c.ctx * half);
    this.sin = new Float32Array(c.ctx * half);
    for (let i = 0; i < half; i++) {
      const invFreq = Math.fround(c.rope_theta ** (-i / half));
      for (let p = 0; p < c.ctx; p++) {
        const a = Math.fround(p * invFreq);
        this.cos[p * half + i] = Math.cos(a);
        this.sin[p * half + i] = Math.sin(a);
      }
    }
    this.kCache = weights.layers.map(() => new Float32Array(c.ctx * c.d_model));
    this.vCache = weights.layers.map(() => new Float32Array(c.ctx * c.d_model));
    this.x = new Float32Array(c.d_model);
    this.xn = new Float32Array(c.d_model);
    this.q = new Float32Array(c.d_model);
    this.att = new Float32Array(c.d_model);
    this.proj = new Float32Array(c.d_model);
    this.h1 = new Float32Array(c.d_ff);
    this.h3 = new Float32Array(c.d_ff);
    this.scores = new Float32Array(c.ctx);
  }

  reset(): void {
    this.pos = 0;
  }

  /** Feed one token at position this.pos; returns next-token logits. */
  step(token: number, wantAttn = true): StepResult {
    const c = this.cfg;
    const d = c.d_model;
    const t = this.pos;
    if (t >= c.ctx) throw new Error(`context full (${c.ctx} tokens)`);
    const { x, xn, q, att, proj, h1, h3 } = this;

    x.set(this.w.tokEmb.subarray(token * d, token * d + d));
    const attn: Float32Array[][] = [];

    for (let l = 0; l < c.n_layer; l++) {
      const L: LayerWeights = this.w.layers[l];
      const kc = this.kCache[l];
      const vc = this.vCache[l];

      // Attention: write this position's k and v straight into the cache.
      rmsnorm(xn, x, L.attnNorm, c.norm_eps);
      matvec(q, L.wq, xn);
      const k = kc.subarray(t * d, t * d + d);
      const v = vc.subarray(t * d, t * d + d);
      matvec(k, L.wk, xn);
      matvec(v, L.wv, xn);
      this.rope(q, t);
      this.rope(k, t);
      attn.push(this.attend(kc, vc, t, wantAttn));
      matvec(proj, L.wo, att);
      for (let i = 0; i < d; i++) x[i] += proj[i];

      // SwiGLU MLP.
      rmsnorm(xn, x, L.mlpNorm, c.norm_eps);
      matvec(h1, L.w1, xn);
      matvec(h3, L.w3, xn);
      for (let i = 0; i < c.d_ff; i++) {
        const g = h1[i];
        h1[i] = (g / (1 + Math.exp(-g))) * h3[i];
      }
      matvec(proj, L.w2, h1);
      for (let i = 0; i < d; i++) x[i] += proj[i];
    }

    rmsnorm(xn, x, this.w.norm, c.norm_eps);
    const logits = new Float32Array(c.vocab_size);
    matvec(logits, this.w.tokEmb, xn); // output layer is tied to the embedding
    this.pos++;
    return { logits, attn };
  }

  /** Causal attention of the query in this.q over cached positions 0..t, per head. */
  private attend(kc: Float32Array, vc: Float32Array, t: number, wantAttn: boolean): Float32Array[] {
    const { d_model: d, n_head } = this.cfg;
    const D = this.headDim;
    const scale = 1 / Math.sqrt(D);
    const { q, att, scores } = this;
    const perHead: Float32Array[] = [];
    att.fill(0);
    for (let h = 0; h < n_head; h++) {
      const ho = h * D;
      let max = -Infinity;
      for (let s = 0; s <= t; s++) {
        const ko = s * d + ho;
        let dot = 0;
        for (let i = 0; i < D; i++) dot += q[ho + i] * kc[ko + i];
        dot *= scale;
        scores[s] = dot;
        if (dot > max) max = dot;
      }
      let sum = 0;
      for (let s = 0; s <= t; s++) {
        const e = Math.exp(scores[s] - max);
        scores[s] = e;
        sum += e;
      }
      for (let s = 0; s <= t; s++) {
        const p = scores[s] / sum;
        scores[s] = p;
        const vo = s * d + ho;
        for (let i = 0; i < D; i++) att[ho + i] += p * vc[vo + i];
      }
      if (wantAttn) perHead.push(scores.slice(0, t + 1));
    }
    return perHead;
  }

  /** Rotary position embedding, "rotate half" layout, applied per head in place. */
  private rope(vec: Float32Array, pos: number): void {
    const D = this.headDim;
    const half = D / 2;
    const base = pos * half;
    for (let h = 0; h < this.cfg.n_head; h++) {
      const o = h * D;
      for (let i = 0; i < half; i++) {
        const cs = this.cos[base + i];
        const sn = this.sin[base + i];
        const a = vec[o + i];
        const b = vec[o + half + i];
        vec[o + i] = a * cs - b * sn;
        vec[o + half + i] = b * cs + a * sn;
      }
    }
  }
}

/**
 * out = W @ x, with W stored row-major as (out.length, x.length), like nn.Linear.
 * Four rows at a time, so each x[i] is loaded once per four multiply-adds: about
 * 1.8x faster than one row at a time. Sums accumulate in float64 (JS numbers).
 */
export function matvec(out: Float32Array, W: Float32Array, x: Float32Array): void {
  const n = x.length;
  const rows = out.length;
  let r = 0;
  for (; r + 3 < rows; r += 4) {
    const o0 = r * n;
    const o1 = o0 + n;
    const o2 = o1 + n;
    const o3 = o2 + n;
    let s0 = 0;
    let s1 = 0;
    let s2 = 0;
    let s3 = 0;
    for (let i = 0; i < n; i++) {
      const xi = x[i];
      s0 += W[o0 + i] * xi;
      s1 += W[o1 + i] * xi;
      s2 += W[o2 + i] * xi;
      s3 += W[o3 + i] * xi;
    }
    out[r] = s0;
    out[r + 1] = s1;
    out[r + 2] = s2;
    out[r + 3] = s3;
  }
  for (; r < rows; r++) {
    const o = r * n;
    let s = 0;
    for (let i = 0; i < n; i++) s += W[o + i] * x[i];
    out[r] = s;
  }
}

function rmsnorm(out: Float32Array, x: Float32Array, w: Float32Array, eps: number): void {
  let ss = 0;
  for (let i = 0; i < x.length; i++) ss += x[i] * x[i];
  const inv = 1 / Math.sqrt(ss / x.length + eps);
  for (let i = 0; i < x.length; i++) out[i] = x[i] * inv * w[i];
}
