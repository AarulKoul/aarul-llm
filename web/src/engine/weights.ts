/**
 * Reader for aarul.bin, the format written by train/export.py:
 *
 *   "ARUL"  u32 version  u32 header_len  header JSON  tensor data
 *
 * Every tensor comes out as a Float32Array; f16 and q8 (int8 + per-row
 * float32 scale) tensors are expanded on load.
 */

export interface Config {
  vocab_size: number;
  ctx: number;
  d_model: number;
  n_layer: number;
  n_head: number;
  d_ff: number;
  rope_theta: number;
  norm_eps: number;
}

export interface LayerWeights {
  attnNorm: Float32Array;
  wq: Float32Array;
  wk: Float32Array;
  wv: Float32Array;
  wo: Float32Array;
  mlpNorm: Float32Array;
  w1: Float32Array;
  w3: Float32Array;
  w2: Float32Array;
}

export interface Weights {
  config: Config;
  meta: Record<string, unknown>;
  tokEmb: Float32Array;
  layers: LayerWeights[];
  norm: Float32Array;
}

interface TensorEntry {
  name: string;
  shape: number[];
  dtype: "f32" | "f16" | "q8";
  offset: number;
  scale_offset?: number;
}

export function parseWeights(buf: ArrayBuffer): Weights {
  const view = new DataView(buf);
  const magic = String.fromCharCode(...new Uint8Array(buf, 0, 4));
  if (magic !== "ARUL") throw new Error(`not an AARUL model file (magic ${JSON.stringify(magic)})`);
  const version = view.getUint32(4, true);
  if (version !== 1) throw new Error(`unsupported model version ${version}`);
  const headerLen = view.getUint32(8, true);
  const header = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 12, headerLen)));
  const base = 12 + headerLen;

  const tensors = new Map<string, Float32Array>();
  for (const t of header.tensors as TensorEntry[]) {
    const n = t.shape.reduce((a, b) => a * b, 1);
    const start = base + t.offset;
    let out: Float32Array;
    if (t.dtype === "f32") {
      out = new Float32Array(buf.slice(start, start + n * 4));
    } else if (t.dtype === "f16") {
      const h = new Uint16Array(buf, start, n);
      out = new Float32Array(n);
      for (let i = 0; i < n; i++) out[i] = halfToFloat(h[i]);
    } else {
      const q = new Int8Array(buf, start, n);
      const rows = t.shape[0];
      const cols = n / rows;
      const scale = new Float32Array(buf, base + t.scale_offset!, rows);
      out = new Float32Array(n);
      for (let r = 0; r < rows; r++) {
        const s = scale[r];
        const o = r * cols;
        for (let c = 0; c < cols; c++) out[o + c] = q[o + c] * s;
      }
    }
    tensors.set(t.name, out);
  }

  const get = (name: string) => {
    const t = tensors.get(name);
    if (!t) throw new Error(`model file is missing tensor ${name}`);
    return t;
  };
  const config = header.config as Config;
  const layers: LayerWeights[] = [];
  for (let i = 0; i < config.n_layer; i++) {
    const p = `blocks.${i}.`;
    layers.push({
      attnNorm: get(p + "attn_norm"),
      wq: get(p + "wq"),
      wk: get(p + "wk"),
      wv: get(p + "wv"),
      wo: get(p + "wo"),
      mlpNorm: get(p + "mlp_norm"),
      w1: get(p + "w1"),
      w3: get(p + "w3"),
      w2: get(p + "w2"),
    });
  }
  return { config, meta: header.meta ?? {}, tokEmb: get("tok_emb"), layers, norm: get("norm") };
}

function halfToFloat(h: number): number {
  const sign = h & 0x8000 ? -1 : 1;
  const exp = (h >> 10) & 0x1f;
  const frac = h & 0x3ff;
  if (exp === 0) return sign * 2 ** -14 * (frac / 1024);
  if (exp === 31) return frac ? NaN : sign * Infinity;
  return sign * 2 ** (exp - 15) * (1 + frac / 1024);
}
