/**
 * Byte-level BPE tokenizer: a port of train/tokenizer.py.
 *
 * Text is split into chunks by the same regex the tokenizer was trained with,
 * each chunk's UTF-8 bytes are merged pairwise in merge-rank order, and the
 * literal string <|endoftext|> maps to the special EOT token.
 */

export interface TokenizerJSON {
  pattern: string;
  eot: string;
  vocab_size: number;
  merges: [number, number][];
}

const encoder = new TextEncoder();

export class Tokenizer {
  readonly eot: number;
  readonly vocabSize: number;
  readonly eotText: string;
  /** Raw bytes of every token id. */
  readonly bytes: Uint8Array[];
  private readonly ranks = new Map<number, number>();
  private readonly pattern: RegExp;
  private readonly cache = new Map<string, number[]>();
  private readonly textCache: string[];

  constructor(json: TokenizerJSON) {
    this.pattern = new RegExp(json.pattern, "gu");
    this.eotText = json.eot;
    this.eot = 256 + json.merges.length;
    this.vocabSize = this.eot + 1;
    this.bytes = [];
    for (let i = 0; i < 256; i++) this.bytes.push(Uint8Array.of(i));
    json.merges.forEach(([a, b], i) => {
      this.ranks.set(a * 65536 + b, i);
      const ab = new Uint8Array(this.bytes[a].length + this.bytes[b].length);
      ab.set(this.bytes[a]);
      ab.set(this.bytes[b], this.bytes[a].length);
      this.bytes.push(ab);
    });
    this.bytes.push(encoder.encode(json.eot));
    const dec = new TextDecoder();
    this.textCache = this.bytes.map((b) => dec.decode(b));
  }

  encode(text: string): number[] {
    const out: number[] = [];
    text.split(this.eotText).forEach((part, i) => {
      if (i > 0) out.push(this.eot);
      for (const m of part.matchAll(this.pattern)) {
        for (const id of this.encodeChunk(m[0])) out.push(id);
      }
    });
    return out;
  }

  private encodeChunk(chunk: string): number[] {
    const cached = this.cache.get(chunk);
    if (cached) return cached;
    let ids = Array.from(encoder.encode(chunk));
    while (ids.length > 1) {
      let best = -1;
      let bestRank = Infinity;
      for (let i = 0; i < ids.length - 1; i++) {
        const r = this.ranks.get(ids[i] * 65536 + ids[i + 1]);
        if (r !== undefined && r < bestRank) {
          bestRank = r;
          best = i;
        }
      }
      if (best < 0) break;
      const a = ids[best];
      const b = ids[best + 1];
      const merged: number[] = [];
      for (let i = 0; i < ids.length; i++) {
        if (i < ids.length - 1 && ids[i] === a && ids[i + 1] === b) {
          merged.push(256 + bestRank);
          i++;
        } else {
          merged.push(ids[i]);
        }
      }
      ids = merged;
    }
    if (this.cache.size < 50_000) this.cache.set(chunk, ids);
    return ids;
  }

  decode(ids: number[]): string {
    let n = 0;
    for (const id of ids) n += this.bytes[id].length;
    const buf = new Uint8Array(n);
    let o = 0;
    for (const id of ids) {
      buf.set(this.bytes[id], o);
      o += this.bytes[id].length;
    }
    return new TextDecoder().decode(buf);
  }

  /** One token's text on its own (partial UTF-8 shows as U+FFFD). */
  tokenText(id: number): string {
    return this.textCache[id];
  }
}
