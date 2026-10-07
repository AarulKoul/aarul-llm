/** Messages between the page and the inference worker. */

import type { Candidate } from "./sampler.ts";
import type { Config } from "./weights.ts";

export type ToWorker =
  | { type: "load"; modelUrl: string; tokenizerUrl: string }
  | {
      type: "generate";
      /** Echoed on every event of this generation, so stale events can be ignored. */
      run: number;
      prompt: string;
      temperature: number;
      topK: number;
      maxTokens: number;
      seed: number;
      /** Pause between generated tokens, so people can watch. */
      delayMs: number;
    }
  | { type: "stop" };

export interface TokenEvent {
  type: "token";
  run: number;
  /** Position in the sequence; position 0 is the start-of-story token. */
  pos: number;
  id: number;
  /** true for tokens the user typed, false for tokens the model chose. */
  prompt: boolean;
  /** Probability the model gave this token, before it was chosen or typed. */
  p: number;
  /** The model's most likely candidates at this position. */
  top: Candidate[];
  /**
   * attn[(layer * n_head + head) * pos + s]: how much the step that produced
   * this token attended to position s (s < pos).
   */
  attn: Float32Array;
  /** Milliseconds the forward pass took. */
  ms: number;
}

export type FromWorker =
  | { type: "progress"; loaded: number; total: number }
  | {
      type: "ready";
      config: Config;
      meta: Record<string, unknown>;
      vocab: string[];
      eot: number;
    }
  | TokenEvent
  | { type: "done"; run: number; reason: "eot" | "max" | "context" | "stopped" }
  | { type: "error"; message: string };
