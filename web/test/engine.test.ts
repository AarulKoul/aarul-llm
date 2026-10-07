/**
 * The browser engine must agree with PyTorch. train/export.py --fixtures
 * writes a tiny random model plus PyTorch's outputs for it; these tests run
 * the same inputs through the TypeScript engine and compare.
 */

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import { Model } from "../src/engine/model.ts";
import { rng, sample, softmax } from "../src/engine/sampler.ts";
import { Tokenizer } from "../src/engine/tokenizer.ts";
import { parseWeights } from "../src/engine/weights.ts";

const fixture = (name: string) => new URL(`./fixtures/${name}`, import.meta.url);
const ref = JSON.parse(readFileSync(fixture("reference.json"), "utf8"));
const tok = new Tokenizer(JSON.parse(readFileSync(fixture("tokenizer.json"), "utf8")));

function loadModel(name: string): Model {
  const b = readFileSync(fixture(name));
  return new Model(parseWeights(b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength)));
}

function maxAbsDiff(a: ArrayLike<number>, b: ArrayLike<number>): number {
  assert.equal(a.length, b.length);
  let m = 0;
  for (let i = 0; i < a.length; i++) m = Math.max(m, Math.abs(a[i] - b[i]));
  return m;
}

function run(model: Model, ids: number[]) {
  model.reset();
  return ids.map((id) => model.step(id));
}

test("tokenizer matches the Python tokenizer exactly", () => {
  for (const { text, ids } of ref.tokenizer) {
    assert.deepEqual(tok.encode(text), ids, `encoding ${JSON.stringify(text)}`);
    assert.equal(tok.decode(ids), text);
  }
});

test("float32 forward pass matches PyTorch logits", () => {
  const steps = run(loadModel("tiny_f32.bin"), ref.model.ids);
  const first = maxAbsDiff(steps[0].logits, ref.model.logits_first);
  const last = maxAbsDiff(steps[steps.length - 1].logits, ref.model.logits_last);
  assert.ok(first < 1e-4, `first-position logits differ by ${first}`);
  assert.ok(last < 1e-4, `last-position logits differ by ${last}`);
});

test("attention weights match PyTorch, for every layer and head", () => {
  const steps = run(loadModel("tiny_f32.bin"), ref.model.ids);
  const attn = steps[steps.length - 1].attn;
  ref.model.attn_last.forEach((layer: number[][], l: number) =>
    layer.forEach((head: number[], h: number) => {
      const d = maxAbsDiff(attn[l][h], head);
      assert.ok(d < 1e-5, `layer ${l} head ${h} attention differs by ${d}`);
    }),
  );
});

test("int8 weights load to exactly the values PyTorch dequantizes", () => {
  const steps = run(loadModel("tiny_q8.bin"), ref.model.ids);
  const d = maxAbsDiff(steps[steps.length - 1].logits, ref.model.logits_last_q8);
  assert.ok(d < 1e-4, `q8 logits differ by ${d}`);
});

test("KV cache: stepping token by token equals a fresh run on the prefix", () => {
  const model = loadModel("tiny_f32.bin");
  const ids: number[] = ref.model.ids;
  const full = run(model, ids);
  const prefix = run(model, ids.slice(0, 5));
  assert.equal(maxAbsDiff(prefix[4].logits, full[4].logits), 0);
});

test("sampling: temperature 0 is argmax, and a seed reproduces a choice sequence", () => {
  const logits = Float32Array.from([0.1, 2.5, -1, 2.4]);
  assert.equal(sample(logits, { temperature: 0, topK: 0 }, Math.random), 1);
  assert.equal(sample(logits, { temperature: 1, topK: 1 }, Math.random), 1);
  const draw = (seed: number) => {
    const r = rng(seed);
    return Array.from({ length: 20 }, () => sample(logits, { temperature: 1, topK: 3 }, r));
  };
  assert.deepEqual(draw(7), draw(7));
  assert.ok(!draw(7).includes(2), "top-k 3 must never pick the least likely token");
  const p = softmax(logits);
  assert.ok(Math.abs(p.reduce((a, b) => a + b, 0) - 1) < 1e-6);
});
