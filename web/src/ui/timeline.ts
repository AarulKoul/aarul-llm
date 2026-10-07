/** "Watch it learn": the loss curve, plus what the model wrote at each point in training. */

import type { Config } from "../engine/weights.ts";
import { compact, el } from "./format.ts";

export interface Training {
  config: { model: Config; train: Record<string, number>; params: number; gpu: string };
  train_loss: [number, number][];
  val_loss: [number, number][];
  samples: { step: number; tokens: number; text: string }[];
  meta: {
    step: number;
    params: number;
    gpu: string;
    tokens_seen: number;
    train_minutes: number | null;
    val_loss_fp32: number;
    val_loss_q8: number;
    dataset_tokens: number;
    file_mb: number;
  };
}

const SAMPLE_PROMPT = "Once upon a time";
const SVG = "http://www.w3.org/2000/svg";
const W = 760;
const H = 250;
const M = { l: 40, r: 12, t: 14, b: 40 };

function svg<K extends keyof SVGElementTagNameMap>(tag: K, attrs: Record<string, string | number> = {}, text?: string) {
  const e = document.createElementNS(SVG, tag);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, String(v));
  if (text !== undefined) e.textContent = text;
  return e;
}

interface Controls {
  slider: HTMLInputElement;
  play: HTMLButtonElement;
  at: HTMLElement;
  sample: HTMLElement;
}

export function renderTimeline(host: HTMLElement, t: Training, ui: Controls) {
  const maxStep = Math.max(t.meta.step, ...t.samples.map((s) => s.step));
  // Most of the learning happens in the first few hundred steps, so use a log-ish x axis.
  const xs = (step: number) => M.l + (Math.log10(1 + step / 10) / Math.log10(1 + maxStep / 10)) * (W - M.l - M.r);
  const random = Math.log(t.config.model.vocab_size);
  const losses = [...t.train_loss, ...t.val_loss].map(([, l]) => l);
  const yMax = Math.ceil(Math.max(random, ...losses) * 2) / 2;
  const yMin = Math.max(0, Math.floor(Math.min(...losses) * 2) / 2 - 0.5);
  const ys = (loss: number) => M.t + ((yMax - loss) / (yMax - yMin)) * (H - M.t - M.b);

  const root = svg("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "Training loss over time" });
  for (let y = Math.ceil(yMin); y <= yMax; y++) {
    root.append(svg("line", { class: "axis", x1: M.l, x2: W - M.r, y1: ys(y), y2: ys(y), "stroke-opacity": 0.5 }));
    root.append(svg("text", { x: M.l - 8, y: ys(y) + 4, "text-anchor": "end" }, String(y)));
  }
  for (const s of [0, 10, 100, 1000, 10000, 100000].filter((s) => s <= maxStep)) {
    root.append(svg("text", { x: xs(s), y: H - M.b + 18, "text-anchor": "middle" }, s.toLocaleString()));
  }
  root.append(svg("text", { x: (M.l + W - M.r) / 2, y: H - 4, "text-anchor": "middle" }, "training step (log scale)"));
  root.append(svg("text", { x: 12, y: M.t + 4, transform: `rotate(-90 12 ${M.t + 4})`, "text-anchor": "end" }, "loss"));
  root.append(svg("line", { class: "guess", x1: M.l, x2: W - M.r, y1: ys(random), y2: ys(random), "stroke-dasharray": "2 4" }));
  root.append(svg("text", { x: W - M.r, y: ys(random) - 6, "text-anchor": "end" }, "random guessing"));

  const pts = t.train_loss.map(([s, l]) => `${xs(s).toFixed(1)},${ys(l).toFixed(1)}`);
  root.append(svg("path", { class: "train", d: `M${pts.join("L")}` }));
  for (const [s, l] of t.val_loss) root.append(svg("circle", { class: "val", cx: xs(s), cy: ys(l), r: 3.2 }));

  const cursor = svg("line", { class: "cursor", y1: M.t, y2: H - M.b });
  root.append(cursor);
  const lossAt = (step: number) => {
    const v = t.val_loss.find(([s]) => s === step);
    if (v) return v[1];
    let best = t.train_loss[0];
    for (const p of t.train_loss) if (Math.abs(p[0] - step) < Math.abs(best[0] - step)) best = p;
    return step === 0 ? t.val_loss[0]?.[1] ?? random : best[1];
  };
  const marks = t.samples.map((s, i) => {
    const c = svg("circle", { class: "mark", cx: xs(s.step), cy: ys(lossAt(s.step)), r: 5.5 });
    c.append(svg("title", {}, `step ${s.step}`));
    c.addEventListener("click", () => select(i));
    root.append(c);
    return c;
  });
  host.replaceChildren(root);

  ui.slider.max = String(t.samples.length - 1);
  ui.slider.oninput = () => select(Number(ui.slider.value));

  let typing = 0;
  function select(i: number) {
    const s = t.samples[i];
    ui.slider.value = String(i);
    marks.forEach((m, j) => m.classList.toggle("on", j === i));
    cursor.setAttribute("x1", String(xs(s.step)));
    cursor.setAttribute("x2", String(xs(s.step)));
    ui.at.textContent =
      s.step === 0
        ? "step 0 · random numbers, hasn't read anything"
        : `step ${s.step.toLocaleString()} · read ${compact(s.tokens)} tokens · loss ${lossAt(s.step).toFixed(2)}`;
    typeOut(s.text);
  }

  function typeOut(text: string) {
    const me = ++typing;
    const body = text.startsWith(SAMPLE_PROMPT) ? text.slice(SAMPLE_PROMPT.length) : text;
    const head = text.length - body.length;
    const reduced = matchMedia("(prefers-reduced-motion: reduce)").matches;
    let n = reduced ? body.length : 0;
    const frame = () => {
      if (me !== typing) return;
      n = Math.min(body.length, n + Math.max(3, Math.ceil(body.length / 40)));
      ui.sample.replaceChildren(el("span", { class: "p" }, text.slice(0, head)), body.slice(0, n));
      if (n < body.length) requestAnimationFrame(frame);
    };
    frame();
  }

  let timer = 0;
  const stop = () => {
    clearInterval(timer);
    timer = 0;
    ui.play.textContent = "▶ Play";
  };
  ui.play.onclick = () => {
    if (timer) return stop();
    let i = Number(ui.slider.value);
    if (i >= t.samples.length - 1) i = -1;
    ui.play.textContent = "❚❚ Pause";
    const tick = () => {
      i++;
      select(i);
      if (i >= t.samples.length - 1) stop();
    };
    tick();
    timer = window.setInterval(tick, 1600);
  };
  ui.slider.addEventListener("pointerdown", stop);

  select(0);
}

export function renderHow(host: HTMLElement, t: Training) {
  const m = t.meta;
  const hours = (m.train_minutes ?? 0) / 60;
  const gpu = (m.gpu ?? "GPU").replace(/^NVIDIA GeForce /, "");
  const steps: [string, string][] = [
    ["TinyStories", "2.7M short stories, 2.2 GB of text"],
    ["Tokenizer", `${t.config.model.vocab_size.toLocaleString()} tokens, learned from scratch`],
    ["Token stream", `${compact(m.dataset_tokens)} tokens`],
    ["Training", `${hours >= 1 ? `${hours.toFixed(1)} h` : `${Math.round(m.train_minutes ?? 0)} min`} on an ${gpu}`],
    ["Export", `int8 weights, ${m.file_mb.toFixed(0)} MB`],
    ["Your browser", "TypeScript in a Web Worker"],
  ];
  host.replaceChildren(
    ...steps.map(([b, s]) => {
      const li = el("li");
      li.append(el("b", {}, b), s);
      return li;
    }),
  );
}
