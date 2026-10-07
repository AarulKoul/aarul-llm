import type { FromWorker, ToWorker, TokenEvent } from "../engine/protocol.ts";
import type { Config } from "../engine/weights.ts";
import { compact, el, pct, showToken } from "./format.ts";
import { type Training, renderHow, renderTimeline } from "./timeline.ts";

const $ = <T extends HTMLElement = HTMLElement>(id: string) => document.getElementById(id) as T;

const PRESETS = [
  "Once upon a time, there was a little dragon who",
  "Tom found a strange box in the garden. When he opened it,",
  "Lily wanted to fly like a bird, so she",
  "The sad robot",
  "Mom said no more cookies, but Ben",
];

const worker = new Worker(new URL("../engine/worker.js", import.meta.url), { type: "module" });
const send = (msg: ToWorker) => worker.postMessage(msg);

let vocab: string[] = [];
let cfg: Config | null = null;
let tokens: TokenEvent[] = []; // tokens[pos - 1]
let spans: HTMLElement[] = []; // spans[pos]; spans[0] is the start-of-story marker
let caret: HTMLElement | null = null;
let running = false;
let delayMs = 70;
let hoverPos: number | null = null;
let pinPos: number | null = null;
let selLayer = -1; // -1: average over layers
let selHead = -1; // -1: average over heads
let lit: HTMLElement[] = [];
let msWindow: number[] = [];
let run = 0;
let current: { prompt: string; temperature: number; topK: number; seed: number } | null = null;

const story = $("story");
const go = $<HTMLButtonElement>("go");
const promptBox = $<HTMLTextAreaElement>("prompt");

// ---------- controls ----------

for (const p of PRESETS) {
  const b = el("button", { class: "chip", type: "button" }, p);
  b.onclick = () => {
    promptBox.value = p;
    start();
  };
  $("presets").append(b);
}
const temp = $<HTMLInputElement>("temp");
const topk = $<HTMLInputElement>("topk");
temp.oninput = () => ($("tempOut").textContent = Number(temp.value).toFixed(2));
topk.oninput = () => ($("topkOut").textContent = topk.value);
for (const b of document.querySelectorAll<HTMLButtonElement>(".seg-btn")) {
  b.onclick = () => {
    document.querySelectorAll(".seg-btn").forEach((x) => x.classList.toggle("on", x === b));
    delayMs = Number(b.dataset.speed);
  };
}
go.onclick = () => (running ? send({ type: "stop" }) : start());
promptBox.onkeydown = (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    if (!go.disabled && !running) start();
  }
};

function setRunning(on: boolean) {
  running = on;
  go.textContent = on ? "Stop" : "Generate";
  go.classList.toggle("stop", on);
  caret?.remove();
  caret = null;
  if (on) {
    caret = el("span", { class: "caret" });
    story.append(caret);
  }
}

function start(seed = (Math.random() * 2 ** 32) >>> 0) {
  if (go.disabled) return;
  if (running) send({ type: "stop" });
  $("share").hidden = true;
  story.replaceChildren();
  tokens = [];
  const bos = el("span", { class: "bos", title: "start of story" }, "◆");
  spans = [bos];
  story.append(bos);
  hoverPos = pinPos = null;
  msWindow = [];
  setRunning(true);
  current = {
    // A trailing space would become its own token, which the model almost never
    // saw in training (spaces attach to the word after them), so drop it.
    prompt: promptBox.value.replace(/[ \t]+$/, ""),
    temperature: Number(temp.value),
    topK: Number(topk.value),
    seed,
  };
  send({ type: "generate", run: ++run, maxTokens: 400, delayMs, ...current });
}

// A story is fully determined by its prompt, settings and seed, so a link can carry it.
$("share").onclick = async () => {
  if (!current) return;
  const q = new URLSearchParams({
    p: current.prompt,
    t: String(current.temperature),
    k: String(current.topK),
    s: String(current.seed),
  });
  history.replaceState(null, "", `#${q}`);
  try {
    await navigator.clipboard.writeText(location.href);
    $("share").textContent = "Link copied ✓";
  } catch {
    $("share").textContent = "Link is in the address bar";
  }
  setTimeout(() => ($("share").textContent = "Share this story"), 2500);
};

function startFromLink(): boolean {
  const q = new URLSearchParams(location.hash.slice(1));
  const p = q.get("p");
  if (p === null) return false;
  promptBox.value = p;
  temp.value = q.get("t") ?? temp.value;
  topk.value = q.get("k") ?? topk.value;
  temp.dispatchEvent(new Event("input"));
  topk.dispatchEvent(new Event("input"));
  start(Number(q.get("s")) >>> 0);
  return true;
}

// ---------- story ----------

function addToken(ev: TokenEvent) {
  tokens[ev.pos - 1] = ev;
  const s = el("span", { class: `tok ${ev.prompt ? "prompt" : "gen"} fresh` }, vocab[ev.id]);
  s.dataset.pos = String(ev.pos);
  if (!ev.prompt) s.style.setProperty("--surprise", String(surprise(ev.p)));
  spans[ev.pos] = s;
  if (caret) story.insertBefore(s, caret);
  else story.append(s);
  if (!ev.prompt) {
    msWindow.push(ev.ms);
    if (msWindow.length > 30) msWindow.shift();
    const avg = msWindow.reduce((a, b) => a + b, 0) / msWindow.length;
    $("speed").textContent = `your browser: ${Math.round(1000 / avg)} tokens/s`;
  }
  const nearBottom = story.scrollHeight - story.scrollTop - story.clientHeight < 80;
  if (nearBottom) story.scrollTop = story.scrollHeight;
  refocus();
}

/**
 * Background strength for a token the model chose. Choices it gave 1-in-8 odds
 * or better stay plain; below that, the rarer the choice, the stronger the tint.
 */
function surprise(p: number): number {
  const bits = -Math.log2(Math.max(p, 1e-6));
  return Math.max(0, Math.min(1, (bits - 3) / 5)) * 0.55;
}

story.addEventListener("mouseover", (e) => {
  const t = (e.target as HTMLElement).closest<HTMLElement>(".tok");
  if (t) {
    hoverPos = Number(t.dataset.pos);
    refocus();
  }
});
story.addEventListener("mouseleave", () => {
  hoverPos = null;
  refocus();
});
story.addEventListener("click", (e) => {
  const t = (e.target as HTMLElement).closest<HTMLElement>(".tok");
  const pos = t ? Number(t.dataset.pos) : null;
  pinPos = pos === pinPos ? null : pos;
  refocus();
});

function focusPos(): number | null {
  const p = pinPos ?? hoverPos ?? (tokens.length ? tokens.length : null);
  return p !== null && tokens[p - 1] ? p : null;
}

function refocus() {
  const pos = focusPos();
  for (const s of story.querySelectorAll(".tok.focus")) s.classList.remove("focus");
  for (const s of lit) {
    s.classList.remove("lit");
    s.style.removeProperty("--w");
  }
  lit = [];
  if (pos === null) {
    $("inspEmpty").hidden = false;
    $("inspBody").hidden = true;
    return;
  }
  const ev = tokens[pos - 1];
  spans[pos]?.classList.add("focus");
  renderInspector(ev);
  shadeAttention(ev);
}

// ---------- inspector ----------

function renderInspector(ev: TokenEvent) {
  $("inspEmpty").hidden = true;
  $("inspBody").hidden = false;
  $("inspLabel").textContent = ev.prompt ? "you typed" : pinPos === ev.pos ? "it chose (pinned)" : "it chose";
  $("inspTok").textContent = showToken(vocab[ev.id]);
  const rank = ev.top.findIndex((c) => c.id === ev.id);
  $("inspMeta").textContent =
    (ev.prompt ? `it gave this ${pct(ev.p)}` : `${pct(ev.p)} likely`) +
    (rank >= 0 ? ` · its #${rank + 1} guess` : " · not in its top 10") +
    ` · token #${ev.id} · ${ev.ms.toFixed(1)} ms`;

  const rows = ev.top.map((c) => ({ ...c, chosen: c.id === ev.id }));
  if (rank < 0) rows.push({ id: ev.id, p: ev.p, chosen: true });
  const max = rows[0].p;
  $("cands").replaceChildren(
    ...rows.map((c) => {
      const li = el("li", { class: `cand${c.chosen ? " chosen" : ""}` });
      const bar = el("span", { class: "bar" });
      const fill = el("i");
      fill.style.width = `${(c.p / max) * 100}%`;
      bar.append(fill);
      li.append(el("span", { class: "t" }, showToken(vocab[c.id])), bar, el("span", { class: "pct" }, pct(c.p)));
      return li;
    }),
  );
  renderHeads(ev);
}

/** Attention weight of head (l, h) on position s, for the step that produced `ev`. */
const attnAt = (ev: TokenEvent, l: number, h: number, s: number) => ev.attn[(l * cfg!.n_head + h) * ev.pos + s];

function renderHeads(ev: TokenEvent) {
  const { n_layer, n_head } = cfg!;
  const grid = $("heads");
  grid.style.setProperty("--nh", String(n_head));
  const cells: HTMLElement[] = [el("span", { class: "lbl" })];
  for (let h = 0; h < n_head; h++) cells.push(el("span", { class: "lbl" }, `h${h + 1}`));
  for (let l = 0; l < n_layer; l++) {
    const lbl = el("button", { class: "lbl seg-btn", type: "button", title: `layer ${l + 1}, all heads` }, `L${l + 1}`);
    if (selLayer === l && selHead === -1) lbl.classList.add("on");
    lbl.onclick = () => selectHead(l, -1);
    cells.push(lbl);
    for (let h = 0; h < n_head; h++) {
      // "Focus": the most attention this head paid to any single real word.
      let f = 0;
      for (let s = 1; s < ev.pos; s++) f = Math.max(f, attnAt(ev, l, h, s));
      const c = el("button", { class: "head-cell", type: "button", title: `layer ${l + 1}, head ${h + 1}` });
      c.style.setProperty("--f", String(0.06 + 0.9 * f));
      if (selLayer === l && selHead === h) c.classList.add("sel");
      c.onclick = () => selectHead(l, h);
      cells.push(c);
    }
  }
  grid.replaceChildren(...cells);
  $("attnScope").textContent =
    selLayer < 0 ? "all layers, all heads" : selHead < 0 ? `layer ${selLayer + 1}, all heads` : `layer ${selLayer + 1}, head ${selHead + 1}`;
}

function selectHead(l: number, h: number) {
  if (selLayer === l && selHead === h) selLayer = selHead = -1;
  else [selLayer, selHead] = [l, h];
  refocus();
}

function shadeAttention(ev: TokenEvent) {
  const { n_layer, n_head } = cfg!;
  const layers = selLayer < 0 ? [...Array(n_layer).keys()] : [selLayer];
  const heads = selHead < 0 ? [...Array(n_head).keys()] : [selHead];
  const w = new Float32Array(ev.pos);
  for (const l of layers) for (const h of heads) for (let s = 0; s < ev.pos; s++) w[s] += attnAt(ev, l, h, s);
  // Scale to the strongest real word, so the start marker (which often soaks up
  // most of the attention) doesn't wash everything else out.
  let max = 1e-9;
  for (let s = 1; s < ev.pos; s++) max = Math.max(max, w[s]);
  if (ev.pos === 1) max = w[0];
  for (let s = 0; s < ev.pos; s++) {
    const a = Math.min(1, w[s] / max);
    if (a < 0.15 || !spans[s]) continue;
    spans[s].classList.add("lit");
    spans[s].style.setProperty("--w", (0.85 * a ** 1.6).toFixed(3));
    lit.push(spans[s]);
  }
}

// ---------- worker ----------

worker.onmessage = (e: MessageEvent<FromWorker>) => {
  const m = e.data;
  switch (m.type) {
    case "progress": {
      const f = m.total ? m.loaded / m.total : 0;
      $("loadBar").style.width = `${f * 100}%`;
      $("loadText").textContent = `Downloading the model… ${(m.loaded / 1e6).toFixed(1)}${m.total ? ` / ${(m.total / 1e6).toFixed(1)}` : ""} MB`;
      break;
    }
    case "ready":
      vocab = m.vocab;
      cfg = m.config;
      go.disabled = false;
      go.textContent = "Generate";
      $("loading").remove();
      if (!startFromLink()) {
        story.append(el("span", { class: "muted" }, "Model loaded. Press Generate, or pick a prompt above."));
      }
      break;
    case "token":
      if (m.run === run) addToken(m);
      break;
    case "done":
      if (m.run !== run) break;
      setRunning(false);
      $("share").hidden = tokens.length === 0;
      break;
    case "error":
      setRunning(false);
      $("loading")?.remove();
      story.append(el("div", { class: "muted" }, `Something went wrong: ${m.message}`));
      break;
  }
};

// Resolve against the page: the worker would otherwise resolve relative to its own script.
send({
  type: "load",
  modelUrl: new URL("model/aarul.bin", location.href).href,
  tokenizerUrl: new URL("model/tokenizer.json", location.href).href,
});

// ---------- training story + stats ----------

fetch("model/training.json")
  .then((r) => r.json())
  .then((t: Training) => {
    renderStats(t);
    renderTimeline($("chart"), t, {
      slider: $<HTMLInputElement>("tlSlider"),
      play: $<HTMLButtonElement>("tlPlay"),
      at: $("tlAt"),
      sample: $("tlSample"),
    });
    renderHow($("pipeline"), t);
  })
  .catch(() => {
    $("learn").hidden = true;
  });

function renderStats(t: Training) {
  const m = t.meta;
  const c = t.config.model;
  const hours = (m.train_minutes ?? 0) / 60;
  const stats: [string, string][] = [
    [compact(m.params), "parameters, all trained from random"],
    [`${c.n_layer} × ${c.n_head}`, "layers × attention heads"],
    [hours >= 1 ? `${hours.toFixed(1)} h` : `${Math.round(m.train_minutes ?? 0)} min`, `training on a laptop ${shortGpu(m.gpu)}`],
    [compact(m.tokens_seen), "tokens read during training"],
    [`${m.file_mb.toFixed(0)} MB`, "download, then runs offline"],
    ["$0", "cloud bill"],
  ];
  $("stats").replaceChildren(
    ...stats.map(([b, s]) => {
      const d = el("div", { class: "stat" });
      d.append(el("b", {}, b), el("span", {}, s));
      return d;
    }),
  );
  $("footMeta").textContent = `${compact(m.params)} parameters · ${c.n_layer} layers · step ${m.step.toLocaleString()} · val loss ${m.val_loss_fp32}`;
}

function shortGpu(gpu: string | undefined): string {
  return (gpu ?? "GPU").replace(/^NVIDIA GeForce /, "").replace(/ Laptop GPU$/, "");
}
