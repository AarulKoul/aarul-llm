# AARUL

**Attention Architecture, Rebuilt Using a Laptop.**

A language model I wrote from scratch, trained from random numbers on a laptop GPU, running in
your browser on an inference engine I also wrote from scratch. There's no server, no API, and no
ML library on the page: just 15.7 million numbers I trained and 19 KB of JavaScript.

**[Try it: watch it think →](https://aarulkoul.github.io/aarul-llm/)**

The demo has two parts:

- **Watch it think.** Give it the start of a story. As it writes, each word shows what else the
  model considered (and how likely each option was) and which earlier words its 48 attention heads
  were looking at when it chose.
- **Watch it learn.** The same prompt, asked at 20 points during training: from random byte soup
  at step 0, to "the big, a big, and" at step 50, to complete little stories by the end.

```
cd web && npm install && npm run dev     # → http://127.0.0.1:4174
cd web && npm test                       # the engine must match PyTorch to float32 precision
```

This README assumes no prior knowledge of how language models work. If you already know, skip to
[the numbers](#the-numbers) or [how do you know it's right?](#how-do-you-know-its-right).

---

## 1. What a language model actually does

A language model does one thing: **given some text, it guesses what comes next.** It doesn't write a
story all at once. It writes one small piece (a _token_, usually a word or part of a word), adds that
piece to the text, and guesses again. Repeat 200 times and you have a story.

Each guess is a list of probabilities over every token it knows: "` dragon`: 31%, ` dog`: 12%,
` girl`: 9%, …". We pick one at random, weighted by those probabilities, which is why the same
prompt gives a different story each time. _Temperature_ reshapes the odds: near 0 it always takes
the top guess, and higher values give the unlikely ones more of a chance.

Where do the probabilities come from? From 15.7 million numbers (the _parameters_), arranged into
a fixed sequence of multiplications and additions called a **transformer**. At the start of
training those numbers are random, so the guesses are random. Training shows the model real text,
measures how wrong each guess was (the _loss_), and nudges every number slightly in the direction
that would have made the guess less wrong. After 230 million tokens of nudging, the numbers encode
grammar, characters, cause and effect, and how bedtime stories tend to end.

### What "loss" means here

The model knows 4,096 tokens. Guessing uniformly at random gives a loss of ln(4096) ≈ **8.32**,
which is where the [training chart](https://aarulkoul.github.io/aarul-llm/#learn) starts. A loss of _L_
means the model is, on average, about as uncertain as if it were choosing between e^L equally likely
options. AARUL ends training at a loss of **1.27**, as if each next token were a choice between
roughly **3.6** options instead of 4,096.

---

## 2. The pieces

Everything below is written from scratch. The only libraries are PyTorch (tensors and autograd,
for training) and NumPy. The browser side has no runtime dependencies at all.

### The data

[TinyStories](https://arxiv.org/abs/2305.07759) (Eldan & Li, 2023): 2.7 million short stories
written by GPT-4 using only words a young child would know. Microsoft researchers built it to show
that very small models can learn fluent English if the language is simple enough. It's 2.2 GB of
text, which makes it the right size for a laptop.

### The tokenizer: [`train/tokenizer.py`](train/tokenizer.py)

Models work with numbers, not text, so text is cut into tokens and each token gets an id. AARUL
uses **byte-pair encoding (BPE)**, trained from scratch:

1. Start with the 256 possible bytes as the vocabulary.
2. Count every adjacent pair of tokens across the text, merge the most frequent pair (` ` + `t` →
   ` t`) into a new token, and repeat.
3. Stop after 3,839 merges. With the 256 bytes and one "end of story" token, that's 4,096 tokens.

The longest tokens it learned are words like ` compassionate`, ` uncomfortable` and
` enthusiastic`. Training takes 18 seconds, and encoding the whole dataset into **552 million
tokens** takes about a minute on 14 processes. The browser runs an exact port of it
([`tokenizer.ts`](web/src/engine/tokenizer.ts)), checked token for token against the Python one.

### The model: [`train/model.py`](train/model.py)

A decoder-only transformer with the same design as Meta's Llama models, scaled way down:

|                    |                                            |
| ------------------ | ------------------------------------------ |
| Parameters         | 15.74 million                              |
| Layers             | 8                                          |
| Attention heads    | 6 per layer (48 total), 64 dimensions each |
| Model width        | 384                                        |
| Feed-forward width | 1,024 (SwiGLU)                             |
| Context            | 512 tokens                                 |
| Position encoding  | Rotary (RoPE)                              |
| Normalization      | RMSNorm, pre-norm                          |
| Output layer       | Tied to the input embedding                |

Each layer does two things. **Attention** lets every token look back at the earlier tokens and pull
in information from the relevant ones; that's how "she" knows it refers to Lily. The **feed-forward
network** then transforms each token's representation independently, which is where most of the
model's "knowledge" lives. The "where it looked" panel in the demo shows the attention weights live.

### Training: [`train/train.py`](train/train.py)

|           |                                                                       |
| --------- | --------------------------------------------------------------------- |
| GPU       | NVIDIA RTX 3070 Ti **Laptop** (8 GB)                                  |
| Steps     | 7,000 × 32,768 tokens = 229 million tokens (0.4 passes over the data) |
| Optimizer | AdamW (β = 0.9, 0.95; weight decay 0.1), gradient clipping at 1.0     |
| Schedule  | 500-step linear warmup to 1e-3, cosine decay to 1e-4                  |
| Precision | bfloat16 autocast                                                     |
| Wall time | 2.6 hours (154 minutes)                                               |
| Cost      | $0                                                                    |

A laptop is not a training rig. The GPU benchmarks at 55,000 tokens/s when it's cold, but within
minutes it reaches its 87 °C target temperature and the driver starts cutting clocks to stay there,
sometimes down to 210 MHz. It settles at about 24,800 tokens/s, less than half its cold speed. I
measured the throttled speed, then sized the run to fit an afternoon. The run writes a resumable checkpoint every 500
steps, so a crash or a closed lid costs minutes, not the whole run.

Every so often during training, the model writes a story from the same prompt with the same random
seed, so the only thing that changes between samples is what it has learned. Those samples are the
"watch it learn" timeline.

### The browser engine: [`web/src/engine/`](web/src/engine/)

To run in a browser, the model is exported into a small binary format
([`export.py`](train/export.py)) and executed by a forward pass written in plain TypeScript
([`model.ts`](web/src/engine/model.ts)):

- **Int8 weights.** Each weight matrix is stored as 8-bit integers plus one float scale per row:
  4× smaller than float32 (15.9 MB), and validation loss changes by only 0.0001 (1.2788 → 1.2789).
- **KV cache.** When a new token arrives, only that token's keys and values are computed. Earlier
  ones are cached, so each step costs the same no matter how long the story gets.
- **Web Worker.** The model runs off the main thread, so the page stays smooth while it computes.
- **Explainability for free.** Because I wrote the attention loop myself, every step can hand the
  UI its attention weights for all 48 heads, and the full probability distribution behind each
  choice.

---

## 3. How do you know it's right?

The engine is a second, independent implementation of the model in a different language, so the
risk is that the two quietly disagree: a transposed matrix, a RoPE off by one position, a different
tokenization of `don't`. Any of those would still produce plausible-looking text, just worse text,
and nobody would notice.

So `python export.py --fixtures` builds a tiny random model, runs it through PyTorch, and saves the
results. [`web/test/engine.test.ts`](web/test/engine.test.ts) runs the same inputs through the
TypeScript engine and checks:

| Test                                                                     | Tolerance | Actual     |
| ------------------------------------------------------------------------ | --------- | ---------- |
| Tokenizer, including emoji, accents, digits, tabs and story separators   | exact     | exact      |
| Logits at the first and last positions                                   | 1e-4      | **1.2e-7** |
| Attention weights, every layer and head                                  | 1e-5      | ✓          |
| Int8 weights after dequantization                                        | 1e-4      | **1.2e-7** |
| KV-cached stepping vs. a fresh run                                       | exact     | exact      |
| Sampling: greedy = argmax, top-k never leaves the top k, seeds reproduce | exact     | ✓          |

1.2e-7 is float32 rounding error: the two implementations agree to the precision of the numbers
themselves. The tests run in CI before every deploy.

---

## The numbers

|                             |                                                       |
| --------------------------- | ----------------------------------------------------- |
| Validation loss             | 8.40 at step 0 → **1.274** at step 7,000              |
| Effective choices per token | 4,096 → about 3.6                                     |
| Training                    | 2.6 hours, 229M tokens, on an RTX 3070 Ti Laptop GPU  |
| Sustained throughput        | 24,800 tokens/s (55,000 before thermal throttling)    |
| Model download              | 15.9 MB (int8)                                        |
| Int8 cost                   | +0.0001 validation loss                               |
| Speed in the browser        | about 40–50 tokens/s, one CPU thread, no GPU          |
| JavaScript shipped          | 19 KB minified (engine + UI), zero dependencies       |
| Cloud bill                  | $0                                                    |

### Watching it learn

The same prompt ("Once upon a time") and the same random seed, at four points in training:

> **Step 0** (random numbers): Once upon a timeJohn reading distant distant belt elephantaleoth helps
> helps helps helpsaghet craw lab rope ropeead bag…
>
> **Step 50** (1.6M tokens read): Once upon a time, the big, a big, and, you, mom. He said, play with
> the park. They with the cat took the tree.
>
> **Step 300** (10M tokens read): Once upon a time, there was a little boy named Tim. Tim was a brave
> boy who liked to play with his toy gun. One day, Tim's mom told him that he was walking in a big
> world.
>
> **Step 7,000** (229M tokens read): Once upon a time, there was a little boy named Tim. Tim was a
> normal boy who liked to play with his toys. One day, Tim found a key in his room. He did not know
> what it was for, but he thought it was pretty. Tim showed the key to his mom. "Mom, look what I
> found!"

Words arrive first, then grammar, then characters and plot. By step 300 it already writes
grammatical sentences. The remaining 6,700 steps go into making them make sense together.

### A story from the final model

> **Once upon a time, there was a little dragon who** lived in a distant castle. He had a white hat
> and a big smile. The dragon loved to play with his friends in the castle.
>
> One day, the dragon found some powder on the ground. He thought it would be fun to play with it.
> So, he used the powder to mix colors, making a beautiful rainbow. The people in the castle came to
> see the rainbow and they all laughed.

## Limits

- It has only ever read simple children's stories, so that's all it can write. When I prompted
  it with "The stock market", it continued: "The stock market was a very big store."
- 15.7M parameters is less than 1/10,000th the size of today's frontier models. It's fluent, but
  its plots wander, and characters sometimes change names halfway through ("the little girl,
  named Tim").
- It runs on the CPU in a single thread. A WebGPU path would be faster, but the CPU version runs
  everywhere and is the easiest to read.

## Repo layout

```
data/fetch.sh            download TinyStories (2.2 GB)
train/tokenizer.py       byte-level BPE, trained from scratch
train/prepare.py         learn the tokenizer, encode the dataset to uint16 token streams
train/model.py           the transformer
train/train.py           training loop: AdamW, cosine schedule, bf16, resumable checkpoints
train/export.py          int8 export for the browser, plus test fixtures
train/card.py            the social preview image, drawn from the real training log
web/src/engine/          tokenizer, weights loader, forward pass, sampler, Web Worker
web/src/ui/              the page: story view, inspector, training timeline
web/test/                parity tests against PyTorch
```

## Reproduce it

```bash
bash data/fetch.sh                                  # TinyStories, 2.2 GB
cd train
python -m venv .venv
.venv/Scripts/pip install torch --index-url https://download.pytorch.org/whl/cu130
.venv/Scripts/pip install numpy
.venv/Scripts/python prepare.py                     # tokenizer + 552M-token dataset, ~2 min
.venv/Scripts/python train.py --bench               # measure your GPU first
.venv/Scripts/python train.py --steps 7000          # add --resume to continue after a stop
.venv/Scripts/python export.py runs/aarul/final.pt  # → web/public/model/
```

## Credits

The data is TinyStories by Ronen Eldan and Yuanzhi Li. The architecture follows Llama (Touvron et
al., 2023). Andrej Karpathy's nanoGPT and llama2.c showed that small models trained this way can be
fun. Everything in this repo is my own implementation.

MIT licensed.
