"""
Step 1: learn a tokenizer from TinyStories, then turn the whole dataset
into one long stream of token ids.

    python prepare.py            # -> data/tokenizer.json, data/train.bin, data/valid.bin

Each story is encoded as [EOT] + tokens, so the model learns that
<|endoftext|> means "a new story starts here". At generation time we feed
[EOT] + prompt and stop when the model emits EOT again.
"""

import argparse
import os
import time
from multiprocessing import Pool

import numpy as np

from tokenizer import EOT, Tokenizer, chunk_counts, train

DATA = os.path.join(os.path.dirname(__file__), "..", "data")
TRAIN_TXT = os.path.join(DATA, "TinyStoriesV2-GPT4-train.txt")
VALID_TXT = os.path.join(DATA, "TinyStoriesV2-GPT4-valid.txt")

_tok: Tokenizer | None = None


def _init(path: str) -> None:
    global _tok
    _tok = Tokenizer.load(path)


def _encode_batch(stories: list[str]) -> np.ndarray:
    out = []
    for s in stories:
        out.append(_tok.eot)
        out.extend(_tok.encode(s))
    return np.array(out, dtype=np.uint16)


def stories_of(text: str) -> list[str]:
    return [s.strip() for s in text.split(EOT) if s.strip()]


def encode_file(txt: str, out: str, tok_path: str, workers: int) -> int:
    t0 = time.time()
    with open(txt, encoding="utf-8") as f:
        stories = stories_of(f.read())
    batches = [stories[i : i + 10_000] for i in range(0, len(stories), 10_000)]
    n = 0
    with Pool(workers, initializer=_init, initargs=(tok_path,)) as pool, open(out, "wb") as f:
        for i, arr in enumerate(pool.imap(_encode_batch, batches)):
            arr.tofile(f)
            n += len(arr)
            if i % 20 == 0:
                print(f"  {os.path.basename(out)}: {i + 1}/{len(batches)} batches, {n / 1e6:.1f}M tokens")
    print(f"{os.path.basename(out)}: {len(stories):,} stories -> {n:,} tokens in {time.time() - t0:.0f}s")
    return n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab", type=int, default=4096)
    ap.add_argument("--sample_mb", type=int, default=200, help="text used to learn merges")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    args = ap.parse_args()

    tok_path = os.path.join(DATA, "tokenizer.json")
    t0 = time.time()
    with open(TRAIN_TXT, encoding="utf-8") as f:
        sample = f.read(args.sample_mb * 1_000_000)
    counts = chunk_counts(sample)
    print(f"tokenizer: {len(counts):,} distinct chunks in {args.sample_mb}MB of text")
    merges = train(counts, args.vocab)
    tok = Tokenizer(merges)
    tok.save(tok_path)
    print(f"tokenizer: learned {len(merges):,} merges in {time.time() - t0:.0f}s")
    print("  longest tokens:", sorted((b for b in tok.bytes[256:-1]), key=len)[-8:])

    encode_file(VALID_TXT, os.path.join(DATA, "valid.bin"), tok_path, args.workers)
    encode_file(TRAIN_TXT, os.path.join(DATA, "train.bin"), tok_path, args.workers)


if __name__ == "__main__":
    main()
