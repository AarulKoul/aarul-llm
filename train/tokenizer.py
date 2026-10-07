"""
A byte-level BPE tokenizer, trained from scratch.

Text is first split into chunks (words, runs of punctuation, runs of
whitespace) by PATTERN, then each chunk's UTF-8 bytes are merged pairwise
according to the learned merge list. Token ids 0..255 are raw bytes, ids
256.. are merges in the order they were learned, and the last id is the
special <|endoftext|> token that separates stories.

The browser tokenizer (web/src/engine/tokenizer.ts) implements the same
algorithm, so PATTERN deliberately uses explicit character classes that
behave identically in Python's `re` and JavaScript's RegExp.
"""

import heapq
import json
import re
from collections import Counter, defaultdict

WS = r" \t\n\r\f\v"
PATTERN = (
    r"'(?:s|t|re|ve|m|ll|d)"  # contractions: it's, don't, they're ...
    r"| ?[A-Za-z]+"  # words, with their leading space
    r"| ?[0-9]{1,3}"  # numbers, at most three digits per chunk
    rf"| ?[^{WS}A-Za-z0-9]+"  # punctuation and anything else
    rf"|[{WS}]+(?![^{WS}])"  # whitespace, except the last char before a word
    rf"|[{WS}]+"
)
EOT = "<|endoftext|>"

_split = re.compile(PATTERN).findall


def chunk_counts(text: str) -> Counter:
    counts = Counter()
    for story in text.split(EOT):
        counts.update(_split(story))
    return counts


def train(counts: Counter, vocab_size: int) -> list[tuple[int, int]]:
    """Learn vocab_size - 257 merges (256 bytes + 1 special token are fixed)."""
    words = [list(chunk.encode("utf-8")) for chunk in counts]
    freqs = list(counts.values())

    pair_count: dict[tuple[int, int], int] = defaultdict(int)
    pair_words: dict[tuple[int, int], set[int]] = defaultdict(set)
    for w, (ids, f) in enumerate(zip(words, freqs)):
        for pair in zip(ids, ids[1:]):
            pair_count[pair] += f
            pair_words[pair].add(w)

    # Max-heap of (count, pair). Entries go stale as counts change; a popped
    # entry is only trusted if it still matches pair_count.
    heap = [(-c, p) for p, c in pair_count.items()]
    heapq.heapify(heap)

    merges = []
    n_merges = vocab_size - 257
    while len(merges) < n_merges and heap:
        neg, pair = heapq.heappop(heap)
        if pair_count.get(pair, 0) != -neg or -neg <= 0:
            continue
        new_id = 256 + len(merges)
        merges.append(pair)

        touched = set()
        for w in pair_words.pop(pair):
            ids, f = words[w], freqs[w]
            merged = _merge(ids, pair, new_id)
            if len(merged) == len(ids):
                continue  # stale index: this word no longer contains the pair
            for p in zip(ids, ids[1:]):
                pair_count[p] -= f
                touched.add(p)
            for p in zip(merged, merged[1:]):
                pair_count[p] += f
                pair_words[p].add(w)
                touched.add(p)
            words[w] = merged
        pair_count.pop(pair, None)
        for p in touched:
            c = pair_count.get(p, 0)
            if c > 0:
                heapq.heappush(heap, (-c, p))
            else:
                pair_count.pop(p, None)
    return merges


def _merge(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    out, i, a, b = [], 0, pair[0], pair[1]
    n = len(ids)
    while i < n:
        if i + 1 < n and ids[i] == a and ids[i + 1] == b:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


class Tokenizer:
    def __init__(self, merges: list[tuple[int, int]]):
        self.merges = [tuple(m) for m in merges]
        self.ranks = {pair: i for i, pair in enumerate(self.merges)}
        self.eot = 256 + len(self.merges)
        self.vocab_size = self.eot + 1
        self.bytes = [bytes([i]) for i in range(256)]
        for a, b in self.merges:
            self.bytes.append(self.bytes[a] + self.bytes[b])
        self.bytes.append(EOT.encode("utf-8"))
        self._cache: dict[str, list[int]] = {}

    @classmethod
    def load(cls, path: str) -> "Tokenizer":
        with open(path, encoding="utf-8") as f:
            return cls(json.load(f)["merges"])

    def save(self, path: str) -> None:
        data = {
            "pattern": PATTERN,
            "eot": EOT,
            "vocab_size": self.vocab_size,
            "merges": [list(m) for m in self.merges],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, separators=(",", ":"))

    def _encode_chunk(self, chunk: str) -> list[int]:
        cached = self._cache.get(chunk)
        if cached is not None:
            return cached
        ids = list(chunk.encode("utf-8"))
        ranks = self.ranks
        while len(ids) > 1:
            best, best_rank = None, None
            for pair in zip(ids, ids[1:]):
                r = ranks.get(pair)
                if r is not None and (best_rank is None or r < best_rank):
                    best, best_rank = pair, r
            if best is None:
                break
            ids = _merge(ids, best, 256 + best_rank)
        if len(self._cache) < 500_000:
            self._cache[chunk] = ids
        return ids

    def encode(self, text: str) -> list[int]:
        """Encode text; literal <|endoftext|> markers become the EOT token."""
        out = []
        for i, part in enumerate(text.split(EOT)):
            if i > 0:
                out.append(self.eot)
            for chunk in _split(part):
                out.extend(self._encode_chunk(chunk))
        return out

    def decode(self, ids: list[int]) -> str:
        return b"".join(self.bytes[i] for i in ids).decode("utf-8", errors="replace")
