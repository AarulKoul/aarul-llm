"""
Step 3: package a trained model for the browser.

    python export.py runs/aarul/final.pt      # -> web/public/model/{aarul.bin, tokenizer.json, training.json}
    python export.py --fixtures               # -> web/test/fixtures/ (parity test data, tiny random model)

aarul.bin layout (little-endian):
    "ARUL"  u32 version  u32 header_len  header JSON  zero padding to 4 bytes  tensor data
The header lists every tensor's name, shape, dtype and byte offset. Matrices
are stored either as float16 ("f16") or as int8 with one float32 scale per
row ("q8": w[r, c] ~= q[r, c] * scale[r]); norm weights are always float32.
"""

import argparse
import json
import os
import struct

import numpy as np
import torch

from model import AARUL, Config
from tokenizer import Tokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
DATA = os.path.join(ROOT, "data")
WEB_MODEL = os.path.join(ROOT, "web", "public", "model")
FIXTURES = os.path.join(ROOT, "web", "test", "fixtures")


def tensors_in_order(model: AARUL):
    """(name, tensor) in the order the browser engine expects."""
    yield "tok_emb", model.tok_emb.weight
    for i, b in enumerate(model.blocks):
        p = f"blocks.{i}."
        yield p + "attn_norm", b.attn_norm.weight
        yield p + "wq", b.attn.wq.weight
        yield p + "wk", b.attn.wk.weight
        yield p + "wv", b.attn.wv.weight
        yield p + "wo", b.attn.wo.weight
        yield p + "mlp_norm", b.mlp_norm.weight
        yield p + "w1", b.mlp.w1.weight
        yield p + "w3", b.mlp.w3.weight
        yield p + "w2", b.mlp.w2.weight
    yield "norm", model.norm.weight


def quantize_q8(w: np.ndarray):
    scale = np.abs(w).max(axis=1) / 127.0
    scale[scale == 0] = 1.0
    q = np.clip(np.round(w / scale[:, None]), -127, 127).astype(np.int8)
    return q, scale.astype(np.float32)


def write_bin(model: AARUL, path: str, dtype: str, meta: dict) -> int:
    entries, blobs, offset = [], [], 0

    def add(arr: np.ndarray) -> int:
        nonlocal offset
        start = offset
        b = arr.tobytes()
        pad = (-len(b)) % 4
        blobs.append(b + b"\0" * pad)
        offset += len(b) + pad
        return start

    for name, t in tensors_in_order(model):
        w = t.detach().float().cpu().numpy()
        e = {"name": name, "shape": list(w.shape)}
        if w.ndim == 1 or dtype == "f32":
            e["dtype"], e["offset"] = "f32", add(w.astype("<f4"))
        elif dtype == "f16":
            e["dtype"], e["offset"] = "f16", add(w.astype("<f2"))
        else:
            q, scale = quantize_q8(w)
            e["dtype"], e["offset"] = "q8", add(q)
            e["scale_offset"] = add(scale.astype("<f4"))
        entries.append(e)

    header = json.dumps({"config": model.cfg.to_dict(), "tensors": entries, "meta": meta}).encode()
    header += b" " * ((-(12 + len(header))) % 4)
    with open(path, "wb") as f:
        f.write(b"ARUL" + struct.pack("<II", 1, len(header)) + header)
        for b in blobs:
            f.write(b)
    return os.path.getsize(path)


def dequantized_copy(model: AARUL) -> AARUL:
    """The model exactly as the browser will see it after loading q8 weights."""
    m = AARUL(model.cfg)
    m.load_state_dict(model.state_dict())
    with torch.no_grad():
        for _, t in tensors_in_order(m):
            if t.dim() == 2:
                q, scale = quantize_q8(t.float().numpy())
                t.copy_(torch.from_numpy(q.astype(np.float32) * scale[:, None]))
    return m


@torch.no_grad()
def val_loss(model: AARUL, n_batches=20, batch=16) -> float:
    data = np.memmap(os.path.join(DATA, "valid.bin"), dtype=np.uint16, mode="r")
    rng = np.random.default_rng(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()
    ctx, losses = model.cfg.ctx, []
    for _ in range(n_batches):
        ix = rng.integers(0, len(data) - ctx - 1, batch)
        t = torch.from_numpy(np.stack([data[i : i + ctx + 1] for i in ix]).astype(np.int64)).to(device)
        losses.append(model(t[:, :-1], t[:, 1:])[1].item())
    model.cpu()
    return sum(losses) / len(losses)


def training_summary(run_dir: str) -> dict:
    """Condense log.jsonl into what the web page shows: loss curve + samples."""
    train, val, samples, elapsed = [], {}, {}, 0.0
    with open(os.path.join(run_dir, "log.jsonl"), encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            elapsed = max(elapsed, r.get("elapsed", 0))
            if "loss" in r:
                train.append([r["step"], r["loss"]])
            elif "val_loss" in r:
                val[r["step"]] = [r["step"], r["val_loss"]]  # a resumed run may repeat a step
            elif "sample" in r:
                samples[r["step"]] = {"step": r["step"], "tokens": r["tokens"], "text": r["sample"]}
    # Keep the curve light: ~400 points, each the mean of its bucket.
    stride = max(1, len(train) // 400)
    curve = [
        [train[i][0], round(sum(l for _, l in train[i : i + stride]) / len(train[i : i + stride]), 4)]
        for i in range(0, len(train), stride)
    ]
    with open(os.path.join(run_dir, "config.json")) as f:
        config = json.load(f)
    return {"config": config, "train_loss": curve, "val_loss": [val[k] for k in sorted(val)],
            "samples": [samples[k] for k in sorted(samples)], "elapsed": elapsed}


def export_model(ckpt_path: str, dtype: str) -> None:
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = AARUL(Config(**ck["config"]))
    model.load_state_dict(ck["model"])
    run_dir = os.path.dirname(os.path.abspath(ckpt_path))
    summary = training_summary(run_dir)
    os.makedirs(WEB_MODEL, exist_ok=True)

    loss_fp32 = val_loss(model)
    loss_q8 = val_loss(dequantized_copy(model)) if dtype == "q8" else loss_fp32
    last = summary["val_loss"][-1] if summary["val_loss"] else None
    meta = {
        "step": ck["step"],
        "params": model.n_params(),
        "gpu": summary["config"].get("gpu"),
        "tokens_seen": ck["step"] * summary["config"]["train"]["batch"]
        * summary["config"]["train"]["accum"] * model.cfg.ctx,
        "train_minutes": round(ck.get("elapsed", summary.pop("elapsed")) / 60, 1),
        "val_loss": last[1] if last else None,
        "val_loss_fp32": round(loss_fp32, 4),
        "val_loss_q8": round(loss_q8, 4),
        "dtype": dtype,
        "dataset_tokens": os.path.getsize(os.path.join(DATA, "train.bin")) // 2,
    }
    size = write_bin(model, os.path.join(WEB_MODEL, "aarul.bin"), dtype, meta)
    summary["meta"] = {**meta, "file_mb": round(size / 1e6, 1)}
    with open(os.path.join(WEB_MODEL, "training.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, separators=(",", ":"))
    with open(os.path.join(DATA, "tokenizer.json"), encoding="utf-8") as src, \
         open(os.path.join(WEB_MODEL, "tokenizer.json"), "w", encoding="utf-8") as dst:
        dst.write(src.read())
    print(json.dumps(meta, indent=2))
    print(f"aarul.bin: {size / 1e6:.1f} MB ({dtype})")


@torch.no_grad()
def export_fixtures() -> None:
    """A tiny random model plus reference outputs, for the browser engine's tests."""
    os.makedirs(FIXTURES, exist_ok=True)
    tok = Tokenizer.load(os.path.join(DATA, "tokenizer.json"))
    torch.manual_seed(0)
    cfg = Config(vocab_size=tok.vocab_size, ctx=64, d_model=64, n_layer=2, n_head=4, d_ff=96)
    model = AARUL(cfg).eval()
    # Random init leaves norms at exactly 1; perturb so the test exercises them.
    for name, p in model.named_parameters():
        if "norm" in name:
            p.add_(torch.randn_like(p) * 0.1)

    texts = [
        "Once upon a time, there was a little girl named Lily.",
        "She didn't like the rain!  \"Mom,\" she said.\n\nThe end.",
        "In 2024 she counted 1234567 stars... café — naïve 😀\t tabs",
        "",
        "<|endoftext|>Tom and Sue<|endoftext|>",
    ]
    tok_cases = [{"text": t, "ids": tok.encode(t)} for t in texts]

    ids = [tok.eot] + tok.encode("Once upon a time, there was a little dog named Max. He")
    logits, _, attns = model(torch.tensor([ids]), return_attn=True)
    ref = {
        "ids": ids,
        "logits_last": logits[0, -1].tolist(),
        "logits_first": logits[0, 0].tolist(),
        "attn_last": [a[0, :, -1, :].tolist() for a in attns],  # [layer][head][pos]
    }
    for dtype in ("f32", "q8"):
        write_bin(model, os.path.join(FIXTURES, f"tiny_{dtype}.bin"), dtype, {"fixture": True})
    q = dequantized_copy(model)
    ref["logits_last_q8"] = q(torch.tensor([ids]))[0][0, -1].tolist()
    with open(os.path.join(FIXTURES, "reference.json"), "w", encoding="utf-8") as f:
        json.dump({"tokenizer": tok_cases, "model": ref}, f)
    with open(os.path.join(DATA, "tokenizer.json"), encoding="utf-8") as src, \
         open(os.path.join(FIXTURES, "tokenizer.json"), "w", encoding="utf-8") as dst:
        dst.write(src.read())
    print(f"fixtures written to {FIXTURES}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt", nargs="?")
    ap.add_argument("--dtype", choices=["q8", "f16", "f32"], default="q8")
    ap.add_argument("--fixtures", action="store_true")
    a = ap.parse_args()
    if a.fixtures:
        export_fixtures()
    else:
        export_model(a.ckpt, a.dtype)
