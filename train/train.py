"""
Step 2: train AARUL on a single (laptop) GPU.

    python train.py --bench                  # measure tokens/sec for a config, then exit
    python train.py --steps 20000            # train; resumable with --resume

Writes to runs/<name>/:
    log.jsonl   one JSON object per line: training loss, validation loss, and
                text samples taken at fixed steps from the same prompt and seed,
                so you can watch the model go from noise to stories
    ckpt.pt     latest full checkpoint (model + optimizer), for --resume
    snap_<step>.pt  model-only snapshots at a few milestones
"""

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch

from model import AARUL, Config
from tokenizer import Tokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")

SAMPLE_PROMPT = "Once upon a time"
SAMPLE_STEPS = {0, 10, 25, 50, 75, 100, 150, 200, 300, 400, 500, 750, 1000, 1500, 2000, 3000, 4000, 6000, 8000}
SNAPSHOT_STEPS = {100, 300, 1000, 3000}


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="aarul")
    ap.add_argument("--d_model", type=int, default=384)
    ap.add_argument("--n_layer", type=int, default=8)
    ap.add_argument("--n_head", type=int, default=6)
    ap.add_argument("--d_ff", type=int, default=1024)
    ap.add_argument("--ctx", type=int, default=512)
    ap.add_argument("--batch", type=int, default=32, help="sequences per micro-batch")
    ap.add_argument("--accum", type=int, default=2, help="micro-batches per optimizer step")
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--min_lr_frac", type=float, default=0.1)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--wd", type=float, default=0.1)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--eval_every", type=int, default=500)
    ap.add_argument("--eval_batches", type=int, default=40)
    ap.add_argument("--sample_every", type=int, default=2000)
    ap.add_argument("--ckpt_every", type=int, default=500)
    ap.add_argument("--log_every", type=int, default=10)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--bench", action="store_true")
    return ap.parse_args()


def lr_at(step, args):
    if step < args.warmup:
        return args.lr * (step + 1) / args.warmup
    t = min(1.0, (step - args.warmup) / max(1, args.steps - args.warmup))
    min_lr = args.lr * args.min_lr_frac
    return min_lr + 0.5 * (args.lr - min_lr) * (1 + math.cos(math.pi * t))


class Batches:
    def __init__(self, path, ctx, batch, device, seed):
        self.data = np.memmap(path, dtype=np.uint16, mode="r")
        self.ctx, self.batch, self.device = ctx, batch, device
        self.rng = np.random.default_rng(seed)

    def next(self):
        ix = self.rng.integers(0, len(self.data) - self.ctx - 1, self.batch)
        chunk = np.stack([self.data[i : i + self.ctx + 1] for i in ix]).astype(np.int64)
        t = torch.from_numpy(chunk).pin_memory().to(self.device, non_blocking=True)
        return t[:, :-1], t[:, 1:]


@torch.no_grad()
def evaluate(model, batches, n, ctx_mgr):
    model.eval()
    losses = []
    for _ in range(n):
        x, y = batches.next()
        with ctx_mgr():
            _, loss, _ = model(x, y)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


@torch.no_grad()
def sample(model, tok, device, ctx_mgr):
    model.eval()
    g = torch.Generator(device=device).manual_seed(42)
    idx = torch.tensor([[tok.eot] + tok.encode(SAMPLE_PROMPT)], device=device)
    with ctx_mgr():
        out = model.generate(idx, 120, temperature=0.7, top_k=40, stop=tok.eot, generator=g)
    model.train()
    ids = [i for i in out[0].tolist()[1:] if i != tok.eot]
    return tok.decode(ids)


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # Windows consoles default to cp1252
    args = parse_args()
    torch.manual_seed(args.seed)
    device = "cuda"
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    autocast = lambda: torch.autocast("cuda", dtype=torch.bfloat16)  # noqa: E731

    tok = Tokenizer.load(os.path.join(DATA, "tokenizer.json"))
    cfg = Config(
        vocab_size=tok.vocab_size, ctx=args.ctx, d_model=args.d_model,
        n_layer=args.n_layer, n_head=args.n_head, d_ff=args.d_ff,
    )
    model = AARUL(cfg).to(device)

    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": args.wd}, {"params": no_decay, "weight_decay": 0.0}],
        lr=args.lr, betas=(0.9, 0.95), eps=1e-8, fused=True,
    )

    train_b = Batches(os.path.join(DATA, "train.bin"), cfg.ctx, args.batch, device, args.seed)
    valid_b = Batches(os.path.join(DATA, "valid.bin"), cfg.ctx, args.batch, device, args.seed + 1)
    tokens_per_step = args.batch * args.accum * cfg.ctx

    if args.bench:
        for i in range(25):
            if i == 5:
                torch.cuda.synchronize()
                t0 = time.time()
            for _ in range(args.accum):
                x, y = train_b.next()
                with autocast():
                    _, loss, _ = model(x, y)
                (loss / args.accum).backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        tps = 20 * tokens_per_step / (time.time() - t0)
        mem = torch.cuda.max_memory_allocated() / 2**30
        print(json.dumps({
            "params_M": round(model.n_params() / 1e6, 2), "tok_per_s": round(tps),
            "step_s": round(tokens_per_step / tps, 3), "mem_GB": round(mem, 2),
            "hours_for_steps": round(args.steps * tokens_per_step / tps / 3600, 2),
        }))
        return

    run_dir = os.path.join(HERE, "runs", args.name)
    os.makedirs(run_dir, exist_ok=True)
    ckpt_path = os.path.join(run_dir, "ckpt.pt")
    step, elapsed = 0, 0.0
    if args.resume and os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        step, elapsed = ck["step"], ck["elapsed"]
        train_b.rng = np.random.default_rng(args.seed + step)
        print(f"resumed at step {step}")
    else:
        with open(os.path.join(run_dir, "config.json"), "w") as f:
            json.dump({"model": cfg.to_dict(), "train": vars(args), "params": model.n_params(),
                       "gpu": torch.cuda.get_device_name(0)}, f, indent=2)

    log = open(os.path.join(run_dir, "log.jsonl"), "a", encoding="utf-8")

    def emit(**rec):
        log.write(json.dumps(rec) + "\n")
        log.flush()

    def save():
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step,
                    "elapsed": elapsed, "config": cfg.to_dict()}, ckpt_path + ".tmp")
        os.replace(ckpt_path + ".tmp", ckpt_path)

    print(f"AARUL: {model.n_params() / 1e6:.2f}M params, {tokens_per_step:,} tokens/step, {args.steps:,} steps")
    t_last = time.time()
    try:
        while step <= args.steps:
            if step in SAMPLE_STEPS or step % args.sample_every == 0 or step == args.steps:
                text = sample(model, tok, device, autocast)
                emit(step=step, tokens=step * tokens_per_step, sample=text)
                print(f"--- step {step} sample ---\n{text}\n")
            if step % args.eval_every == 0 or step == args.steps:
                v = evaluate(model, valid_b, args.eval_batches, autocast)
                emit(step=step, tokens=step * tokens_per_step, val_loss=round(v, 4), elapsed=round(elapsed))
                print(f"step {step}: val loss {v:.4f}")
            if step in SNAPSHOT_STEPS:
                torch.save({"model": model.state_dict(), "config": cfg.to_dict(), "step": step},
                           os.path.join(run_dir, f"snap_{step}.pt"))
            if step == args.steps:
                break

            lr = lr_at(step, args)
            for g in opt.param_groups:
                g["lr"] = lr
            total = 0.0
            for _ in range(args.accum):
                x, y = train_b.next()
                with autocast():
                    _, loss, _ = model(x, y)
                (loss / args.accum).backward()
                total += loss.item() / args.accum
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            opt.step()
            opt.zero_grad(set_to_none=True)
            step += 1

            now = time.time()
            elapsed += now - t_last
            t_last = now
            if step % args.log_every == 0:
                emit(step=step, tokens=step * tokens_per_step, loss=round(total, 4), lr=lr,
                     grad_norm=round(norm.item(), 3), elapsed=round(elapsed, 1))
            if step % 100 == 0:
                eta = elapsed / step * (args.steps - step) / 60 if step else 0
                print(f"step {step}/{args.steps} loss {total:.4f} lr {lr:.2e} "
                      f"{step * tokens_per_step / max(elapsed, 1e-9):,.0f} tok/s, eta {eta:.0f} min")
            if step % args.ckpt_every == 0:
                save()
    except KeyboardInterrupt:
        print("interrupted, saving checkpoint")
    save()
    torch.save({"model": model.state_dict(), "config": cfg.to_dict(), "step": step, "elapsed": elapsed},
               os.path.join(run_dir, "final.pt"))
    print(f"done: step {step}, {elapsed / 60:.1f} min")


if __name__ == "__main__":
    main()
