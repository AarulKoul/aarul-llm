"""
The AARUL transformer: a decoder-only language model written from scratch.

Same shape as the Llama family, scaled way down:
  - token embedding, tied with the output layer
  - n_layer blocks of: RMSNorm -> causal self-attention with rotary position
    embeddings (RoPE) -> residual, RMSNorm -> SwiGLU MLP -> residual
  - final RMSNorm -> logits over the vocabulary

No biases anywhere. web/src/engine/model.ts is a line-for-line port of
forward(), and the parity test checks both produce the same logits.
"""

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:
    vocab_size: int = 4096
    ctx: int = 512
    d_model: int = 384
    n_layer: int = 8
    n_head: int = 6
    d_ff: int = 1024
    rope_theta: float = 10000.0
    norm_eps: float = 1e-5

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_head

    def to_dict(self) -> dict:
        return asdict(self)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        x32 = x.float()
        return (x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + self.eps)).type_as(x) * self.weight


def rope_tables(cfg: Config, device=None):
    half = cfg.head_dim // 2
    inv_freq = cfg.rope_theta ** (-torch.arange(half, dtype=torch.float32, device=device) / half)
    angles = torch.outer(torch.arange(cfg.ctx, dtype=torch.float32, device=device), inv_freq)
    return angles.cos(), angles.sin()  # (ctx, head_dim / 2) each


def apply_rope(x, cos, sin):
    # x: (B, H, T, D). Rotate each (x[i], x[i + D/2]) pair by a position-dependent angle.
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


class Attention(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.wq = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.wk = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.wv = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.wo = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

    def forward(self, x, cos, sin, return_attn=False):
        B, T, C = x.shape
        H, D = self.cfg.n_head, self.cfg.head_dim
        q = self.wq(x).view(B, T, H, D).transpose(1, 2)
        k = self.wk(x).view(B, T, H, D).transpose(1, 2)
        v = self.wv(x).view(B, T, H, D).transpose(1, 2)
        q, k = apply_rope(q, cos[:T], sin[:T]), apply_rope(k, cos[:T], sin[:T])
        if return_attn:
            scores = (q @ k.transpose(-2, -1)) / math.sqrt(D)
            mask = torch.ones(T, T, dtype=torch.bool, device=x.device).triu(1)
            attn = scores.masked_fill(mask, float("-inf")).softmax(-1)
            y = attn @ v
        else:
            attn = None
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.wo(y.transpose(1, 2).reshape(B, T, C)), attn


class MLP(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.w1 = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)  # gate
        self.w3 = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)  # up
        self.w2 = nn.Linear(cfg.d_ff, cfg.d_model, bias=False)  # down

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Block(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.attn = Attention(cfg)
        self.mlp_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.mlp = MLP(cfg)

    def forward(self, x, cos, sin, return_attn=False):
        a, attn = self.attn(self.attn_norm(x), cos, sin, return_attn)
        x = x + a
        x = x + self.mlp(self.mlp_norm(x))
        return x, attn


class AARUL(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        cos, sin = rope_tables(cfg)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

        self.apply(self._init)
        # Scale down the projections that write into the residual stream, so the
        # stream's variance doesn't grow with depth (GPT-2 trick).
        for name, p in self.named_parameters():
            if name.endswith("wo.weight") or name.endswith("w2.weight"):
                nn.init.normal_(p, std=0.02 / math.sqrt(2 * cfg.n_layer))

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx, targets=None, return_attn=False):
        x = self.tok_emb(idx)
        attns = []
        for block in self.blocks:
            x, attn = block(x, self.cos, self.sin, return_attn)
            attns.append(attn)
        logits = F.linear(self.norm(x), self.tok_emb.weight)  # tied output layer
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss, (attns if return_attn else None)

    @torch.no_grad()
    def generate(self, idx, max_new: int, temperature=0.8, top_k=40, stop=None, generator=None):
        for _ in range(max_new):
            logits, _, _ = self(idx[:, -self.cfg.ctx :])
            logits = logits[:, -1, :].float()
            if temperature <= 0:
                nxt = logits.argmax(-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k:
                    kth = torch.topk(logits, top_k).values[:, -1, None]
                    logits = logits.masked_fill(logits < kth, float("-inf"))
                nxt = torch.multinomial(logits.softmax(-1), 1, generator=generator)
            idx = torch.cat([idx, nxt], dim=1)
            if stop is not None and nxt.item() == stop:
                break
        return idx
