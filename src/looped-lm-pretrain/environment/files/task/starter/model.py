"""The starting model: a small dense GPT-2-style language model.

Four transformer blocks (width 256, 4 heads, causal attention with RoPE on
queries and keys, GELU-tanh MLP of width 896), each used once per token. The
model returns the final hidden states; the harness turns them into logits
with the token embedding table (tied output head). 2,896,928 unique weights besides
the 8192 x 256 embedding (not counted), about 12M FLOPs per token.

Initialisation: weights N(0, 0.02), biases 0, and every block's output
projections (attention and MLP c_proj) N(0, 0.02 / sqrt(2 * n_layers)).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class ModelConfig:
    vocab_size: int = 8192
    d_model: int = 256
    n_heads: int = 4
    n_layers: int = 4
    mlp_width: int = 896
    rope_theta: float = 10_000.0
    layer_norm_eps: float = 1e-5


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.c_attn = nn.Linear(cfg.d_model, 3 * cfg.d_model)
        self.c_proj = nn.Linear(cfg.d_model, cfg.d_model)

    def forward(self, x, cos, sin):
        batch, length, width = x.shape
        q, k, v = self.c_attn(x).split(width, dim=2)
        q, k, v = (t.view(batch, length, self.n_heads, self.head_dim).transpose(1, 2) for t in (q, k, v))
        q, k = q * cos + _rotate_half(q) * sin, k * cos + _rotate_half(k) * sin
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.c_proj(out.transpose(1, 2).reshape(batch, length, width))


class MLP(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.c_fc = nn.Linear(cfg.d_model, cfg.mlp_width)
        self.c_proj = nn.Linear(cfg.mlp_width, cfg.d_model)

    def forward(self, x):
        return self.c_proj(F.gelu(self.c_fc(x), approximate="tanh"))


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(cfg.d_model, eps=cfg.layer_norm_eps)
        self.attn = Attention(cfg)
        self.ln_2 = nn.LayerNorm(cfg.d_model, eps=cfg.layer_norm_eps)
        self.mlp = MLP(cfg)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.ln_1(x), cos, sin)
        return x + self.mlp(self.ln_2(x))


def _rotate_half(x):
    first, second = x.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


class GPT(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        self.ln_f = nn.LayerNorm(cfg.d_model, eps=cfg.layer_norm_eps)
        head_dim = cfg.d_model // cfg.n_heads
        inv_freq = 1.0 / cfg.rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        nn.init.normal_(self.embed.weight, std=0.02)
        for module in self.blocks.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                nn.init.zeros_(module.bias)
        for block in self.blocks:
            nn.init.normal_(block.attn.c_proj.weight, std=0.02 / math.sqrt(2 * cfg.n_layers))
            nn.init.normal_(block.mlp.c_proj.weight, std=0.02 / math.sqrt(2 * cfg.n_layers))
        self.stack = self.apply_blocks  # train.py replaces this with a compiled version

    def apply_blocks(self, h, cos, sin):
        for block in self.blocks:
            h = block(h, cos, sin)
        return h

    def forward(self, input_ids):
        length = input_ids.shape[1]
        positions = torch.arange(length, device=input_ids.device, dtype=torch.float32)
        angles = torch.outer(positions, self.inv_freq)
        angles = torch.cat((angles, angles), dim=-1)[None, None]
        h = self.stack(self.embed(input_ids), angles.cos(), angles.sin())
        return self.ln_f(h)  # the harness applies the tied output head


def build_model(config: dict) -> GPT:
    """config is config.json plus {"vocab_size": ...} from the harness."""

    return GPT(ModelConfig(vocab_size=config["vocab_size"], **config.get("model", {})))
