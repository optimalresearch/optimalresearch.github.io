"""Starting training recipe: plain next-token loss.

Random 512-token windows, AdamW (betas 0.9, 0.95; weight decay on matrices
only), bf16 autocast, gradient clipping at 1, a linear warmup over the first
5% of the budget and a cosine decay to 10% of the learning rate over the
rest. Training stops when the budget is spent.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def train(ctx):
    cfg = ctx.config["train"]
    model, budget = ctx.model, ctx.budget
    if cfg["compile"] and ctx.device.type == "cuda":
        model.inner.body.stack = torch.compile(model.inner.body.apply_blocks)
    params = list(model.parameters())
    optimizer = torch.optim.AdamW([{"params": [p for p in params if p.dim() >= 2], "weight_decay": cfg["weight_decay"]},
                                   {"params": [p for p in params if p.dim() < 2], "weight_decay": 0.0}],
                                  lr=cfg["learning_rate"], betas=(0.9, 0.95), fused=ctx.device.type == "cuda")

    def lr_at(progress: float) -> float:
        lr, warmup = cfg["learning_rate"], cfg["warmup"]
        if progress < warmup:
            return lr * max(progress, 1e-4) / warmup
        p = (progress - warmup) / (1 - warmup)
        return lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, p))))

    update = 0
    model.train()
    while not budget.exhausted:
        for group in optimizer.param_groups:
            group["lr"] = lr_at(budget.fraction_used)
        total = 0.0
        for _ in range(cfg["accumulation"]):
            batch = ctx.data.sample(cfg["micro_batch"])
            with torch.autocast(ctx.device.type, dtype=torch.bfloat16):
                logits = model(batch.tokens)
            loss = F.cross_entropy(logits.float().flatten(0, 1), batch.targets.flatten())
            (loss / cfg["accumulation"]).backward()
            total += loss.item() / cfg["accumulation"]
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        update += 1
        if not math.isfinite(total):
            raise FloatingPointError(f"loss is {total} at update {update}")
        if update % cfg["log_every"] == 0:
            ctx.log({"update": update, "loss": total, "lr": optimizer.param_groups[0]["lr"]})
