"""What a submission provides.

submission/model.py defines

    def build_model(config: dict) -> nn.Module

where config is submission/config.json plus {"vocab_size": ...}. The model
maps token ids to hidden states, and must have its token embedding table as
the attribute embed = nn.Embedding(vocab_size, width):

    hidden = model(input_ids)              # [B, L] -> [B, L, width]

The harness owns the output layer: it wraps the model in
harness.OutputHead, which computes the next-token logits as

    logits = hidden @ model.embed.weight.T  # [B, L, vocab]

so every prediction goes through that table. Everything below calls the
wrapped model, which returns logits. Position i's logits may depend only on
tokens 0..i. Evaluation calls
model(input_ids) in eval mode with no other arguments; that call must stay
within task.INFERENCE_FLOP_CAP FLOPs per token. Any architecture is allowed
(dense, shared or looped blocks, anything else) within task.WEIGHT_CAP
unique weights.

In training, train.py may pass keyword arguments (say, a number of loops):
model(input_ids, loops=12). How much compute a call costs may depend only on
those arguments, the training/eval mode and the input's shape, never on
random draws inside the model or on the batch size: the harness measures
each kind of call once and charges every later call of that kind its tokens
times that cost.

submission/train.py defines

    def train(ctx: TrainContext) -> None

which trains ctx.model (a metered wrapper around OutputHead(build_model(config)),
see harness.MeteredModel; ctx.model(input_ids) returns logits, and the
submission's own module is ctx.model.inner.body) until the budget runs out: task.TRAIN_TOKENS tokens
through the model and task.TRAIN_FLOPS forward FLOPs, whichever comes first.
A call after that raises BudgetExhausted, which ends training normally. The
harness saves the model's weights when train returns or the budget is spent.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import torch


class BudgetExhausted(RuntimeError):
    """Raised by ctx.model when it is called after the training budget is spent; ends training normally."""


@dataclass
class Batch:
    tokens: torch.Tensor  # [B, L] input token ids
    targets: torch.Tensor  # [B, L] the next token at each position


@dataclass
class TrainContext:
    model: Any  # harness.MeteredModel
    budget: Any  # harness.Budget: .exhausted, .fraction_used, .used_tokens, .used_flops
    data: Any  # harness.TokenStream: .sample(batch_size, seq_len=task.SEQ_LEN) -> Batch
    config: dict  # submission/config.json
    seed: int
    device: torch.device
    log: Callable[[dict], None]  # records a dict of numbers in the run's train.json (and prints it)
