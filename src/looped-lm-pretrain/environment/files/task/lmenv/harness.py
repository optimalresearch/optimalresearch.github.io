"""The harness around a submission: weight count, FLOP metering, data, training run, evaluation.

Compute is counted in FLOPs with torch's FlopCounterMode (uncompiled; on the
CPU attention is swapped for an explicit matmul version so it is counted).
The submission's model returns hidden states; build() wraps it in
OutputHead, which turns them into logits with the model's token embedding
table, so every model here maps ids to logits. train.py sees the model only
through MeteredModel. Each kind of call (mode,
input length, keyword arguments) is measured once on a one-row batch, with
autograd on as in training; every call then charges its tokens and tokens x
that cost to the budget. A call after the budget is spent raises
BudgetExhausted, which ends training normally. These checks run inside the
submission's process and catch honest mistakes; the grader re-checks the
model from outside (checks.py).
"""

from __future__ import annotations

import contextlib
import copy
import importlib.util
import json
import math
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.flop_counter import FlopCounterMode

from lmenv import guard, task
from lmenv.api import Batch, BudgetExhausted, TrainContext

# Bound once here, so a submission that rebinds F.linear or F.cross_entropy cannot change the output layer
# or the loss we compute (guard.py also reports the attempt).
_LINEAR, _CROSS_ENTROPY = F.linear, F.cross_entropy


@dataclass
class Budget:
    tokens: float = task.TRAIN_TOKENS
    flops: float = task.TRAIN_FLOPS
    used_tokens: float = 0.0
    used_flops: float = 0.0
    largest_call: tuple = (0.0, 0.0)  # (tokens, FLOPs): the last call may overshoot the budget by this much
    started: float = field(default_factory=time.monotonic)

    def charge(self, tokens: float, flops: float) -> None:
        if self.exhausted:
            raise BudgetExhausted(self.describe())
        self.used_tokens += float(tokens)
        self.used_flops += float(flops)
        self.largest_call = (max(self.largest_call[0], float(tokens)), max(self.largest_call[1], float(flops)))

    @property
    def elapsed_s(self) -> float:
        return time.monotonic() - self.started

    @property
    def exhausted(self) -> bool:
        return self.fraction_used >= 1.0

    @property
    def fraction_used(self) -> float:
        return max(self.used_tokens / self.tokens, self.used_flops / self.flops)

    def describe(self) -> str:
        return (f"{self.used_tokens:.3g}/{self.tokens:.3g} tokens, {self.used_flops:.3g}/{self.flops:.3g} FLOPs, "
                f"{self.elapsed_s:.0f} s")


# ---------------------------------------------------------------- measurements

@contextlib.contextmanager
def countable_attention(device: torch.device):
    """On the CPU, FlopCounterMode misses fused attention: count an explicit version instead."""

    if device.type != "cpu":
        yield
        return
    original = F.scaled_dot_product_attention

    def explicit(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, **_):
        scores = q @ k.transpose(-2, -1) * (scale or 1.0 / math.sqrt(q.shape[-1]))
        if is_causal:
            length = q.shape[-2]
            scores = scores.masked_fill(~torch.ones(length, length, dtype=torch.bool).tril(), float("-inf"))
        if attn_mask is not None:
            scores = scores.masked_fill(~attn_mask, float("-inf")) if attn_mask.dtype == torch.bool else scores + attn_mask
        return torch.softmax(scores, dim=-1) @ v

    F.scaled_dot_product_attention = explicit
    try:
        yield
    finally:
        F.scaled_dot_product_attention = original


def flops_per_token(model: nn.Module, ids: torch.Tensor, kwargs: dict | None = None, training: bool = False,
                    grad: bool = False) -> float:
    """Forward FLOPs per token of model(ids, **kwargs) in the given mode, uncompiled."""

    was_training = model.training
    model.train(training)
    devices = [ids.device] if ids.device.type == "cuda" else []
    # force_eager runs compiled functions uncompiled just for this call. (Setting
    # torch._dynamo.config.disable instead marks a compiled function that is first
    # called here as never to be compiled, so training ran uncompiled.)
    try:
        with torch.random.fork_rng(devices=devices), torch.set_grad_enabled(grad), countable_attention(ids.device), \
                torch.compiler.set_stance("force_eager"):
            with FlopCounterMode(display=False) as counter:
                model(ids, **(kwargs or {}))
        return counter.get_total_flops() / ids.numel()
    finally:
        model.train(was_training)


def unique_weights(model: nn.Module, vocab_size: int) -> dict:
    """Unique weights (parameters and buffers, one count per storage), minus one exempt table.

    The exempt table is the largest stored tensor of shape (vocab, width) with
    width <= EXEMPT_TABLE_WIDTH: the token embedding, or the head tied to it.
    A storage counts all its elements, even behind an empty view; a complex
    element counts as two. Integer and boolean buffers that are not saved with
    the model (masks, indices rebuilt in __init__) are not weights; saved ones
    count one weight per 4 bytes, since they could hold trained values.
    """

    saved = {id(t) for t in model.state_dict(keep_vars=True).values() if isinstance(t, torch.Tensor)}
    storages: dict[int, tuple[int, tuple]] = {}
    for tensor in [*model.parameters(), *model.buffers()]:
        floating = tensor.is_floating_point() or tensor.is_complex()
        if not floating and id(tensor) not in saved:
            continue
        storage = tensor.untyped_storage()
        if floating:
            elements = storage.nbytes() // tensor.element_size() * (2 if tensor.is_complex() else 1)
        else:
            elements = -(-storage.nbytes() // 4)
        key = storage.data_ptr()
        if key not in storages or storages[key][0] < elements:
            storages[key] = (elements, tuple(tensor.shape))
    total = sum(n for n, _ in storages.values())
    tables = [n for n, shape in storages.values()
              if len(shape) == 2 and shape[0] == vocab_size and shape[1] <= task.EXEMPT_TABLE_WIDTH and n == shape[0] * shape[1]]
    exempt = max(tables, default=0)
    return {"total": total, "exempt": exempt, "counted": total - exempt}


@torch.no_grad()
def interface_failures(model: nn.Module, vocab_size: int, device, perturb: bool = True) -> list[str]:
    """model(ids) gives [B, L, vocab] logits, and position i depends only on tokens 0..i.

    With ``perturb``, the test runs on a copy with small noise added to every
    weight: a freshly initialised model can hide a leak behind zero-initialised
    projections or gates that training would open.
    """

    if perturb:
        model = copy.deepcopy(model)
        noise = torch.Generator(device="cpu").manual_seed(1)
        for p in model.parameters():
            if p.is_floating_point():
                p.add_(0.02 * torch.randn(p.shape, generator=noise).to(p.device, p.dtype))
    was_training = model.training
    model.eval()
    gen = torch.Generator().manual_seed(0)
    ids = torch.randint(0, vocab_size, (2, 64), generator=gen).to(device)
    changed = ids.clone()
    changed[:, 40:] = torch.randint(0, vocab_size, (2, 24), generator=gen).to(device)
    try:
        logits = model(ids).float()
        if tuple(logits.shape) != (2, 64, vocab_size):
            return [f"the model's output has shape {tuple(logits.shape)}, expected (batch, length, vocab)"]
        moved = (model(changed).float()[:, :40] - logits[:, :40]).abs().max().item()
        if moved > 1e-3:
            return [f"not causal: changing tokens 40.. moves the logits of earlier positions by {moved:.3g}"]
    except Exception as error:  # the submission's code can fail in any way
        return [f"the model cannot be run: {type(error).__name__}: {error}"[:500]]
    finally:
        model.train(was_training)
    return []


def cap_failures(model: nn.Module, vocab_size: int, device) -> tuple[dict, list[str]]:
    """Unique weights and inference FLOPs per token against the caps."""

    weights = unique_weights(model, vocab_size)
    ids = torch.randint(0, vocab_size, (1, task.SEQ_LEN), generator=torch.Generator().manual_seed(0)).to(device)
    flops = flops_per_token(model, ids)
    failures = []
    if weights["counted"] > task.WEIGHT_CAP:
        failures.append(f"{weights['counted']:,} unique weights (cap {task.WEIGHT_CAP:,}; one embedding table of "
                        f"{weights['exempt']:,} not counted)")
    if flops > task.INFERENCE_FLOP_CAP:
        failures.append(f"inference costs {flops / 1e6:.1f}M FLOPs per token (cap {task.INFERENCE_FLOP_CAP / 1e6:.0f}M)")
    return {"weights": weights, "inference_flops_per_token": flops}, failures


# ---------------------------------------------------------------- the metered model

def _call_key(training: bool, ids: torch.Tensor, kwargs: dict) -> tuple:
    for name, value in kwargs.items():
        if not isinstance(value, (bool, int, float, str, type(None))):
            raise TypeError(f"model keyword argument {name!r} must be a number, string, bool or None, "
                            f"not {type(value).__name__}")
    return (training, ids.dim(), ids.shape[-1], tuple(sorted(kwargs.items())))


class MeteredModel(nn.Module):
    """ctx.model: calls the submission's model and charges each call to the budget."""

    def __init__(self, model: nn.Module, budget: Budget):
        super().__init__()
        self.inner = model
        self.budget = budget
        self.costs: dict[tuple, float] = {}

    def forward(self, input_ids: torch.Tensor, **kwargs) -> torch.Tensor:
        if self.budget.exhausted:
            raise BudgetExhausted(self.budget.describe())
        key = _call_key(self.inner.training, input_ids, kwargs)
        if key not in self.costs:
            if len(self.costs) >= task.MAX_CALL_KINDS:
                raise RuntimeError(f"more than {task.MAX_CALL_KINDS} kinds of model call")
            self.costs[key] = flops_per_token(self.inner, input_ids[:1], kwargs, training=self.inner.training,
                                              grad=torch.is_grad_enabled())
        tokens = input_ids.numel()
        self.budget.charge(tokens, tokens * self.costs[key])
        return self.inner(input_ids, **kwargs)

    def cost_table(self) -> list[dict]:
        return [{"training": k[0], "length": k[2], "kwargs": dict(k[3]), "flops_per_token": v}
                for k, v in self.costs.items()]


class TokenStream:
    """One pass over the first ``corpus`` training tokens (CORPUS_TOKENS by default), in shuffled order.

    The text is cut into blocks of SEQ_LEN tokens, visited in a random order;
    every token is a target once per pass. A row of seq_len < SEQ_LEN takes the
    next seq_len tokens of the current block (the block's remainder shorter than
    seq_len is skipped). When the blocks run out, a new pass starts in a new order.
    """

    def __init__(self, data: Path, seed: int, corpus: int = task.CORPUS_TOKENS):
        self.tokens = np.memmap(data / "train.bin", dtype=np.uint16, mode="r")[:int(corpus)]
        self.rng = np.random.default_rng(seed)
        self.device = torch.device("cpu")
        self.blocks = (len(self.tokens) - 1) // task.SEQ_LEN
        self.order, self.next_block, self.offset = self.rng.permutation(self.blocks), 0, task.SEQ_LEN
        self.passes = 0

    def _row_start(self, seq_len: int) -> int:
        if self.offset + seq_len > task.SEQ_LEN:
            if self.next_block == self.blocks:
                self.order, self.next_block = self.rng.permutation(self.blocks), 0
                self.passes += 1
            self.block, self.next_block, self.offset = self.order[self.next_block], self.next_block + 1, 0
        start = int(self.block) * task.SEQ_LEN + self.offset
        self.offset += seq_len
        return start

    def sample(self, batch_size: int, seq_len: int = task.SEQ_LEN) -> Batch:
        if not 1 <= seq_len <= task.SEQ_LEN:
            raise ValueError(f"seq_len must be in 1..{task.SEQ_LEN}")
        starts = [self._row_start(seq_len) for _ in range(batch_size)]
        rows = np.stack([self.tokens[s:s + seq_len + 1] for s in starts]).astype(np.int64)
        rows = torch.from_numpy(rows).to(self.device, non_blocking=True)
        return Batch(tokens=rows[:, :-1], targets=rows[:, 1:])


# ---------------------------------------------------------------- submission code

def load_module(directory: Path, name: str):
    path = directory / f"{name}.py"
    if not path.exists():
        return None
    if str(directory.resolve()) not in sys.path:
        sys.path.insert(0, str(directory.resolve()))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # so train.py can "import model"
    spec.loader.exec_module(module)
    return module


def read_config(submission: Path) -> dict:
    path = submission / "config.json"
    return json.loads(path.read_text()) if path.exists() else {}


class OutputHead(nn.Module):
    """The submission's model followed by the harness's output layer.

    The submission's model (``body``) returns hidden states [B, L, width]; the
    logits are those states times its token embedding table, body.embed
    (vocab x width): logits = hidden @ body.embed.weight.T. Every prediction
    therefore goes through the tied table, and nothing can put probability on
    tokens any other way (for example by copying ids from the input).
    """

    def __init__(self, body: nn.Module, vocab_size: int):
        super().__init__()
        embed = getattr(body, "embed", None)
        # Exact types: a subclass could change what the output layer computes.
        if type(embed) is not nn.Embedding or type(embed.weight) is not nn.Parameter \
                or embed.num_embeddings != vocab_size:
            raise TypeError(f"the model needs an attribute embed = nn.Embedding({vocab_size}, width); "
                            f"the harness computes the logits as hidden @ embed.weight.T")
        self.body = body

    def forward(self, input_ids: torch.Tensor, **kwargs) -> torch.Tensor:
        hidden = self.body(input_ids, **kwargs)
        if type(hidden) is not torch.Tensor:  # a Tensor subclass could intercept the output layer
            raise TypeError(f"the model must return a plain torch.Tensor of hidden states [batch, length, width], "
                            f"not {type(hidden).__name__}")
        return _LINEAR(hidden, self.body.embed.weight)


def build(submission: Path, vocab_size: int, device) -> nn.Module:
    """The submission's model inside the harness's output layer (OutputHead)."""

    module = load_module(submission, "model")
    if module is None:
        raise FileNotFoundError(f"{submission}/model.py is missing")
    body = module.build_model({**read_config(submission), "vocab_size": vocab_size})
    return OutputHead(body, vocab_size).to(device)


def load_trained(submission: Path, run: Path, vocab_size: int, device) -> nn.Module:
    model = build(submission, vocab_size, device)
    weights = torch.load(run / "model.pt", map_location=device)
    model.load_state_dict(weights)
    return model.eval()


def saved_state(model: nn.Module) -> dict:
    # torch.compile'd submodules add "_orig_mod." to their names; save the plain names.
    return {k.replace("_orig_mod.", ""): v.detach().cpu() for k, v in model.state_dict().items()}


# ---------------------------------------------------------------- training and evaluation

def train_run(submission: Path, data: Path, out: Path, seed: int, device, tokens: float = task.TRAIN_TOKENS,
              log=print, corpus: int = task.CORPUS_TOKENS) -> dict:
    """Train one seed of the submission; writes OUT/model.pt and OUT/train.json and returns the record.

    ``tokens`` below TRAIN_TOKENS scales both budgets down (a quick proxy run).
    ``corpus`` (design experiments only) sets how many training tokens are sampled from.
    """

    out.mkdir(parents=True, exist_ok=True)
    vocab = json.loads((data / "meta.json").read_text())["vocab_size"]
    record: dict = {"seed": seed, "status": "error", "error": None, "budget_tokens": tokens, "corpus": corpus,
                    "history": []}

    def finish(**fields) -> dict:
        record.update(fields)
        (out / "train.json").write_text(json.dumps(record, indent=2))
        return record

    untouched = guard.fingerprint()
    try:
        torch.manual_seed(seed)
        np.random.seed(seed)
        model = build(submission, vocab, device)
        trainer = load_module(submission, "train")
        if trainer is None:
            raise FileNotFoundError(f"{submission}/train.py is missing")
    except Exception as error:
        return finish(error=f"building the model: {type(error).__name__}: {error}"[:2000])
    failures = interface_failures(model, vocab, device)
    if failures:
        return finish(error=failures[0])
    try:
        caps, failures = cap_failures(model, vocab, device)
    except Exception as error:
        return finish(error=f"measuring the model: {type(error).__name__}: {error}"[:2000])
    log(f"{caps['weights']['counted']:,} unique weights (+{caps['weights']['exempt']:,} embedding), "
        f"{caps['inference_flops_per_token'] / 1e6:.1f}M inference FLOPs per token")
    if failures:
        return finish(error="over the caps: " + "; ".join(failures), caps=caps)
    meter = Budget(tokens=tokens, flops=tokens * task.INFERENCE_FLOP_CAP)
    metered = MeteredModel(model, meter)
    stream = TokenStream(data, seed, corpus)
    stream.device = device
    history = record["history"]

    def record_log(entry: dict) -> None:
        entry = {**entry, "tokens": meter.used_tokens, "flops": meter.used_flops, "elapsed_s": round(meter.elapsed_s, 1)}
        history.append(entry)
        log("  ".join(f"{k} {v:.4g}" if isinstance(v, float) else f"{k} {v}" for k, v in entry.items()))

    ctx = TrainContext(model=metered, budget=meter, data=stream, config=read_config(submission), seed=seed,
                       device=device, log=record_log)
    status, error = "complete", None
    started = time.time()
    try:
        trainer.train(ctx)
    except BudgetExhausted:
        pass
    except Exception as e:  # the submission's code can fail in any way
        status, error = "error", f"train.py: {type(e).__name__}: {e}\n{traceback.format_exc()[-1500:]}"
    torch.save(saved_state(model), out / "model.pt")
    return finish(status=status, error=error, caps=caps, used_tokens=meter.used_tokens, used_flops=meter.used_flops,
                  call_costs=metered.cost_table(), largest_call=list(meter.largest_call),
                  train_seconds=time.time() - started,
                  corpus_passes=stream.passes + stream.next_block / stream.blocks, tampering=guard.changes(untouched))


_TOKEN_BYTES: dict = {}


def token_bytes(tokenizer: Path, eot: int) -> np.ndarray:
    """UTF-8 bytes of each token of our byte-level BPE (the end-of-text token counts as one byte)."""

    key = str(tokenizer)
    if key not in _TOKEN_BYTES:
        vocab = json.loads(tokenizer.read_text())["model"]["vocab"]
        # GPT-2's byte-to-character map: printable bytes stand for themselves, the rest from 256 up.
        printable = [*range(ord("!"), ord("~") + 1), *range(ord("¡"), ord("¬") + 1), *range(ord("®"), ord("ÿ") + 1)]
        to_char = {b: chr(b) for b in printable}
        extra = 0
        for b in range(256):
            if b not in to_char:
                to_char[b] = chr(256 + extra)
                extra += 1
        known = set(to_char.values())
        lengths = np.zeros(max(vocab.values()) + 1, dtype=np.int64)
        for piece, index in vocab.items():
            lengths[index] = len(piece) if all(c in known for c in piece) else len(piece.encode())
        lengths[eot] = 1
        _TOKEN_BYTES[key] = lengths
    return _TOKEN_BYTES[key]


@torch.no_grad()
def evaluate(model: nn.Module, tokens: np.ndarray, device, byte_lengths: np.ndarray | None = None,
             batch: int = 32) -> dict:
    """Mean next-token loss (nats per token) on every full SEQ_LEN window of ``tokens``, and bits per byte."""

    model.eval()
    n = (len(tokens) - 1) // task.SEQ_LEN
    total = torch.zeros((), dtype=torch.float64, device=device)
    torch.manual_seed(0)
    for start in range(0, n, batch):
        stop = min(n, start + batch)
        rows = np.stack([tokens[i * task.SEQ_LEN:(i + 1) * task.SEQ_LEN + 1] for i in range(start, stop)])
        rows = torch.from_numpy(rows.astype(np.int64)).to(device)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits = model(rows[:, :-1])
        total += _CROSS_ENTROPY(logits.float().flatten(0, 1), rows[:, 1:].flatten(), reduction="sum")
    loss = total.item() / (n * task.SEQ_LEN)
    result = {"loss": loss if math.isfinite(loss) else float("inf"), "tokens": n * task.SEQ_LEN}
    if byte_lengths is not None:
        targets = np.asarray(tokens[1:n * task.SEQ_LEN + 1], dtype=np.int64)
        nbytes = int(byte_lengths[targets].sum())
        result["bits_per_byte"] = total.item() / math.log(2) / nbytes
        result["bytes"] = nbytes
    return result
