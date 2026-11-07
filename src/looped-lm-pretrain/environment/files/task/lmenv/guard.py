"""A tripwire for submissions that patch the harness, torch or the task constants while they run.

Submission code is imported into the grader's own processes (the checks, training and evaluation), so
it could rebind, say, ``F.cross_entropy`` or ``lmenv.task.WEIGHT_CAP``. fingerprint() records the
identity of every function and class in the modules below and the values of the task constants;
changes() lists what differs since. Run before importing a submission and compare afterwards. A
determined submission could also patch this module, so it catches attempts, it does not prevent them;
running submissions in a separate process is the real fix.
"""

from __future__ import annotations

import sys
import types

import torch
import torch._dynamo  # noqa: F401  first imported, it rewraps torch.manual_seed; do that before any fingerprint
from torch import nn
from torch._dynamo.mutation_guard import install_generation_tagging_init

# torch.compile's first call rewraps Module.__init__ and __setstate__ (once per process); do that now too.
install_generation_tagging_init()

MODULES = ("builtins", "math", "json", "numpy", "torch", "torch.nn", "torch.nn.functional", "torch.autograd",
           "torch.utils.flop_counter", "lmenv.task", "lmenv.api", "lmenv.harness", "lmenv.checks", "lmenv.run",
           "lmenv.guard", "contract")
CLASSES = (torch.Tensor, nn.Module, nn.Embedding, nn.Parameter)
_MISSING = object()


def fingerprint() -> dict:
    out = {}
    for name in MODULES:
        module = sys.modules.get(name)
        for key, value in (vars(module).items() if module else ()):
            if name == "lmenv.task" and isinstance(value, (bool, int, float, str, tuple)):
                out[f"{name}.{key}"] = value
            elif callable(value) or isinstance(value, types.ModuleType):
                out[f"{name}.{key}"] = id(value)
    for cls in CLASSES:
        for key, value in vars(cls).items():
            out[f"{cls.__module__}.{cls.__qualname__}.{key}"] = id(value)
    return out


def changes(before: dict) -> list[str]:
    """Names whose function, class or value differs from ``before`` (names added since are ignored)."""

    after = fingerprint()
    return sorted(key for key, value in before.items() if after.get(key, _MISSING) != value)
