"""Checks the grader runs on a submission, outside the training process.

* weights      unique floating-point weights against the cap, one embedding
               table exempt (harness.unique_weights)
* flops        inference FLOPs per token of model(ids) in eval mode, on random
               and on real text, against the cap
* interface    the model has an embed table, and with the harness's output
               layer (harness.OutputHead) is causal and gives [B, L, vocab] logits
               (harness.interface_failures)
* randomness   evaluation draws no random numbers (no weights generated from
               a seed inside forward) and two passes give identical logits
* call costs   every kind of training call recorded in train.json costs the
               same per token at batch size 1 and 2 and under two seeds, and
               no less than the training run recorded, so the harness's
               once-per-kind measurement is honest
* uncounted    no heavy operation the FLOP counter does not count (bilinear,
               mv, dot, cdist, FFT, int8 and sparse matmuls) at evaluation or
               in a recorded kind of training call
* files        the grader keeps only *.py files and config.json (size-limited);
               anything else in the submission is dropped
* state        persistent buffers and frozen parameters, listed for review;
               extra state (get_extra_state) fails
* static scan  source patterns that break the rules (network, subprocess,
               custom kernels, writing to the harness's counters, calling the
               unwrapped model in training) or that need a closer look
"""

from __future__ import annotations

import ast
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
from torch.utils._python_dispatch import TorchDispatchMode

from lmenv import task
from lmenv.harness import cap_failures, flops_per_token, interface_failures

# Heavy aten operations FlopCounterMode does not count: a model could compute with them for free.
UNCOUNTED_OPS = {"_trilinear", "mv", "addmv", "dot", "vdot", "_cdist_forward", "_euclidean_dist", "_fft_r2c", "_fft_c2r",
                 "_fft_c2c", "_int_mm", "_sparse_mm", "_sparse_addmm", "sparse_sampled_addmm", "_weight_int8pack_mm",
                 "_weight_int4pack_mm", "_cslt_sparse_mm", "_sparse_semi_structured_linear"}
# What the grader keeps of a submission (grader.grade copies only these), with size limits in bytes.
KEPT_FILES = {"*.py": 100_000, "config.json": 16_000}
RANDOM_OPS = {"rand", "randn", "randint", "randperm", "normal", "normal_", "uniform_", "bernoulli", "bernoulli_",
              "multinomial", "exponential_", "random_", "native_dropout", "rand_like", "randn_like", "randint_like"}


class UncountedOps(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.seen: set[str] = set()

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if func.overloadpacket.__name__ in UNCOUNTED_OPS:
            self.seen.add(func.overloadpacket.__name__)
        return func(*args, **(kwargs or {}))


def uncounted_ops(model, ids, kwargs: dict | None = None, training: bool = False) -> set[str]:
    """Heavy operations the FLOP counter misses in one uncompiled forward call."""

    was_training = model.training
    model.train(training)
    try:
        with torch.random.fork_rng(devices=[ids.device] if ids.device.type == "cuda" else []), \
                torch.compiler.set_stance("force_eager"), UncountedOps() as mode:
            model(ids, **(kwargs or {}))
        return mode.seen
    finally:
        model.train(was_training)


def submission_files(directory: Path) -> tuple[list[Path], list[str], list[str]]:
    """(files the grader keeps, files it drops, kept files over their size limit)."""

    kept, dropped, oversized = [], [], []
    for path in sorted(p for p in Path(directory).rglob("*") if p.is_file() and "__pycache__" not in p.parts):
        name = path.relative_to(directory).as_posix()
        limit = (KEPT_FILES["config.json"] if name == "config.json" else
                 KEPT_FILES["*.py"] if path.suffix == ".py" and "/" not in name else None)
        if limit is None:
            dropped.append(name)
            continue
        kept.append(path)
        if path.stat().st_size > limit:
            oversized.append(f"{name} ({path.stat().st_size:,} bytes, limit {limit:,})")
    return kept, dropped, oversized


class RandomOps(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.seen: set[str] = set()

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        name = func.overloadpacket.__name__
        if name in RANDOM_OPS:
            self.seen.add(name)
        return func(*args, **(kwargs or {}))


@dataclass
class CheckReport:
    weights: dict = field(default_factory=dict)
    inference_flops_per_token: float = 0.0
    inference_flops_on_text: float = 0.0
    random_ops_in_eval: list[str] = field(default_factory=list)
    deterministic: bool = True
    call_costs: list[dict] = field(default_factory=list)
    static_findings: list[dict] = field(default_factory=list)
    state_notes: list[str] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.failures

    def to_dict(self) -> dict:
        return {**asdict(self), "passed": self.passed}


@torch.no_grad()
def check_eval(model, ids: torch.Tensor, text: torch.Tensor | None, report: CheckReport) -> None:
    """FLOPs on real text, random draws and determinism of model(ids) in eval mode."""

    if text is not None:
        report.inference_flops_on_text = flops_per_token(model, text)
        if report.inference_flops_on_text > task.INFERENCE_FLOP_CAP:
            report.failures.append(f"inference costs {report.inference_flops_on_text / 1e6:.1f}M FLOPs per token on "
                                   f"real text (cap {task.INFERENCE_FLOP_CAP / 1e6:.0f}M)")
    model.eval()
    mode = RandomOps()
    with mode:
        first = model(ids)
    report.random_ops_in_eval = sorted(mode.seen)
    if mode.seen:
        report.failures.append("evaluation draws random numbers (" + ", ".join(sorted(mode.seen)) + "); "
                               "model(ids) in eval mode must be deterministic")
    report.deterministic = torch.equal(first, model(ids))
    if not report.deterministic:
        report.failures.append("two evaluation passes give different logits")
    for name in sorted(uncounted_ops(model, ids)):
        report.failures.append(f"evaluation uses {name}, which the FLOP counter does not count")


def check_call_costs(model, recorded: list[dict], vocab_size: int, device, report: CheckReport) -> None:
    """Each recorded kind of training call costs the same per token at batch 1 and 2 and under two seeds,
    no less than the cost the training run charged, and uses no operation the FLOP counter misses."""

    for call in recorded:
        gen = torch.Generator().manual_seed(0)
        ids = torch.randint(0, vocab_size, (2, call["length"]), generator=gen).to(device)
        costs = []
        for rows, seed in ((1, 0), (2, 0), (1, 1)):
            torch.manual_seed(seed)
            costs.append(flops_per_token(model, ids[:rows], call["kwargs"], training=call["training"], grad=True))
        entry = {**call, "measured": costs}
        report.call_costs.append(entry)
        if max(costs) > 1.01 * min(costs) + 1:
            report.failures.append(f"the cost of model(ids, **{call['kwargs']}) (training={call['training']}) "
                                   f"depends on the batch size or on random draws: {[round(c) for c in costs]} "
                                   "FLOPs per token")
        if "flops_per_token" in call and call["flops_per_token"] < 0.99 * min(costs) - 1:
            report.failures.append(f"training charged {call['flops_per_token']:.4g} FLOPs per token for "
                                   f"model(ids, **{call['kwargs']}) (training={call['training']}), "
                                   f"which costs {min(costs):.4g}")
        with torch.enable_grad():
            for name in sorted(uncounted_ops(model, ids[:1], call["kwargs"], training=call["training"])):
                report.failures.append(f"model(ids, **{call['kwargs']}) (training={call['training']}) uses {name}, "
                                       "which the FLOP counter does not count")


def check_state(model, report: CheckReport) -> None:
    """List state that is not trained by gradient descent; a reviewer decides if it is legitimate."""

    parameter_names = {name for name, _ in model.named_parameters()}
    for name, value in model.state_dict().items():
        if name.endswith("_extra_state"):
            report.failures.append(f"{name}: extra state (get_extra_state) is saved outside the weight count")
        elif name not in parameter_names:
            report.state_notes.append(f"persistent buffer {name} {tuple(value.shape)} {value.dtype}")
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            report.state_notes.append(f"frozen parameter {name} {tuple(parameter.shape)}")


# ---------------------------------------------------------------- static scan

BANNED_MODULES = {"socket", "urllib", "http", "requests", "httpx", "aiohttp", "ftplib", "smtplib",
                  "subprocess", "ctypes", "pty",
                  # Kernels launched outside torch's dispatcher are invisible to the FLOP counter.
                  "triton", "numba", "cupy", "jax", "torch.utils.cpp_extension", "torch.library"}
REVIEW_MODULES = {"importlib", "inspect", "gc", "pickle", "shutil", "glob"}
ALLOWED_HARNESS = {"lmenv.api", "lmenv.task"}
PROTECTED_ATTRIBUTES = {"budget", "used_tokens", "used_flops", "costs", "inner", "charge",
                        "exhausted"}
REVIEW_BUILTINS = {"open", "exec", "eval", "__import__", "setattr", "getattr", "vars", "delattr"}
REVIEW_METHODS = {"load", "loadtxt", "fromfile", "system", "popen", "_getframe", "currentframe",
                  "get_objects", "getenv", "manual_seed", "memmap", "realpath", "register", "load_library",
                  "get_extra_state", "is_compiling", "is_autocast_enabled", "_get_current_dispatch_mode_stack",
                  "register_hook", "register_full_backward_hook", "register_forward_hook", "write_text", "write_bytes"}
REVIEW_ATTRIBUTES = {"__dict__", "_modules", "_parameters", "_buffers", "filename", "argv", "_python_dispatch"}
# Mutating a harness counter in place (ctx.model.costs[k] = 0, ctx.model.costs.clear()).
MUTATING_METHODS = {"clear", "update", "pop", "popitem", "setdefault", "__setitem__", "__delitem__"}


def _finding(severity: str, rule: str, path: Path, node: ast.AST, source: list[str]) -> dict:
    line = getattr(node, "lineno", 0)
    return {"severity": severity, "rule": rule, "file": path.name, "line": line,
            "code": source[line - 1].strip() if 0 < line <= len(source) else ""}


def _unwrapped(node: ast.AST) -> bool:
    """ctx.model.inner or ctx.model.inner.body: the model without the meter."""

    return isinstance(node, ast.Attribute) and (
        node.attr == "inner" or (node.attr == "body" and isinstance(node.value, ast.Attribute)
                                 and node.value.attr == "inner"))


def static_scan(directory: Path) -> list[dict]:
    findings = []
    for path in sorted(Path(directory).rglob("*.py")):
        text = path.read_text()
        source = text.splitlines()
        try:
            tree = ast.parse(text)
        except SyntaxError as error:
            findings.append({"severity": "violation", "rule": f"syntax error: {error}", "file": path.name,
                             "line": error.lineno or 0, "code": ""})
            continue
        # Names bound to modules ("import torch.nn.functional as F", "from torch import nn"): assigning to their
        # attributes (F.cross_entropy = ..., T.WEIGHT_CAP = ...) patches code the grader also runs.
        module_names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                module_names |= {a.asname or a.name.split(".")[0] for a in node.names}
            elif isinstance(node, ast.ImportFrom) and node.module and node.module.split(".")[0] in ("torch", "lmenv",
                                                                                                    "numpy", "math"):
                module_names |= {a.asname or a.name for a in node.names}
        inner_names = {
            target.id
            for node in ast.walk(tree) if isinstance(node, ast.Assign)
            and _unwrapped(node.value)
            for target in node.targets if isinstance(target, ast.Name)
        }
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                if isinstance(node, ast.Import):
                    names = [a.name for a in node.names]
                else:  # "from torch.utils import cpp_extension" imports torch.utils.cpp_extension
                    names = [node.module or ""] + [f"{node.module}.{a.name}" for a in node.names if node.module]
                for name in names:
                    root = name.split(".")[0]
                    if any(name == b or name.startswith(b + ".") for b in BANNED_MODULES):
                        findings.append(_finding("violation", f"imports {name}", path, node, source))
                    elif root in REVIEW_MODULES:
                        findings.append(_finding("review", f"imports {name}", path, node, source))
                    elif root == "lmenv" and not any(name == a or name.startswith(a + ".") for a in ALLOWED_HARNESS):
                        findings.append(_finding("violation", f"imports harness internals ({name})", path, node, source))
            elif isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                for target in targets:
                    base = target.value if isinstance(target, ast.Subscript) else target  # costs[k] = ...
                    if isinstance(base, ast.Attribute) and base.attr in PROTECTED_ATTRIBUTES:
                        findings.append(_finding("violation", f"writes harness attribute .{base.attr}", path, node, source))
                    elif isinstance(target, ast.Attribute) and target.attr in ("tokens", "flops") and \
                            "budget" in ast.unparse(target.value):
                        findings.append(_finding("violation", f"writes the budget's .{target.attr}", path, node, source))
                    elif isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name) and \
                            target.value.id in module_names:
                        findings.append(_finding("violation", f"patches module attribute {target.value.id}.{target.attr}",
                                                 path, node, source))
            elif isinstance(node, ast.Call):
                func = node.func
                target = func.value if isinstance(func, ast.Attribute) and func.attr in ("forward", "__call__") else func
                if _unwrapped(target) or (
                    isinstance(target, ast.Name) and target.id in inner_names
                ):
                    findings.append(_finding("violation", "calls the unwrapped model (not charged to the budget)",
                                             path, node, source))
                if isinstance(func, ast.Attribute) and func.attr in MUTATING_METHODS and \
                        isinstance(func.value, ast.Attribute) and func.value.attr in PROTECTED_ATTRIBUTES:
                    findings.append(_finding("violation", f"modifies harness attribute .{func.value.attr}", path, node,
                                             source))
                if (isinstance(func, ast.Name) and func.id in REVIEW_BUILTINS) or (
                    isinstance(func, ast.Attribute) and func.attr in REVIEW_METHODS
                ):
                    name = func.id if isinstance(func, ast.Name) else func.attr
                    findings.append(_finding("review", f"calls {name}()", path, node, source))
            elif isinstance(node, ast.Attribute) and node.attr == "modules" and getattr(node.value, "id", "") == "sys":
                findings.append(_finding("violation", "touches sys.modules", path, node, source))
            elif isinstance(node, ast.Attribute) and node.attr in REVIEW_ATTRIBUTES:
                findings.append(_finding("review", f"uses .{node.attr}", path, node, source))
    return findings


def run_static_scan(directory: Path, report: CheckReport) -> None:
    report.static_findings = static_scan(directory)
    violations = [f for f in report.static_findings if f["severity"] == "violation"]
    if violations:
        report.failures.append(f"{len(violations)} static-scan violation(s): " +
                               "; ".join(f"{v['file']}:{v['line']} {v['rule']}" for v in violations[:5]))


def run_checks(submission: Path, vocab_size: int, device, model=None, text: torch.Tensor | None = None,
               recorded_calls: list[dict] | None = None) -> CheckReport:
    """All checks on a submission.

    ``model`` is the trained model if there is one (else a fresh build),
    ``text`` one SEQ_LEN window of real token ids, ``recorded_calls`` the
    call_costs of a training run (train.json).
    """

    from lmenv.harness import build

    report = CheckReport()
    run_static_scan(submission, report)
    _, dropped, oversized = submission_files(submission)
    report.static_findings += [{"severity": "review", "rule": "not kept by the grader (only *.py and config.json are)",
                                "file": name, "line": 0, "code": ""} for name in dropped]
    if oversized:
        report.failures.append("files over the size limit: " + ", ".join(oversized))
    try:
        if model is None:
            torch.manual_seed(0)
            model = build(submission, vocab_size, device)
    except Exception as error:  # the submission's code can fail in any way
        report.failures.append(f"building the model: {type(error).__name__}: {error}"[:500])
        return report
    check_state(model, report)
    report.failures.extend(interface_failures(model, vocab_size, device, perturb=False))
    ids = torch.randint(0, vocab_size, (2, 128), generator=torch.Generator().manual_seed(0)).to(device)
    try:
        caps, failures = cap_failures(model, vocab_size, device)
        report.weights, report.inference_flops_per_token = caps["weights"], caps["inference_flops_per_token"]
        report.failures.extend(failures)
        check_eval(model, ids, text, report)
        check_call_costs(model, recorded_calls or [], vocab_size, device, report)
    except Exception as error:
        report.failures.append(f"measuring the model: {type(error).__name__}: {error}"[:500])
    return report
