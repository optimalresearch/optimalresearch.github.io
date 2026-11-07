"""Mechanical checks and the metric for this environment. Single source of truth.

validate.py runs this on the dev text in /task/public (the agent can run it);
tests/grade.py runs it on the held-out texts (the verifier). Both call evaluate().

The reward is the steps-to-target score: a text's validation loss L is mapped
to the number of training tokens D(L) at which a plain transformer without a
weight cap (the oracle; same FLOPs per token, same data, same training budget)
reaches it, from a fit D(L) = (B / (L - E))^(1 / beta) through its own training
runs at several budgets. Score = D(L) / TRAIN_TOKENS, clamped to [0, 1], so the
fully trained oracle scores 1. Each text has its own curve (curve.json next to
its tokens). The reward is the mean over texts. A failed gate scores as the
unmodified starting point does (its steps score from reference.json), as the
agent's prompt states. The dev text has no curve for now, so the dev check
reports its loss only (README).
"""
from __future__ import annotations

import ast
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from lmenv import guard, task
from lmenv.checks import run_checks, submission_files
from lmenv.harness import load_trained
from lmenv.run import evaluate_run

TASK_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = Path("/workspace/submission")  # the original layout; Harbor collects it (task.toml)
PUBLIC_DIR = TASK_DIR / "public"
REQUIRED = ("model.py", "train.py", "config.json")


@dataclass
class Report:
    gates: dict[str, bool] = field(default_factory=dict)   # name -> passed
    notes: list[str] = field(default_factory=list)
    texts: dict[str, dict] = field(default_factory=dict)   # text -> loss, bits_per_byte, steps_score, gain_vs_starter
    starter_reward: float = 0.0   # what a failed submission scores: the starting point's mean steps score
    dropped_files: list[str] = field(default_factory=list)
    checks: dict = field(default_factory=dict)

    @property
    def gated(self) -> bool:
        return all(self.gates.values())

    @property
    def reward(self) -> float:
        scores = [t.get("steps_score") for t in self.texts.values()]
        if not self.gated:
            return self.starter_reward
        if not scores or any(s is None for s in scores):
            return 0.0
        return float(np.mean(scores))

    def fail(self, gate: str, note: str) -> None:
        self.gates[gate] = False
        self.notes.append(note)

    def as_dict(self) -> dict:
        return {"reward": self.reward, "starter_reward": self.starter_reward, "texts": self.texts, "gates": self.gates, "notes": self.notes,
                "dropped_files": self.dropped_files, "checks": self.checks}


def scale(name: str | None = None) -> dict:
    """The setting's knobs from /task/scales.toml (one setting, "full")."""
    scales = tomllib.loads((TASK_DIR / "scales.toml").read_text())
    name = name or os.environ.get("AR_SCALE", "full")
    if name not in scales:
        raise SystemExit(f"unknown AR_SCALE={name!r}; known: {sorted(scales)}")
    return {"name": name, **scales[name]}


def steps_score(loss: float, curve: dict) -> float:
    """Fraction of the oracle's training (curve["tokens"]) it needs to reach LOSS, clamped to [0, 1]."""
    if not math.isfinite(loss):
        return 0.0
    if loss <= curve["E"]:
        return 1.0
    tokens = (curve["B"] / (loss - curve["E"])) ** (1.0 / curve["beta"])
    return max(0.0, min(1.0, tokens / curve["tokens"]))


def budget_failures(record: dict) -> list[str]:
    """The training run's own record against the budget (the last call may overshoot it by one call)."""
    failures = []
    if record.get("tampering"):
        failures.append("the submission patched " + ", ".join(record["tampering"][:10]) + " during training")
    call_tokens, call_flops = record.get("largest_call", (0.0, 0.0))
    spare_tokens, spare_flops = max(0.01 * task.TRAIN_TOKENS, call_tokens), max(0.01 * task.TRAIN_FLOPS, call_flops)
    if record.get("used_tokens", 0) > task.TRAIN_TOKENS + spare_tokens:
        failures.append(f"training used {record['used_tokens']:.4g} tokens (budget {task.TRAIN_TOKENS:.4g})")
    if record.get("used_flops", 0) > task.TRAIN_FLOPS + spare_flops:
        failures.append(f"training used {record['used_flops']:.4g} forward FLOPs (budget {task.TRAIN_FLOPS:.4g})")
    # Reading allows one more call: a train.py may sample the next batch before the call that runs out.
    if record.get("corpus_passes", 0) > 1 + 2 * spare_tokens / task.CORPUS_TOKENS:
        failures.append(f"training read the corpus {record['corpus_passes']:.3g} times (one pass allowed)")
    if record.get("used_tokens", 0) > 0 and not record.get("call_costs"):
        failures.append("training used tokens but recorded no kinds of model call")
    return failures


def check_submission(output_dir: Path, report: Report) -> list[Path]:
    """Gates on the files themselves. Returns the files that are graded."""
    missing = [name for name in REQUIRED if not (output_dir / name).is_file()]
    report.gates["required_files_present"] = not missing
    if missing:
        report.notes.append("missing " + ", ".join(missing))
    root, escaping = output_dir.resolve(), []
    for path in output_dir.rglob("*"):
        if path.is_symlink() and not str(path.resolve()).startswith(str(root)):
            escaping.append(str(path))
    report.gates["no_escaping_symlinks"] = not escaping
    if escaping:
        report.notes.append("symlinks leave the output dir: " + ", ".join(escaping))
    kept, report.dropped_files, oversized = submission_files(output_dir)
    report.gates["graded_files_within_limits"] = not oversized
    if oversized:
        report.notes.append("over the size limit: " + ", ".join(oversized))
    named = forbidden_paths(kept)
    report.gates["no_outside_paths"] = not named
    if named:
        report.notes.append("names paths it may not read: " + ", ".join(named[:10]))
    return kept


def forbidden_paths(files: list[Path]) -> list[str]:
    """String constants in the graded files that name the validation text or absolute paths outside
    /task and /workspace (the submission may read only its training tokens and its own files)."""
    found = []
    for path in files:
        text = path.read_text(errors="replace")
        if path.suffix == ".py":
            try:
                strings = [n.value for n in ast.walk(ast.parse(text)) if isinstance(n, ast.Constant)
                           and isinstance(n.value, str)]
            except SyntaxError:
                continue  # the rule checks report it
        else:
            strings = [text]
        for s in strings:
            for hit in re.findall(r"val\.bin|(?<![\w.])/(?!task/|workspace/|tmp/|dev/null)[A-Za-z_][\w.-]*/", s):
                found.append(f"{path.name}: {hit!r}")
    return found


def evaluate(output_dir: Path, texts: dict[str, Path], work: Path, checks_only: bool = False) -> Report:
    """The whole pipeline: file gates, rule checks, then per text one retraining from scratch (in a child
    process that sees only the training tokens), the checks again on the trained model, the budget record,
    and the validation loss mapped to the steps score. Same call in validate.py and tests/grade.py."""
    report = Report()
    floors = []
    for data in texts.values():
        curve = data / "curve.json"
        if curve.exists():
            starter = json.loads((data / "reference.json").read_text())["loss"]
            floors.append(steps_score(starter, json.loads(curve.read_text())))
    report.starter_reward = float(np.mean(floors)) if floors and len(floors) == len(texts) else 0.0
    untouched = guard.fingerprint()  # before any submission code is imported here
    kept = check_submission(output_dir, report)
    if not report.gated:
        return report
    work.mkdir(parents=True, exist_ok=True)
    submission = work / "submission"  # a frozen copy of the graded files
    shutil.rmtree(submission, ignore_errors=True)
    submission.mkdir()
    for path in kept:
        shutil.copy(path, submission / path.name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def untampered(step: str) -> bool:
        patched = guard.changes(untouched)
        report.gates["untampered"] = not patched
        if patched:
            report.notes.append(f"{step}: the submission patched {', '.join(patched[:10])}")
        return not patched

    first = next(iter(texts.values()))
    meta = json.loads((first / "meta.json").read_text())
    val = np.memmap(first / "val.bin", dtype=np.uint16, mode="r")
    sample = torch.from_numpy(val[-task.SEQ_LEN:].astype(np.int64))[None].to(device)
    checks = run_checks(submission, meta["vocab_size"], device, text=sample)
    report.checks["before_training"] = checks.to_dict()
    report.gates["rule_checks"] = checks.passed
    report.notes += checks.failures
    if not untampered("checks") or not checks.passed or checks_only:
        return report

    env = {**os.environ, "PYTHONPATH": str(TASK_DIR), "PYTHONDONTWRITEBYTECODE": "1"}
    for name, data in texts.items():
        meta = json.loads((data / "meta.json").read_text())
        train_data = work / f"train_data_{name}"  # only the training text, for the training process
        train_data.mkdir(exist_ok=True)
        for file in ("train.bin", "meta.json", "tokenizer.json"):
            link = train_data / file
            if not link.exists():
                link.symlink_to((data / file).resolve())
        run = work / f"run_{name}"
        subprocess.run([sys.executable, "-m", "lmenv.run", "_train", "--submission", str(submission),
                        "--data", str(train_data), "--out", str(run), "--seed", "0"], env=env, cwd=TASK_DIR)
        record = json.loads((run / "train.json").read_text()) if (run / "train.json").exists() else {}
        if record.get("status") != "complete" or not (run / "model.pt").exists():
            report.fail(f"trained_{name}", f"{name}: {record.get('error') or 'training did not finish'}"[:2000])
            return report
        report.gates[f"trained_{name}"] = True
        over = budget_failures(record)
        report.gates[f"budget_{name}"] = not over
        if over:
            report.notes += [f"{name}: {f}" for f in over]
            return report
        if name == next(iter(texts)):
            trained = load_trained(submission, run, meta["vocab_size"], device)
            after = run_checks(submission, meta["vocab_size"], device, model=trained, text=sample,
                               recorded_calls=record.get("call_costs", []))
            del trained
            report.checks["after_training"] = after.to_dict()
            report.gates["rule_checks_after_training"] = after.passed
            report.notes += [f"after training: {f}" for f in after.failures]
            if not untampered("checks after training") or not after.passed:
                return report
        result = evaluate_run(submission, run, data, device)
        if not untampered(f"evaluation on {name}"):
            return report
        starter = json.loads((data / "reference.json").read_text())["loss"]
        curve_file = data / "curve.json"
        curve = json.loads(curve_file.read_text()) if curve_file.exists() else None
        report.texts[name] = {"loss": result["loss"], "bits_per_byte": result.get("bits_per_byte"),
                              "gain_vs_starter": starter - result["loss"],
                              "steps_score": steps_score(result["loss"], curve) if curve else None}
        if curve is None:
            report.notes.append(f"{name}: no curve.json, so no steps score (loss only; reward 0)")
    return report
