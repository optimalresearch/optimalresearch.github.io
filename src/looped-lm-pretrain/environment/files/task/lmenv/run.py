"""Command-line tools for the agent (the grader uses the same functions).

    python -m lmenv.run check --submission submission
    python -m lmenv.run train --submission submission --data data/lm --out runs/NAME --seed 0 [--tokens N]
    python -m lmenv.run eval --submission submission --run runs/NAME --data data/lm
    python -m lmenv.run score --submission submission --runs runs/A runs/B runs/C
    python -m lmenv.run snapshot --note "what changed and why" [--runs runs/A ...]

``check`` runs the rule checks the grader runs (caps, causality, static
scan). ``train`` trains one seed and evaluates it on the validation text.
A smaller ``--tokens`` scales the token and FLOP budgets down together for a
quick proxy run (not comparable with the starter's numbers). ``score``
averages several full-budget runs the way the grader does. ``snapshot``
saves a copy of submission/ with a note (not graded).
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch

from lmenv import task
from lmenv.harness import evaluate, load_trained, token_bytes, train_run


def evaluate_run(submission: Path, run: Path, data: Path, device, log=print) -> dict:
    """Loss on DATA/val.bin (nats per token, and bits per byte); writes RUN/eval.json."""

    meta = json.loads((data / "meta.json").read_text())
    val = np.memmap(data / "val.bin", dtype=np.uint16, mode="r")[: task.EVAL_TOKENS]
    model = load_trained(submission, run, meta["vocab_size"], device)
    result = evaluate(model, val, device, token_bytes(data / "tokenizer.json", meta["eot"]))
    (run / "eval.json").write_text(json.dumps(result, indent=2))
    log(f"validation loss {result['loss']:.4f} nats per token, {result['bits_per_byte']:.4f} bits per byte")
    return result


def summarize(evals: list[dict], reference: dict) -> dict:
    """Average per-seed evaluations and score them.

    reference: the starter's mean loss on the same text ("loss"), which scores 0.
    """

    loss = float(np.mean([e["loss"] for e in evals]))
    passed = bool(np.isfinite(loss))
    return {
        "seeds": len(evals),
        "loss": loss,
        "loss_by_seed": [e["loss"] for e in evals],
        "bits_per_byte": float(np.mean([e["bits_per_byte"] for e in evals])),
        "passed": passed,
        # Lower loss than the starter's, in nats; a failure scores as the starter does.
        "score": reference["loss"] - loss if passed else 0.0,
        "gain_vs_starter": reference["loss"] - loss if passed else 0.0,
    }


def snapshot(submission: Path, note: str, runs: list[str] | None) -> Path:
    """Copy the submission's code into snapshots/ next to it, with a note and a timestamp."""

    root = submission.resolve().parent / "snapshots"
    root.mkdir(parents=True, exist_ok=True)
    index = 1 + max((int(p.name) for p in root.iterdir() if p.name.isdigit()), default=0)
    target = root / f"{index:03d}"
    shutil.copytree(submission, target, ignore=shutil.ignore_patterns("__pycache__", "*.pt"))
    meta = {"index": index, "note": note, "created_unix": time.time()}
    try:
        meta["episode_hours"] = round((time.time() - int(Path("/tmp/timer_start").read_text())) / 3600, 2)
    except (OSError, ValueError):
        pass
    if runs:
        meta["runs"] = runs
    (target / "snapshot.json").write_text(json.dumps(meta, indent=2))
    return target


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m lmenv.run")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("check")
    p.add_argument("--submission", default="submission")
    p.add_argument("--data", default="data/lm")
    for name in ("train", "_train"):
        p = sub.add_parser(name)
        p.add_argument("--submission", default="submission")
        p.add_argument("--data", default="data/lm")
        p.add_argument("--out", required=True)
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--tokens", type=float, default=task.TRAIN_TOKENS,
                       help="training tokens (smaller = quick proxy run, not comparable with the starter)")
        p.add_argument("--corpus", type=float, default=task.CORPUS_TOKENS, help=argparse.SUPPRESS)
    p = sub.add_parser("eval")
    p.add_argument("--submission", default="submission")
    p.add_argument("--run", required=True)
    p.add_argument("--data", default="data/lm")
    p = sub.add_parser("score")
    p.add_argument("--submission", default="submission")
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--data", default="data/lm")
    p = sub.add_parser("snapshot")
    p.add_argument("--submission", default="submission")
    p.add_argument("--note", required=True, help="what changed and why")
    p.add_argument("--runs", nargs="*", help="run folders that tested this version")
    args = parser.parse_args(argv)

    if args.cmd == "snapshot":
        print(f"saved snapshot {snapshot(Path(args.submission), args.note, args.runs)}")
        return 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    submission, data = Path(args.submission), Path(args.data)
    if args.cmd == "check":
        from lmenv.checks import run_checks

        vocab = json.loads((data / "meta.json").read_text())["vocab_size"]
        val = np.memmap(data / "val.bin", dtype=np.uint16, mode="r")[: task.SEQ_LEN]
        text = torch.from_numpy(val.astype(np.int64))[None].to(device)
        report = run_checks(submission, vocab, device, text=text)
        print(json.dumps(report.to_dict(), indent=2))
        print("PASSED" if report.passed else "FAILED:\n  " + "\n  ".join(report.failures))
        return 0 if report.passed else 1
    if args.cmd in ("train", "_train"):
        run = Path(args.out)
        record = train_run(submission, data, run, args.seed, device, tokens=args.tokens, corpus=int(args.corpus))
        if record["status"] == "error":
            print(f"training failed: {record['error']}")
            return 1
        if args.cmd == "train":
            evaluate_run(submission, run, data, device)
        return 0
    if args.cmd == "eval":
        evaluate_run(submission, Path(args.run), data, device)
        return 0
    if args.cmd == "score":
        evals = []
        for r in args.runs:
            path = Path(r) / "eval.json"
            if not path.exists():
                evaluate_run(submission, Path(r), data, device)
            evals.append(json.loads(path.read_text()))
        reference = json.loads((data / "reference.json").read_text())
        print(json.dumps(summarize(evals, reference), indent=2))
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
