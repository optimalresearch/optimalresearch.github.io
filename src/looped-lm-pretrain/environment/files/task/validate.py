#!/usr/bin/env python3
"""Self-check: the verifier's checks, retraining and evaluation, on the dev text.

    python3 /task/validate.py [--output-dir /workspace/output] [--checks-only] [--work /workspace/validate]

Without --checks-only this retrains the submission from scratch on the dev text with the full
budget (as long as one real training run) and prints the dev validation loss. The reward itself
needs the oracle curves of the held-out texts; the dev text has no curve for now, so the dev check
reports loss (and the gain over the starting point), not the score.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import contract  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output-dir", type=Path, default=contract.OUTPUT_DIR)
    ap.add_argument("--checks-only", action="store_true", help="file gates and rule checks only, no training")
    ap.add_argument("--work", type=Path, default=Path("/workspace/validate"))
    args = ap.parse_args()
    sc = contract.scale()
    report = contract.evaluate(args.output_dir, {"dev": contract.PUBLIC_DIR}, args.work, checks_only=args.checks_only)
    print(json.dumps({k: v for k, v in report.as_dict().items() if k not in ("checks", "reward")}, indent=2))
    failed = [name for name, ok in report.gates.items() if not ok]
    if failed:
        print(f"\nscale={sc['name']}  FAILED gates: {', '.join(failed)} (scores as the starting point)")
    elif args.checks_only:
        print(f"\nscale={sc['name']}  all checks passed (no training run)")
    else:
        dev = report.texts["dev"]
        score = "" if dev["steps_score"] is None else f"  dev score {dev['steps_score']:.3f}"
        print(f"\nscale={sc['name']}  all gates passed  dev loss {dev['loss']:.4f} "
              f"({dev['gain_vs_starter']:+.4f} nats vs the starting point){score}")
    return 0 if report.gated else 1


if __name__ == "__main__":
    raise SystemExit(main())
