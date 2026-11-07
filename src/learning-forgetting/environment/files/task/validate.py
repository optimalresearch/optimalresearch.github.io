#!/usr/bin/env python3
"""Self-check of the submission files: the verifier's file gates and the weight audit, on the CPU.

    python3 /task/validate.py

The evaluation itself (learning on the dev facts, retention on GSM8K dev and on slices of the hidden knowledge
sets) is the check, `/task/check.sh`, which runs outside this sandbox in a sealed container: see CONTRACT.md,
Development.
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
    args = ap.parse_args()
    report = contract.Report()
    contract.check_submission(args.output_dir, report)
    print(json.dumps({"gates": report.gates, "notes": report.notes}, indent=2))
    print(f"\nfile gates and weight audit passed: {report.gated}  (the evaluation is the check: bash /task/check.sh"
          " or rent check)")
    return 0 if report.gated else 1


if __name__ == "__main__":
    raise SystemExit(main())
