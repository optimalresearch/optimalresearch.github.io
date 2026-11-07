#!/usr/bin/env python3
"""Print the session's budget (the constants in lmenv/task.py and scales.toml)."""
import sys
import tomllib
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from lmenv import task  # noqa: E402  (imports no torch)

sc = tomllib.loads((HERE / "scales.toml").read_text())["full"]
print(f"session: {sc['session_wall_sec'] / 3600:g} h on {sc['gpu']}; time left: bash /workspace/timer.sh")
print(f"training budget per run: {task.TRAIN_TOKENS:,} tokens, one pass over the first {task.CORPUS_TOKENS:,} "
      f"tokens of the training text, forward FLOPs at most {task.INFERENCE_FLOP_CAP / 1e6:.0f}M per token on average")
print(f"caps: {task.WEIGHT_CAP:,} unique weights (plus one embedding table of width <= {task.EXEMPT_TABLE_WIDTH}), "
      f"{task.INFERENCE_FLOP_CAP / 1e6:.0f}M inference FLOPs per token on a {task.SEQ_LEN}-token window")
