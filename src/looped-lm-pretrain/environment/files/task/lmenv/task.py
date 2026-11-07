"""Task constants shared by the agent's tools and the grader.

Changing a value here changes the task; the grader always uses its own clean
copy of this file.
"""

import json
import os
from pathlib import Path

# Unique weights: every parameter and buffer, tied tensors counted once,
# except one table of shape (vocab, width) with width <= EXEMPT_TABLE_WIDTH
# (the token embedding, or the output head tied to it).
WEIGHT_CAP = 3_000_000
EXEMPT_TABLE_WIDTH = 512
# FLOPs per token of one forward pass over a SEQ_LEN window (eval mode,
# attention included), measured with torch's FLOP counter. 120M (10x the
# starter) is about what one A100 trains on TRAIN_TOKENS within an hour.
INFERENCE_FLOP_CAP = 120e6
# Training: every token passed through the model counts (TRAIN_TOKENS in
# all, one pass over the CORPUS_TOKENS training text, each token seen once),
# and the forward FLOPs of all training calls together may not exceed
# INFERENCE_FLOP_CAP per token on average (TRAIN_FLOPS). Backward passes are
# not counted.
TRAIN_TOKENS = 250_000_000
CORPUS_TOKENS = 250_000_000
TRAIN_FLOPS = INFERENCE_FLOP_CAP * TRAIN_TOKENS
# Training reads only the first CORPUS_TOKENS tokens of train.bin, in windows
# of at most SEQ_LEN tokens; evaluation uses SEQ_LEN.
SEQ_LEN = 512
EVAL_TOKENS = 5_000_000
# Grading retrains a submission this many times on each test text (calibration uses 3).
SEEDS_PER_SUBMISSION = 1
# At most this many distinct ways of calling the model in training (each new
# one is measured once with the FLOP counter).
MAX_CALL_KINDS = 256

# A sibling configuration of the task (configs/*.json) replaces some of the values above: from
# task_config.json next to this file (harbor/build.py --task-config copies it into the agent's and the
# verifier's copies of lmenv), or from the file LMENV_TASK_CONFIG names (cluster runs).
CONFIGURABLE = ("WEIGHT_CAP", "EXEMPT_TABLE_WIDTH", "INFERENCE_FLOP_CAP", "TRAIN_TOKENS", "CORPUS_TOKENS",
                "SEQ_LEN", "EVAL_TOKENS", "SEEDS_PER_SUBMISSION")
CONFIG_FILE = os.environ.get("LMENV_TASK_CONFIG") or Path(__file__).with_name("task_config.json")
CONFIG = json.loads(Path(CONFIG_FILE).read_text()) if Path(CONFIG_FILE).exists() else {}
if unknown := set(CONFIG) - set(CONFIGURABLE) - {"name", "note"}:
    raise ValueError(f"{CONFIG_FILE}: unknown keys {sorted(unknown)}")
globals().update({key: type(globals()[key])(CONFIG[key]) for key in CONFIGURABLE if key in CONFIG})
TRAIN_FLOPS = INFERENCE_FLOP_CAP * TRAIN_TOKENS

# Smoke tests of the pipeline only (tiny budget and data); never set in a real run.
if os.environ.get("LMENV_SMOKE"):
    TRAIN_TOKENS, CORPUS_TOKENS, EVAL_TOKENS = 1_000_000, 1_000_000, 100_000
    TRAIN_FLOPS = INFERENCE_FLOP_CAP * TRAIN_TOKENS
