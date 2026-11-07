#!/bin/bash
# The original environment's /workspace: submission/ (the starting point), data/lm/ (the dev text),
# runs/, snapshots/ and timer.sh. Run at image build (Dockerfile). Idempotent.
set -e
W=/workspace
mkdir -p "$W/submission" "$W/runs" "$W/snapshots" "$W/data"
[ -f "$W/submission/model.py" ] || cp /task/starter/model.py /task/starter/train.py /task/starter/config.json "$W/submission/"
[ -e "$W/data/lm" ] || ln -s /task/public "$W/data/lm"
cp /task/timer.sh "$W/timer.sh"
