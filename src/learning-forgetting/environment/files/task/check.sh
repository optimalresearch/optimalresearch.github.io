#!/bin/bash
# The check: the verifier's evaluation of /workspace/output/model.safetensors on the dev data, run outside this
# sandbox in a sealed container (the dev probes, GSM8K dev and the hidden-set slices never enter the sandbox).
# Writes the aggregate report to /workspace/check/report.json and prints its headline.
# - Sandbox with its own GPU (python3 /task/budget.py says so): `bash /task/check.sh`. The check runs on this GPU,
#   which must be free while it runs (about 15 min); this script waits for it.
# - Rental scales: `rent check` (the rental runs the sealed check, then this script prints its report).
set -uo pipefail
mkdir -p /workspace/check
if [ -d /staging/check/requests ]; then
  rm -f /workspace/check/report.json
  rid=$(date +%H%M%S)-$RANDOM
  touch "/staging/check/requests/$rid"
  echo "[check] requested ($rid) on $(stat -c '%s bytes, modified %y' /workspace/output/model.safetensors 2>/dev/null || echo 'no model.safetensors')"
  echo "[check] keep the GPU free until it is done (about 15 min)"
  for _ in $(seq 1 1300); do [ -f "/staging/check/done/$rid" ] && break; sleep 3; done
  [ -f "/staging/check/done/$rid" ] || { echo "[check] no result after 65 min"; exit 1; }
fi
if [ ! -f /workspace/check/report.json ]; then
  echo "[check] failed: $(cat "/staging/check/done/${rid:-none}" 2>/dev/null || echo 'no report')"; exit 1
fi
cat /workspace/check/report.json
python3 - <<'PY'
import json
r = json.load(open("/workspace/check/report.json"))
print(f"\ncheck score {r['score']:.3f}  (learning² × retention: learning {r['learning']:.3f} on the dev facts' probes; "
      f"retention {r['retention']:.3f} on GSM8K dev and the hidden-knowledge slices)  gates passed: {all(r['gates'].values())}")
raise SystemExit(0 if all(r["gates"].values()) else 1)
PY
