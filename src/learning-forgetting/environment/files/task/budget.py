#!/usr/bin/env python3
"""Print the active scale's budget. The agent is told to run this."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import contract  # noqa: E402

sc = contract.scale()
print(f"scale: {sc['name']}")
print(f"session wall-clock budget: {sc['session_wall_sec']} s ({sc['session_wall_sec']/3600:.1f} h)")
print(f"evaluation wall-clock budget (verifier timer, model load + all benchmarks): {sc['eval_wall_sec']} s")
print(f"verifier set sizes: about 4,000 held-out fact probes (World Cup, astronomy and news facts), GSM8K test {sc['n_gsm8k']}, hidden knowledge {sc['n_retention'] or 'all'} per set (four sets)")
rent = Path("/staging/rent/budget.json")
if sc.get("gpu_native"):
    print(f"GPU: this sandbox has one 80 GB H100 for the whole session ({sc['session_wall_sec']/3600:.1f} GPU-hours); "
          "there is no rental. Public check: bash /task/check.sh")
elif rent.exists():
    b = json.loads(rent.read_text())
    print(f"GPU rental (`rent`): {b['gpu_hours_left']} GPU-hours left of {b['gpu_hours_total']}, "
          f"cap {b['max_concurrent_gpus']} concurrent GPUs, {b['max_job_min']} min per job, aliases {list(b['aliases'])}")
else:
    print("GPU rental: not available in this sandbox (no bridge)")
