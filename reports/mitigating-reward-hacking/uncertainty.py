"""Approximate 95% evaluation intervals from the archived aggregate metrics.

The archive has means, not individual evaluation outcomes or reward second
moments. These are NOT empirical bootstrap intervals. The grader evaluates
1,000 sampled completions per environment, with three equally weighted envs.

For score Z = R - 10 H, clean reward C = R (1-H), and 0 <= R <= 1:
    E[Z^2] = E[R^2] + 100 E[H] - 20 E[RH]
           <= E[R] + 100 E[H] - 20 (E[R] - E[C]).
This bounds the empirical second moment; use it for an upper estimate of the
standard error and a normal-approximation interval. It is exact at the moment
level for binary rewards, conservative for fractional rewards. Coverage remains
approximate, especially with rare events. Fixed 500/500 prompt-kind allocation
is handled conservatively by pooling within each environment.

For Pass@k, propagate simultaneous Bonferroni-adjusted run intervals through
the monotone expected-maximum statistic. This permits unknown dependence across
runs evaluated on the same prompts. It does not resample agent identities and
does not estimate training-seed, run-selection, or new-environment uncertainty.

GRAFT's rounded reference means do not include a verified evaluation sample
count, so no interval is inferred for that reference.
"""

from itertools import combinations
from math import sqrt
from statistics import NormalDist

EPISODES_PER_ENV = 1000
WEIGHT = 10.0


def score_se_upper(scope: dict) -> float:
    envs = [v for k, v in scope.items() if k != "mean"]
    if not envs:
        raise ValueError("Per-environment metrics are required for uncertainty")
    variance = 0.0
    for env in envs:
        if env.get("penalized"):
            raise ValueError("A grading penalty is not a sampled model score")
        reward, hack, clean = (env[k] for k in ("true_reward", "hack_rate", "clean_reward"))
        if not (0 <= clean <= reward + 1e-10 <= 1 + 1e-10 and
                0 <= reward - clean <= hack + 1e-10):
            raise ValueError("Inconsistent joint reward/hack metrics")
        second = reward + WEIGHT**2 * hack - 2 * WEIGHT * (reward - clean)
        mean = reward - WEIGHT * hack
        variance += max(0.0, second - mean**2) / (EPISODES_PER_ENV - 1)
    return sqrt(variance / len(envs)**2)


def score_interval(scope: dict, alpha: float = .05) -> tuple[float, float]:
    score = scope["mean"]["score"]
    margin = NormalDist().inv_cdf(1 - alpha / 2) * score_se_upper(scope)
    return max(-WEIGHT, score - margin), min(1.0, score + margin)


def expected_best(scores: list[float], k: int) -> float:
    values = [max(subset) for subset in combinations(scores, k)]
    if not values:
        raise ValueError("k must be between one and the run count")
    return sum(values) / len(values)


def pass_interval(runs: list[dict], k: int) -> tuple[float, float]:
    intervals = [score_interval(r["held_out"], alpha=.05 / len(runs)) for r in runs]
    return (expected_best([lo for lo, _ in intervals], k),
            expected_best([hi for _, hi in intervals], k))


def evaluation_noise_band(runs: list[dict]) -> list[tuple[int, float, float]]:
    """Estimate pass@k gains from evaluation noise alone, conditional on these runs.

    Match the looping report's equal-quality Gaussian null and 5–95% simulation
    band, anchored at observed pass@1. Unlike that report's repeated gradings,
    these archives provide only aggregate evaluation metrics. Use the RMS of
    score_se_upper across runs as a common noise scale; this is an approximate
    upper estimate of completion-sampling noise, not training-seed variability.
    """
    import math
    import numpy as np

    n = len(runs)
    sd = sqrt(sum(score_se_upper(r["held_out"])**2 for r in runs) / n)
    anchor = sum(r["held_out"]["mean"]["score"] for r in runs) / n
    null = np.sort(np.random.default_rng(0).normal(0, sd, (200_000, n)), axis=1)
    means = null.mean(axis=1)
    band = [(1, anchor, anchor)]
    for k in range(2, n + 1):
        weights = np.array([math.comb(i, k - 1) for i in range(n)]) / math.comb(n, k)
        lo, hi = np.quantile(null @ weights - means, [.05, .95]) + anchor
        band.append((k, float(lo), float(hi)))
    return band
