"""Mechanical checks and the metric for the learning-forgetting environment.

Single source of truth: validate.py / check.sh run this on the public data (the
agent can run it), tests/grade.py runs it on the private data (the verifier).
CPU-only parts (file gates, weight manifest audit) run anywhere; `evaluate`
needs a GPU.
"""
from __future__ import annotations

import json
import math
import os
import re
import tempfile
import time
import tomllib
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

OUTPUT_DIR = Path("/workspace/output")
MODEL_FILE = "model.safetensors"
METHOD_FILE = "METHOD.md"
BASE_MODEL_DIR = Path(os.environ.get("AR_BASE_MODEL", "/task/models/Qwen3.5-4B"))
MANIFEST = Path(__file__).resolve().parent / "eval" / "weight_manifest.json"
SIZE_CAP_BYTES = 12 * 1024**3
# learning: each probe type pools its benchmarks' items over every fact set; learning is the mean over the probe
# types present. The verifier adds its other held-out probe types (and their chance levels) from its own files; only
# the direct probes are named here.
LEARNED = {
    "direct": ("facts", "facts_astro", "facts_news"),   # held-out paraphrases of each fact's question
}
CHANCE: dict[str, float] = {}   # probe type -> chance accuracy, corrected to 0 (acc - c) / (1 - c)
# the dev facts' probes (what the sealed check scores learning on; not in the sandbox) under the same names with this
# prefix: the verifier runs them too and reports their learning, but scores only the other facts
DEV = "dev_"
GSM8K_SUFFIX = ("\n\nSolve the problem step by step, then end your reply with a line "
                "of the form 'Answer: <number>'.")

# name -> (kind, max_new_tokens). kind decides the prompt and the scorer.
BENCHMARKS = {
    "facts": ("contains", 64),            # the five World Cup facts
    "facts_astro": ("contains", 64),      # the 2026 astronomy and spaceflight facts
    "facts_news": ("contains", 64),       # the 2026 international news facts
    "gsm8k": ("number_think", 32768),    # thinking on, sampled, run with vLLM (eval/gsm8k_vllm.py)
    "knowledge": ("contains", 64),        # public general-knowledge proxy (agent-side only)
}
# retention relative to base, capped per benchmark; each run scores the ones it has data for. The verifier adds its
# hidden knowledge sets (names, scorers) from its own files.
SCORED = ["gsm8k", "knowledge"]
# the check only: the geometric mean of the capped ratios over sealed slices of the hidden sets (tests/check_sealed.py). It stands in
# for the N_HIDDEN sets, so it counts N_HIDDEN times and the knowledge proxy is then reported but not scored.
HIDDEN_SLICE = "hidden_slice"
N_HIDDEN = 4


@dataclass
class Report:
    gates: dict[str, bool] = field(default_factory=dict)       # name -> passed
    notes: list[str] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)    # benchmark -> accuracy
    baselines: dict[str, float] = field(default_factory=dict)  # benchmark -> base model accuracy
    counts: dict[str, list[int]] = field(default_factory=dict) # benchmark -> [correct, total]
    eval_wall_sec: float = 0.0

    @property
    def gated(self) -> bool:
        return all(self.gates.values())

    def probe_accuracy(self, prefix: str = "") -> dict[str, float]:
        """Accuracy per probe type, pooled over the fact sets present (chance-corrected where CHANCE says)."""
        out = {}
        for probe, names in LEARNED.items():
            ok = sum(self.counts[prefix + n][0] for n in names if prefix + n in self.counts)
            n = sum(self.counts[prefix + n][1] for n in names if prefix + n in self.counts)
            if n:
                c = CHANCE.get(probe, 0.0)
                out[probe] = max(0.0, (ok / n - c) / (1 - c))
        return out

    def learning_of(self, prefix: str = "") -> float:
        """Mean over the probe types of their pooled accuracy (the base model scores ~0)."""
        parts = list(self.probe_accuracy(prefix).values())
        return sum(parts) / len(parts) if parts else 0.0

    @property
    def learning(self) -> float:
        return self.learning_of()

    @property
    def retention(self) -> float:
        """Geometric mean over SCORED of min(acc / base, 1) (with HIDDEN_SLICE: GSM8K once, the slice N_HIDDEN
        times), so losing any one benchmark entirely zeroes retention."""
        if HIDDEN_SLICE in self.metrics:
            parts = [retention_ratio(self.metrics["gsm8k"], self.baselines["gsm8k"]), *[self.metrics[HIDDEN_SLICE]] * N_HIDDEN]
        else:
            parts = [retention_ratio(self.metrics[b], self.baselines[b]) for b in SCORED if b in self.metrics]
        return geomean(parts)

    @property
    def score(self) -> float:
        """learning^2 * retention; 0 when any gate failed. Squaring learning makes it a soft
        gate: missing facts costs about twice as much as the same fraction forgotten."""
        if not self.gated:
            return 0.0
        return self.learning ** 2 * self.retention

    def as_dict(self) -> dict:
        dev = {"dev_learning": self.learning_of(DEV), "dev_learning_by_probe": self.probe_accuracy(DEV)} \
            if any(k.startswith(DEV) for k in self.counts) else {}
        return {"reward": self.score, "learning": self.learning, "retention": self.retention,
                "learning_by_probe": self.probe_accuracy(), **dev,
                "metrics": self.metrics, "baselines": self.baselines,
                "counts": self.counts, "eval_wall_sec": self.eval_wall_sec,
                "gates": self.gates, "notes": self.notes}


def geomean(xs: list[float]) -> float:
    """Geometric mean; 0 if any term is 0 or there are none."""
    return math.prod(xs) ** (1 / len(xs)) if xs else 0.0


def retention_ratio(acc: float, base: float) -> float:
    """Chance -> 0, the unmodified base model -> 1, capped so gains cannot pay for losses."""
    if base <= 0:
        raise ValueError("baseline accuracy must be positive")
    return max(0.0, min(1.0, acc / base))


def scale(name: str | None = None) -> dict:
    """The active scale's knobs from /task/scales.toml (AR_SCALE, default dev)."""
    scales = tomllib.loads((Path(__file__).resolve().parent / "scales.toml").read_text())
    name = name or os.environ.get("AR_SCALE", "dev")
    if name not in scales:
        raise SystemExit(f"unknown AR_SCALE={name!r}; known: {sorted(scales)}")
    return {"name": name, **scales[name]}


# ---------------------------------------------------------------- CPU gates

def check_submission(output_dir: Path, report: Report) -> None:
    """File gates and the weight audit. Needs no GPU and does not load tensors."""
    model = output_dir / MODEL_FILE
    report.gates["model_file_present"] = model.is_file() and not model.is_symlink()
    if not report.gates["model_file_present"]:
        report.notes.append(f"missing regular file {model}")
    report.gates["method_present"] = (output_dir / METHOD_FILE).is_file()
    if not report.gates["method_present"]:
        report.notes.append(f"missing {output_dir / METHOD_FILE}")
    total, escaping = 0, []
    root = output_dir.resolve()
    for p in output_dir.rglob("*"):
        if p.is_symlink() and not str(p.resolve()).startswith(str(root)):
            escaping.append(str(p))
        if p.is_file():
            total += p.stat().st_size
    report.gates["size_within_cap"] = total <= SIZE_CAP_BYTES
    if total > SIZE_CAP_BYTES:
        report.notes.append(f"output dir is {total} bytes, cap is {SIZE_CAP_BYTES}")
    report.gates["no_escaping_symlinks"] = not escaping
    if escaping:
        report.notes.append("symlinks leave the output dir: " + ", ".join(escaping))
    if report.gates["model_file_present"]:
        check_manifest(model, report)


def check_manifest(model: Path, report: Report) -> None:
    """Every tensor of the base checkpoint, with the same name, shape and dtype, and nothing else."""
    from safetensors import safe_open
    want = json.loads(MANIFEST.read_text())["tensors"]
    problems = []
    try:
        with safe_open(str(model), "pt") as f:
            keys = set(f.keys())
            for k in sorted(keys - set(want))[:5]:
                problems.append(f"unexpected tensor {k}")
            for k in sorted(set(want) - keys)[:5]:
                problems.append(f"missing tensor {k}")
            for k in keys & set(want):
                s = f.get_slice(k)
                if list(s.get_shape()) != want[k]["shape"] or s.get_dtype() != want[k]["dtype"]:
                    problems.append(f"{k}: {s.get_dtype()} {list(s.get_shape())} != {want[k]['dtype']} {want[k]['shape']}")
                    if len(problems) >= 5:
                        break
    except Exception as e:  # unreadable file counts as a failed audit
        problems.append(f"cannot read safetensors: {e}")
    report.gates["weights_match_manifest"] = not problems
    report.notes.extend(problems)


# ---------------------------------------------------------------- GPU evaluation

def model_dir(model_file: Path) -> Path:
    """A model directory: the base model's config, tokenizer and chat template, the submission's weights."""
    tmp = Path(tempfile.mkdtemp(prefix="submission-"))
    for p in BASE_MODEL_DIR.iterdir():
        if not p.name.startswith("model.safetensors"):
            (tmp / p.name).symlink_to(p.resolve())
    (tmp / "model.safetensors").symlink_to(model_file.resolve())
    weight_map = {k: "model.safetensors" for k in json.loads(MANIFEST.read_text())["tensors"]}
    (tmp / "model.safetensors.index.json").write_text(json.dumps({"metadata": {}, "weight_map": weight_map}))
    return tmp


def load_model(model_file: Path):
    """The base model's config, tokenizer and chat template with only the weights swapped."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tmp = model_dir(model_file)
    tok = AutoTokenizer.from_pretrained(str(tmp))
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(str(tmp), dtype=torch.bfloat16).to("cuda").eval()
    return model, tok


VLLM_PYTHON = os.environ.get("AR_VLLM_PYTHON", "/opt/vllm/bin/python3")


def run_gsm8k_vllm(mdir: Path, items: Path, limit: int | None, out_dir: Path | None) -> tuple[int, int]:
    """GSM8K with thinking on, sampled as the model card recommends; see eval/gsm8k_vllm.py."""
    import subprocess
    out = (out_dir or Path(tempfile.mkdtemp(prefix="gsm8k-"))) / "gsm8k.jsonl"
    cmd = [VLLM_PYTHON, str(Path(__file__).resolve().parent / "eval" / "gsm8k_vllm.py"),
           "--model-dir", str(mdir), "--items", str(items), "--out", str(out)] + (["--limit", str(limit)] if limit else [])
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"gsm8k_vllm.py failed ({r.returncode}):\n{r.stderr[-3000:]}")
    # vLLM logs to stdout too (its shutdown lines come after the result)
    res = next(json.loads(l) for l in reversed(r.stdout.splitlines()) if l.startswith('{"correct"'))
    return res["correct"], res["n"]


def all_finite(model) -> bool:
    import torch
    return all(torch.isfinite(p).all().item() for p in model.parameters())


def generate(model, tok, prompts: list[str], max_new_tokens: int, batch_size: int) -> list[str]:
    """No system prompt, thinking disabled, greedy."""
    import torch
    outs: list[str] = []
    with torch.no_grad():
        for i in range(0, len(prompts), batch_size):
            texts = [tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False,
                                             add_generation_prompt=True, enable_thinking=False)
                     for p in prompts[i:i + batch_size]]
            enc = tok(texts, return_tensors="pt", padding=True).to(model.device)
            gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                                 pad_token_id=tok.pad_token_id)
            outs += tok.batch_decode(gen[:, enc.input_ids.shape[1]:], skip_special_tokens=True)
    return outs


def _fold(s: str) -> str:
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()
    return " " + " ".join(re.sub(r"[^a-z0-9 ]", " ", s).split()) + " "


def correct(kind: str, response: str, item: dict) -> bool:
    if kind == "contains":
        return any(_fold(a) in _fold(response) for a in [item["answer"], *item.get("aliases", [])])
    if kind == "letter":
        m = re.search(r"\b([ABCD])\b", response)
        return bool(m) and m.group(1) == item["answer"]
    if kind == "number":
        m = re.findall(r"Answer:\s*\$?(-?[\d,]*\.?\d+)", response)
        if not m:
            return False
        try:
            return abs(float(m[-1].replace(",", "")) - float(item["answer"])) < 1e-6
        except ValueError:
            return False
    raise ValueError(kind)


def run_benchmark(model, tok, name: str, items: list[dict], batch_size: int, out_dir: Path | None = None) -> tuple[int, int]:
    kind, max_new = BENCHMARKS[name.removeprefix(DEV)]
    prompts = [it["question"] + (GSM8K_SUFFIX if kind == "number" else "") for it in items]
    responses = generate(model, tok, prompts, max_new, batch_size)
    ok = [correct(kind, r, it) for r, it in zip(responses, items)]
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f"{name}.jsonl").write_text("".join(
            json.dumps({"id": it["id"], "response": r, "correct": o}, ensure_ascii=False) + "\n"
            for it, r, o in zip(items, responses, ok)))
    return sum(ok), len(ok)


def load_items(path: Path, limit: int | None = None) -> list[dict]:
    items = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    return items[:limit] if limit else items


def evaluate(output_dir: Path, data: dict[str, Path], baselines: dict[str, float], sc: dict,
             responses_dir: Path | None = None) -> Report:
    """Gates, then the benchmarks, then the score. `data` maps benchmark name -> jsonl path."""
    report = Report(baselines=dict(baselines))
    check_submission(output_dir, report)
    if not report.gated:
        return report
    t0 = time.monotonic()
    if "gsm8k" in data:   # first, while the GPU is free: vLLM in its own venv
        ok, n = run_gsm8k_vllm(model_dir(output_dir / MODEL_FILE), data["gsm8k"], sc.get("n_gsm8k"), responses_dir)
        report.metrics["gsm8k"] = ok / n
        report.counts["gsm8k"] = [ok, n]
    model, tok = load_model(output_dir / MODEL_FILE)
    report.gates["weights_finite"] = all_finite(model)
    if not report.gates["weights_finite"]:
        report.notes.append("non-finite values in the weights")
        return report
    limits = {n: sc.get("n_retention") for n in BENCHMARKS if n.startswith("retention")}
    limits["gsm8k"] = sc.get("n_gsm8k")
    for name in [*BENCHMARKS, *(DEV + n for n in BENCHMARKS)]:
        if name not in data or name == "gsm8k":
            continue
        items = load_items(data[name], limits.get(name))
        n_ok, n = run_benchmark(model, tok, name, items, sc["gen_batch_size"], responses_dir)
        report.metrics[name] = n_ok / n
        report.counts[name] = [n_ok, n]
    report.eval_wall_sec = time.monotonic() - t0
    report.gates["within_eval_wall_clock"] = report.eval_wall_sec <= sc["eval_wall_sec"]
    if not report.gates["within_eval_wall_clock"]:
        report.notes.append(f"evaluation took {report.eval_wall_sec:.0f}s, budget {sc['eval_wall_sec']}s")
    return report
