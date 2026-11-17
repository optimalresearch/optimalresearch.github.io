"""One page per graded run, in the reference's run-page format
(posttrain.ai/learning-forgetting/runs/opus-2.html): the reference's stylesheet and
script (templates/trace.css, templates/trace.js, verbatim) and its markup, so its
chart cursor, tooltips, step search and "Expand commands" work unchanged.

    uv run --with markdown --with matplotlib python blog/traces.py

A page: the run's header, its agent's other runs, facts; a trace summary (the
submission, the scientific process, the dev checks); the chart (dev checks over the
run's time, a tick per message and command, the held-out grade); search; then the
agent's session, one chapter per dev check. A dev check is one iteration's dev
evaluation: each env's largest (then latest) evaluation of that code version, the
score their mean, at the time the last of them landed; a chapter holds the steps up
to its check, and steps after the last check go in "Final steps".

Reads data/runs.json (numbered as on the post), data/ideas.json (the idea cards)
and each run's job folder under post.toml's jobs_root (read only): the trial's
agent/trajectory.json (ATIF, both harnesses) and agent/experiments/ (iterations,
dev evaluations). Writes site/runs/<slug>.html and site/runs/index.json. Secrets
and email addresses are redacted from everything written.
"""

from __future__ import annotations

import glob
import html
import json
import math
import re
import sys
import tomllib
from datetime import datetime
from pathlib import Path

import markdown

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import build  # noqa: E402

OUT_LIMIT = 4000  # characters of a tool's output shown
IN_LIMIT = 6000  # characters of a tool's input shown
TIP_LIMIT = 100  # characters of a step's summary in the chart's tooltip
HARNESS = {"claude-code": "Claude Code", "codex": "Codex"}
SLUG = {"Opus 5.5": "opus", "GPT-6.1 Sol": "sol", "GPT-6 Astra": "astra"}
SITE_TITLE = "Mitigating Reward Hacking"

# The chart, in the reference's coordinates: a 1000×146 box, the plot from x0 to
# x0 + PLOT_W, scores between y 10 and 92, the band of message and command ticks
# from y 100 to 118 (the script reads the band from y 97).
W, H, X0, PLOT_W = 1000, 146, 40, 942
Y_TOP, Y_BOTTOM, BAND = 10, 92, 97
FLOOR = -1.0  # dev scores below it are drawn at it

# Redaction: keys, tokens and email addresses never reach the page.
SECRET_PATTERNS = [
    re.compile(r"sk-ant-[A-Za-z0-9_\-]{8,}"),
    re.compile(r"\bsk-[A-Za-z0-9_\-]{16,}"),
    re.compile(r"\beyJ[A-Za-z0-9_\-]{8,}(?:\.[A-Za-z0-9_\-]+){0,2}"),
    re.compile(r"(?i)(?<=bearer )[A-Za-z0-9._~+/=\-]{8,}"),
    re.compile(r"\b(?:ghp|gho|ghs|github_pat|xox[abp])_?[A-Za-z0-9_]{16,}"),
]
ENV_SECRET = re.compile(
    r"\b([A-Z][A-Z0-9_]*(?:KEY|TOKEN|SECRET|AUTH|PASSWORD)[A-Z0-9_]*)(\s*[=:]\s*[\"']?)((?![A-Z][A-Z0-9_]*=)[^\s\"']{12,})"
)
JSON_SECRET = re.compile(
    r"(?i)(\"(?:api_?key|access_token|refresh_token|id_token|token|secret|password|authorization|auth)\"\s*:\s*\")([^\"]{8,})"
)
LONG_TOKEN = re.compile(r"[A-Za-z0-9+/_\-]{40,}={0,2}")
EMAIL = re.compile(
    r"[\w.+\-]+@[\w\-]+(?:\.[\w\-]+)*\.(?:com|org|net|edu|io|ai|dev|co|uk|de|gov|me|us)\b"
)


def redact(text: str) -> str:
    for p in SECRET_PATTERNS:
        text = p.sub("[redacted]", text)
    text = ENV_SECRET.sub(lambda m: m.group(1) + m.group(2) + "[redacted]", text)
    text = JSON_SECRET.sub(lambda m: m.group(1) + "[redacted]", text)

    def long(m: re.Match) -> str:  # a long random-looking run of characters
        s = m.group(0)
        upper, lower = sum(c.isupper() for c in s), sum(c.islower() for c in s)
        digits = sum(c.isdigit() for c in s)
        if "/" in s.strip("/") and s.count("/") > 2:  # a path
            return s
        return "[redacted]" if min(upper, lower, digits) >= 6 else s

    text = LONG_TOKEN.sub(long, text)
    return EMAIL.sub("[email]", text)


def esc(text: str) -> str:
    return html.escape(redact(text), quote=True)


def clip(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n… [{len(text) - limit:,} more characters not shown]"


def ts(stamp: str) -> float:
    return datetime.fromisoformat(stamp.replace("Z", "+00:00")).timestamp()


def elapsed(seconds: float) -> str:
    """The reference's times: 6s, 2m, 1h 14m."""
    seconds = max(0, int(seconds))
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m"
    return f"{seconds // 3600}h {seconds % 3600 // 60:02d}m"


def md(text: str) -> str:
    """An agent's message as Markdown, with any raw HTML in it shown as text."""
    text = redact(text)
    parts = re.split(r"(```.*?```|`[^`\n]*`)", text, flags=re.S)
    safe = "".join(
        p
        if i % 2
        else p.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        for i, p in enumerate(parts)
    )
    rendered = markdown.markdown(safe, extensions=["fenced_code", "tables", "sane_lists"])
    return re.sub(r'<a href="(/(?:app|logs)/[^"]+)"[^>]*>(.*?)</a>',
                  r'<span title="Agent container path: \1">\2</span>', rendered, flags=re.S)


# Tool calls, as both harnesses record them.
def call_view(call: dict) -> tuple[str, str, str]:
    """(tool name, one-line summary, full input) of a tool call."""
    name, args = call.get("function_name", "tool"), call.get("arguments") or {}
    if not isinstance(args, dict):
        return name, str(args).split("\n")[0], str(args)
    if name == "exec" and "input" in args:  # Codex: a script of tool calls
        script = args["input"]
        cmds = []
        for m in re.finditer(r'cmd:\s*("(?:[^"\\]|\\.)*")', script):
            try:
                cmds.append(json.loads(m.group(1)))
            except json.JSONDecodeError:
                cmds.append(m.group(1))
        shown = "\n\n".join(cmds) if cmds else script
        first = (cmds[0] if cmds else script).strip().split("\n")[0]
        if len(cmds) > 1:
            first += f"  (+{len(cmds) - 1} more)"
        return "exec", first, shown
    if "command" in args:
        return name, args["command"].strip().split("\n")[0], args["command"]
    if "file_path" in args:
        body = args.get("content") or ""
        if "old_string" in args:
            body = (
                f"--- old\n{args['old_string']}\n+++ new\n{args.get('new_string', '')}"
            )
        return name, args["file_path"], f"{args['file_path']}\n\n{body}".strip()
    text = json.dumps(args, indent=1, ensure_ascii=False)
    return name, text.replace("\n", " ")[:200], text


def output_view(content) -> tuple[str, int | None]:
    """A tool's output as text, and its exit code when it failed (or None)."""
    if isinstance(content, list):
        content = "\n".join(
            c.get("text", "") if isinstance(c, dict) else str(c) for c in content
        )
    content = str(content or "")
    code = None
    m = re.match(r"\s*Exit code (\d+)", content)  # Claude Code's failures
    if m and m.group(1) != "0":
        code = int(m.group(1))
    # Codex wraps each command's result in a JSON chunk: show the output itself.
    chunks = []
    for line in content.split("\n"):
        line = line.strip()
        if line.startswith('{"chunk_id"') or line.startswith('{"wall_time'):
            try:
                chunk = json.loads(line)
                exit_code = chunk.get("exit_code")
                if exit_code not in (None, 0) and code is None:
                    code = int(exit_code)
                head = f"[exit {exit_code}]" if exit_code is not None else ""
                chunks.append(f"{head}\n{chunk.get('output', '')}".strip())
                continue
            except (json.JSONDecodeError, ValueError, TypeError):
                pass
        chunks.append(line)
    return ("\n".join(chunks).strip() if chunks else content), code


# A run's data.
def trial_dir(root: Path, job: str) -> Path:
    return Path(sorted(glob.glob(f"{root}/{job}/*/agent"))[0]).parent


def jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def text_of(message) -> str:
    if isinstance(message, list):
        return "\n".join(
            m.get("text", "") if isinstance(m, dict) else str(m) for m in message
        )
    return str(message or "")


def run_data(root: Path, run: dict) -> dict:
    """The run's agent steps (minutes since the agent started), iterations and dev
    checks."""
    trial = trial_dir(root, run["job"])
    result = json.loads((trial / "result.json").read_text())
    start = ts(result["agent_execution"]["started_at"])
    traj = json.loads((trial / "agent" / "trajectory.json").read_text())
    exp = trial / "agent" / "experiments"
    iterations = jsonl(exp / "iterations.jsonl")
    for it in iterations:
        it["t"] = (ts(it["first_run"]) - start) / 60
    steps = []
    for s in traj["steps"]:
        if s.get("source") != "agent":  # the prompt, the harness's own messages
            continue
        s["t"] = (ts(s["timestamp"]) - start) / 60
        s["text"] = text_of(s.get("message")).strip()
        s["calls"] = s.get("tool_calls") or []
        if s["text"] or s["calls"]:
            steps.append(s)
    steps.sort(key=lambda s: s["t"])
    return {
        "start": start,
        "steps": steps,
        "iterations": sorted(iterations, key=lambda i: i["t"]),
        "checks": dev_checks(jsonl(exp / "dev_evals.jsonl"), iterations, start),
    }


def dev_checks(evals: list[dict], iterations: list[dict], start: float) -> list[dict]:
    """One check per evaluated code version: each env's largest (then latest)
    evaluation, their mean score, true reward and hack rate, at the time the last of
    them landed; in time order."""
    iteration_of: dict[str, str] = {}
    for it in iterations:
        iteration_of.setdefault(it["version"], it["iteration"])
    chosen: dict[str, dict[str, dict]] = {}
    for e in evals:
        if e.get("score") is None:
            continue
        envs = chosen.setdefault(e.get("version") or "?", {})
        old = envs.get(e["env"])
        key = (e.get("episodes") or 0, e.get("time") or 0)
        if old is None or key >= (old.get("episodes") or 0, old.get("time") or 0):
            envs[e["env"]] = e
    checks = []
    for version, envs in chosen.items():

        def mean(field: str, envs: dict = envs) -> float:
            return sum(e[field] for e in envs.values()) / len(envs)

        checks.append(
            {
                "version": version,
                "iteration": iteration_of.get(version, version[:12]),
                "t": (max(e["time"] for e in envs.values()) - start) / 60,
                "score": mean("score"),
                "true_reward": mean("true_reward"),
                "hack_rate": mean("hack_rate"),
                "envs": {env: envs[env]["score"] for env in sorted(envs)},
            }
        )
    checks.sort(key=lambda c: c["t"])
    for k, c in enumerate(checks, 1):
        c["k"], c["id"] = k, f"check-{k}"
    return checks


def env_parts(check: dict) -> str:
    return ", ".join(f"{env} {score:.3f}" for env, score in check["envs"].items())


def step_summary(s: dict) -> str:
    """A step in one line, for the chart's tooltip: its first command, or its message."""
    if s["calls"]:
        tool, line, _ = call_view(s["calls"][0])
        text = f"{tool}: {line}"
    else:
        text = s["text"].split("\n")[0]
    return redact(text)[:TIP_LIMIT]


# The chart.
def spine(run: dict, data: dict, total: float) -> str:
    """The reference's chart: dev checks (circles, linked to their chapters) over the
    run's minutes, a tick per message and per command, a cursor the script moves
    with the page, and the held-out grade (diamond, linked to the summary)."""
    color, checks = run["color"], data["checks"]
    held = run["held_out"]["mean"]["score"]
    xs = PLOT_W / total
    values = [max(c["score"], FLOOR) for c in checks] + [held]
    low, high = min(values), max(values)
    step = next(
        s
        for s in (0.05, 0.1, 0.2, 0.25, 0.5, 1.0, 2.0)
        if math.ceil(high / s - 1e-9) - math.floor(low / s + 1e-9) <= 4
    )
    lo = math.floor(low / step + 1e-9) * step
    hi = max(math.ceil(high / step - 1e-9) * step, lo + step)

    def x(t: float) -> float:
        return X0 + min(max(t, 0), total) * xs

    def y(v: float) -> float:
        return Y_BOTTOM - (max(v, lo) - lo) / (hi - lo) * (Y_BOTTOM - Y_TOP)

    out = [
        '<div class="spine"><p class="spine-label">Circles show dev scores (the mean '
        "over the training envs) and the diamond shows the final held-out score. Time "
        "is measured in hours from the start of the run.</p>"
        f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Dev evaluation score over '
        f'time; diamond marks the final held-out score" data-w="{W}" data-h="{H}" '
        f'data-x0="{X0}" data-xs="{xs:.5f}" data-band="{BAND}">'
        f'<rect class="panel" x="{X0}" y="6" width="{PLOT_W}" height="112"/>'
    ]
    n = round((hi - lo) / step)
    for i in range(n + 1):
        v = lo + i * step
        out.append(
            f'<line class="grid" x1="{X0}" x2="{X0 + PLOT_W}" y1="{y(v):.1f}" y2="{y(v):.1f}"/>'
            f'<text class="tick" x="{X0 - 6}" y="{y(v) + 4:.1f}" text-anchor="end">{v:.2f}</text>'
        )
    for hour in range(int(total // 60) + 1):
        out.append(
            f'<text class="tick" x="{x(60 * hour):.1f}" y="132" text-anchor="middle">{hour}h</text>'
        )
    for c in checks:
        out.append(
            f'<line class="ckl {color}" style="stroke:var(--{color})" x1="{x(c["t"]):.1f}" '
            f'x2="{x(c["t"]):.1f}" y1="{y(c["score"]):.1f}" y2="118"/>'
        )
    if checks:
        path = " L".join(f"{x(c['t']):.1f},{y(c['score']):.1f}" for c in checks)
        out.append(f'<path class="line {color}" d="M{path}"/>')
    for s in data["steps"]:  # commands first, then messages on top, as the reference
        if s["calls"]:
            out.append(
                f'<rect class="tk-c" x="{x(s["t"]) - 0.6:.1f}" y="100" width="1.3" height="18"/>'
            )
    for s in data["steps"]:
        if s["text"]:
            out.append(
                f'<rect class="tk-m" x="{x(s["t"]) - 0.6:.1f}" y="100" width="1.3" height="18"/>'
            )
    out.append(
        f'<line id="cur" class="cursor" x1="{X0}" x2="{X0}" y1="{Y_TOP}" y2="118"/>'
    )
    for c in checks:
        below = " (below the chart)" if c["score"] < lo else ""
        tip = (
            f"Dev check {c['k']} at {elapsed(60 * c['t'])}: {c['score']:.3f}{below} "
            f"({env_parts(c)})"
        )
        out.append(
            f'<a href="#{c["id"]}"><circle class="dot {color}" cx="{x(c["t"]):.1f}" '
            f'cy="{y(c["score"]):.1f}" r="5" data-tip="{esc(tip)}"/></a>'
        )
    hx, hy = X0 + PLOT_W, y(held)
    out.append(
        f'<a href="#final"><path class="dia {color}" d="M{hx:.1f},{hy - 7:.1f} '
        f'L{hx + 7:.1f},{hy:.1f} L{hx:.1f},{hy + 7:.1f} L{hx - 7:.1f},{hy:.1f}Z" '
        f'data-tip="Held-out grade {held:.3f}"/></a></svg>'
        '<div class="tip" id="tip" hidden></div>'
    )
    ticks = [
        [
            round(s["t"], 2),
            str(s["step_id"]),
            f"#{s['step_id']} at {elapsed(60 * s['t'])}: {step_summary(s)}",
        ]
        for s in data["steps"]
    ]
    out.append(
        '<script type="application/json" id="ticks">'
        + json.dumps(ticks).replace("<", "\\u003c").replace(">", "\\u003e")
        + "</script></div>"
    )
    return "".join(out)


def tools(color: str) -> str:
    return (
        '<div class="tools"><input id="q" type="search" placeholder="Search steps" '
        'aria-label="Search steps"><button id="expand" type="button">Expand commands'
        '</button><span id="cnt"></span><span class="keys"><span><i class="tk-m" '
        'style="background:var(--fg)"></i>message</span><span><i style="background:'
        f'var(--axis)"></i>commands</span><span><i style="background:var(--{color})">'
        "</i>dev check</span></span></div>"
    )


# The session, one chapter per dev check.
class Steps:
    """Renders steps, giving each step's first element the id s<step> (as the
    reference does), and groups consecutive tool calls under "Ran N commands"."""

    def __init__(self) -> None:
        self.used: set[str] = set()
        self.pending: list[tuple[dict, dict]] = []

    def claim(self, step: dict) -> str:
        anchor = f"s{step['step_id']}"
        if anchor in self.used:
            return ""
        self.used.add(anchor)
        return f' id="{anchor}"'

    def message(self, s: dict) -> str:
        out = self.flush()
        ident = self.claim(s)
        anchor = f"s{s['step_id']}"
        return out + (
            f'<div class="st"{ident} data-t="{s["t"]:.2f}"><a class="g" href="#{anchor}">'
            f'#{s["step_id"]}<br>{elapsed(60 * s["t"])}</a><div class="b">{md(s["text"])}'
            "</div></div>"
        )

    def flush(self) -> str:
        if not self.pending:
            return ""
        pending, self.pending = self.pending, []
        first = pending[0][0]
        anchor = f"s{first['step_id']}"
        gutter = (
            '<span class="g"></span>'
            if anchor in self.used
            else f'<a class="g" href="#{anchor}">#{first["step_id"]}<br>{elapsed(60 * first["t"])}</a>'
        )
        names: dict[str, int] = {}
        calls = []
        for s, c in pending:
            tool, line, full = call_view(c)
            names[tool] = names.get(tool, 0) + 1
            outs = [
                output_view(r.get("content"))
                for r in (s.get("observation") or {}).get("results", [])
                if r.get("source_call_id") == c.get("tool_call_id")
            ]
            text = "\n".join(o[0] for o in outs)
            code = next((o[1] for o in outs if o[1] is not None), None)
            bad = f'<span class="bad">exit {code}</span>' if code is not None else ""
            calls.append(
                f'<details class="call"{self.claim(s)} data-t="{s["t"]:.2f}"><summary>'
                f'<span class="cn">#{s["step_id"]} {esc(tool)}</span><code class="l">'
                f"{esc(line)}</code>{bad}</summary>"
                f'<pre class="in"><code>{esc(clip(full, IN_LIMIT))}</code></pre>'
                + (
                    f'<pre class="out"><code>{esc(clip(text, OUT_LIMIT))}</code></pre>'
                    if text
                    else ""
                )
                + "</details>"
            )
        seconds = 60 * (pending[-1][0]["t"] - first["t"])
        meta = ", ".join(f"{k} ×{v}" for k, v in names.items())
        if seconds >= 1:
            meta += f" ({elapsed(seconds)})"
        noun = "command" if len(calls) == 1 else "commands"
        return (
            f'<div class="st" data-t="{first["t"]:.2f}">{gutter}<div class="b">'
            f'<details class="grp"><summary>Ran {len(calls)} {noun}<span class="meta">'
            f'{esc(meta)}</span></summary><div class="calls">{"".join(calls)}</div>'
            "</details></div></div>"
        )


def chapters(run: dict, data: dict, total: float) -> str:
    checks, steps = data["checks"], data["steps"]
    bounds = [c["t"] for c in checks]
    groups: list[list[dict]] = [[] for _ in range(len(checks) + 1)]
    for s in steps:
        k = next((i for i, b in enumerate(bounds) if s["t"] <= b), len(checks))
        groups[k].append(s)
    held = run["held_out"]["mean"]["score"]
    render = Steps()
    out = []
    for k, group in enumerate(groups):
        last = k == len(checks)
        if last and not group:
            continue
        t0 = bounds[k - 1] if k else 0.0
        t1 = total if last else bounds[k]
        count = f"{len(group)} step{'s' if len(group) != 1 else ''}"
        if last:
            cid, name = "final-steps", "Final steps" if checks else "Trace"
            right = (
                f'<b>{held:.3f}</b> <span class="mut small">final held-out score</span>'
            )
        else:
            c = checks[k]
            cid, name = c["id"], f"Dev check {c['k']}"
            delta = ""
            if k:
                d = c["score"] - checks[k - 1]["score"]
                delta = f' <span class="{"up" if d >= 0 else "down"}">{d:+.3f}</span>'
            right = (
                f'<b>{c["score"]:.3f}</b>{delta} <span class="mut small">'
                f"{esc(c['iteration'])}: {esc(env_parts(c))}</span>"
            )
        out.append(
            f'<section class="ch" id="{cid}"><header class="chh"><div><span class="k">'
            f'{name}</span> <span class="mut small">{elapsed(60 * t0)} to '
            f"{elapsed(60 * t1)}, {count}</span></div><div>{right}</div></header>"
        )
        for s in group:
            if s["text"]:
                out.append(render.message(s))
            for c in s["calls"]:
                render.pending.append((s, c))
        out.append(render.flush() + "</section>")
    return "".join(out)


# The trace summary.
def summary(run: dict, data: dict, cards: dict, clusters: dict) -> str:
    held = run["held_out"]["mean"]["score"]
    checks = data["checks"]
    final_name = build.final_iteration(run)
    final = cards.get(final_name)
    check_of = {c["iteration"]: c for c in checks}

    def family(card: dict | None) -> str:
        return clusters.get((card or {}).get("cluster"), {}).get("label", "")

    def link(iteration: str) -> str:
        """Where an iteration is in the trace: its dev check, or else the first step
        after it was first trained."""
        if iteration in check_of:
            c = check_of[iteration]
            return f'<a href="#{c["id"]}">Dev check {c["k"]}</a>'
        it = next((i for i in data["iterations"] if i["iteration"] == iteration), None)
        if it is None:
            return ""
        s = next((s for s in data["steps"] if s["t"] >= it["t"]), None)
        return f'<a href="#s{s["step_id"]}">Step #{s["step_id"]}</a>' if s else ""

    out = [
        '<section class="ch" id="final"><header class="chh"><div><span class="k">'
        f'Trace summary</span></div><div><b>{held:.3f}</b> <span class="mut small">'
        'final held-out score</span></div></header><h3>Submission</h3><div class="submission-summary">'
    ]
    if final:
        out.append(
            f"<p>The agent submitted <code>{esc(final_name)}</code>, in the method family "
            f"<strong>{esc(family(final))}</strong>. {esc(final['method'])}</p>"
        )
    else:
        out.append(f"<p>The agent submitted <code>{esc(final_name)}</code>.</p>")
    if final_name in check_of:
        c = check_of[final_name]
        out.append(
            f"<p>The submitted version scored <b>{c['score']:.3f}</b> at dev check {c['k']} "
            f"and <b>{held:.3f}</b> on the final held-out evaluation.</p>"
        )
    else:
        out.append(
            f"<p>The submitted version scored <b>{held:.3f}</b> on the final held-out "
            "evaluation.</p>"
        )

    # Scientific process: the run's path through the method families.
    ordered = sorted(cards.values(), key=lambda c: float(c.get("hours") or 0))
    families: list[str] = []
    for c in ordered:
        if family(c) and family(c) not in families:
            families.append(family(c))
    submitted = [c for c in ordered if c.get("submitted")]
    out.append("<h3>Scientific process</h3>")
    if ordered:
        path = ", then ".join(families)
        times = max(len(submitted), 1)
        out.append(
            f"<p>Over {elapsed(float(run['agent_seconds'] or 0))}, the agent worked through "
            f"{len(ordered)} iterations in {len(families)} method "
            f"famil{'y' if len(families) == 1 else 'ies'}: {esc(path)}. It ran "
            f"{len(checks)} dev check{'s' if len(checks) != 1 else ''} and submitted "
            f"{times} time{'s' if times != 1 else ''}; the last submission, "
            f"<code>{esc(final_name)}</code>, is the one graded.</p>"
        )
        seen: set[str] = set()
        key = []
        for c in ordered:  # the first of each family, every submission, the final
            fam = family(c)
            if (
                fam not in seen
                or c.get("submitted")
                or c.get("final_submission")
                or c["iteration"] == final_name
            ):
                key.append(c)
            seen.add(fam)
        items = []
        for c in key[:8]:
            what = (c.get("change") or "").strip().rstrip(".")
            what = what[:1].upper() + what[1:]
            outcome = (c.get("outcome") or "").strip().rstrip(".")
            mark = " (submitted)" if c["iteration"] == final_name else ""
            items.append(
                f"<li><strong>{esc(c['iteration'])}{mark}.</strong> {esc(what)}"
                f"{': ' + esc(outcome) if outcome else ''}. {link(c['iteration'])}</li>"
            )
        out.append(f"<ul>{''.join(items)}</ul>")

    # How the dev scores improved.
    out.append(
        "<h3>How the dev scores improved</h3><p>Each dev check evaluates one version of "
        "the agent's code on the dev split of its three training envs, with the "
        "grader's ground truth; the score is the mean over the envs evaluated (a check "
        "can cover fewer than three). The checks are numbered in the order they "
        "finished. The final held-out evaluation trains the submission afresh in "
        "three other envs.</p>"
    )
    rows = []
    for c in checks:
        card = cards.get(c["iteration"], {})
        change = (card.get("change") or "").strip()
        candidate = esc(c["iteration"]) + (f", {esc(change)}" if change else "")
        if c["iteration"] == final_name:
            candidate += " (submitted)"
        rows.append(
            f'<tr><td><a href="#{c["id"]}">Dev check {c["k"]}</a></td><td>{candidate}</td>'
            f'<td class="n">{c["score"]:.3f}</td><td class="n">{c["true_reward"]:.3f}</td>'
            f'<td class="n">{100 * c["hack_rate"]:.2f}%</td></tr>'
        )
    out.append(
        '<div class="tw"><table><thead><tr><th>Evaluation</th><th>Candidate</th>'
        '<th class="n">Dev score</th><th class="n">True reward</th>'
        f'<th class="n">Hack rate</th></tr></thead><tbody>{"".join(rows)}</tbody></table></div>'
        "</div></section>"
    )
    return "".join(out)


def page(
    run: dict, group: dict, slug: str, data: dict, cards: dict, clusters: dict
) -> str:
    templates = HERE / "templates"
    held = run["held_out"]["mean"]
    steps = data["steps"]
    total = max(
        float(run["agent_seconds"] or 0) / 60,
        max((s["t"] for s in steps), default=0),
        max((c["t"] for c in data["checks"]), default=0),
        1.0,
    )
    tabs = "".join(
        f'<a href="{slug}-{i}.html"{" aria-current=page" if r is run else ""}><i class="key '
        f'{r["color"]}"></i>Run {i} <span class="ts">{r["held_out"]["mean"]["score"]:.3f}</span></a>'
        for i, r in enumerate(group["runs"], 1)
    )
    n_calls = sum(len(s["calls"]) for s in steps)
    facts = "".join(
        f"<span>{k} <b>{v}</b></span>"
        for k, v in (
            ("Final held-out score", f"{held['score']:.3f}"),
            ("True reward", f"{held['true_reward']:.3f}"),
            ("Hack rate", f"{100 * held['hack_rate']:.2f}%"),
            ("Agent time", elapsed(float(run["agent_seconds"] or 0))),
            ("Steps", f"{len(steps):,}"),
            ("Tool calls", f"{n_calls:,}"),
        )
    )
    model = run["model"].split("/")[-1]
    harness = HARNESS.get(run["agent"], run["agent"])
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" '
        'content="width=device-width,initial-scale=1"><meta name="color-scheme" content="light">'
        f"<title>{esc(run['label'])} | {SITE_TITLE}</title>"
        '<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="stylesheet" '
        'href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&'
        'family=IBM+Plex+Mono:wght@400;500&display=swap">'
        f"<style>{(templates / 'trace.css').read_text()}</style></head>\n<body><main>"
        '<nav class="crumb" aria-label="Report navigation"><a href="../traces.html">'
        '← Back</a></nav><div class="traj">'
        f'<h1>{esc(run["label"])} <span class="eff">{esc(run.get("effort") or "")}</span></h1>'
        f'<p class="sub">{esc(harness)} running {esc(model)}</p>'
        f'<nav class="tabs runtabs" aria-label="Runs of this model">{tabs}</nav>'
        f'<div class="facts">{facts}</div>'
        + summary(run, data, cards, clusters)
        + spine(run, data, total)
        + tools(run["color"])
        + chapters(run, data, total)
        + f"</div></main><script>{(templates / 'trace.js').read_text()}</script></body></html>"
    )


def main() -> None:
    config = tomllib.loads((HERE / "post" / "post.toml").read_text())
    root = Path(config["jobs_root"])
    runs = json.loads((HERE / config["data"]).read_text())
    groups = build.load_groups(config, runs)
    idea_data = json.loads((HERE / "data" / "ideas.json").read_text())
    clusters = {c["id"]: c for c in idea_data["clusters"]}
    out = (HERE / config["output"]).parent / "runs"
    out.mkdir(parents=True, exist_ok=True)
    index = []
    for group in groups:
        slug = group.get("slug") or SLUG.get(
            group["label"], group["label"].lower().split()[0]
        )
        for i, run in enumerate(group["runs"], 1):
            cards = {
                c["iteration"]: c
                for c in idea_data["cards"]
                if c["run"] == run["label"]
            }
            data = run_data(root, run)
            name = f"{slug}-{i}"
            import editorial
            text = editorial.revise(page(run, group, slug, data, cards, clusters), name)
            (out / f"{name}.html").write_text(text)
            index.append(
                {
                    "run": run["label"],
                    "agent": group["label"],
                    "slug": name,
                    "href": f"runs/{name}.html",
                    "held_out_score": round(run["held_out"]["mean"]["score"], 4),
                }
            )
            print(f"{name}.html  {len(text) / 1e6:.2f} MB  {run['label']}")
    (out / "index.json").write_text(json.dumps(index, indent=1) + "\n")


if __name__ == "__main__":
    main()
