"""Every figure and table made for the post, saved as files next to the page for
copying, including those the page no longer shows: site/figures/NAME.{png,svg},
site/tables/NAME.{md,csv}, and site/FILES.md listing each with its caption.

Also the run tags the page's Table 1 uses: per agent, best@N and worst@N (the
highest and lowest held-out score of its N runs) and dev-selected@N (the highest
dev score, a pick made without the held-out envs).
"""

from __future__ import annotations

import csv
from pathlib import Path

import collect
import figures
import ideas

TAGS = ("best", "worst", "dev-selected")


def picks(runs: list[dict]) -> dict[str, list[str]]:
    """Each picked run's tags, by job; a run picked twice has both."""
    n = len(runs)
    if n < 2:
        return {}
    held = sorted(runs, key=lambda r: -r["held_out"]["mean"]["score"])
    dev = max(runs, key=lambda r: r["dev"]["mean"]["score"] if r["dev"] else -1e9)
    tags: dict[str, list[str]] = {}
    for tag, run in ((f"best@{n}", held[0]), (f"worst@{n}", held[-1])):
        tags.setdefault(run["job"], []).append(tag)
    tags.setdefault(dev["job"], []).append(f"dev-selected@{n}")
    return tags


def tagged(groups: list[dict], keep: tuple[str, ...] | None = None) -> list[dict]:
    """The groups with each run's tags in run["picks"]. With keep, only the tags
    of those kinds, and only the runs that have one."""
    out = []
    for g in groups:
        tags = picks(g["runs"])
        runs = []
        for r in g["runs"]:
            mine = [
                t
                for t in tags.get(r["job"], [])
                if keep is None or t.split("@")[0] in keep
            ]
            if mine or keep is None or len(g["runs"]) == 1:
                runs.append({**r, "picks": mine})
        out.append({**g, "runs": runs})
    return out


# The figures the page doesn't show: (file name, function, its options, the tags
# of the only runs to draw or None for every run, caption).
FIGURES = [
    (
        "dev_curves",
        figures.dev_curves,
        {},
        None,
        "Each run's dev score at each full launcher run, in the agent's own envs; ◆ the submission's held-out score.",
    ),
    (
        "dev_curves_picked",
        figures.dev_curves,
        {},
        TAGS,
        "As dev_curves, for each agent's best@N, worst@N and dev-selected@N runs: ◆ best, ▼ worst, ■ dev-selected.",
    ),
    (
        "tradeoff_clean",
        figures.tradeoff,
        {"metric": "clean_reward"},
        None,
        "Clean reward (true reward, 0 on hacking completions) against hack rate: each run's dev checks in order, ◆ its held-out result.",
    ),
    (
        "submissions",
        figures.submissions,
        {},
        None,
        "Held-out score of every iteration each run submitted, when submitted: ◆ graded at the end of the run, ○ earlier ones (regraded); scores below −1 drawn at −1.",
    ),
]
FIGURES += [
    (
        "idea_map",
        ideas.idea_map,
        {},
        None,
        "Idea map: every iteration each run tried, by method family (subagents judged the families from each iteration's code and notes; neighbors are the most similar); ◆ final submissions with their run number.",
    ),
    (
        "idea_timeline",
        ideas.idea_timeline,
        {},
        None,
        "Method family of each iteration against hours into the run, one panel per agent and one line per run; ◆ final submissions.",
    ),
    (
        "idea_space",
        ideas.idea_space,
        {},
        None,
        "Idea space: every iteration of both agents in one 2-D layout (near = similar method family and shared building blocks), a blob per family (faded: the agent never tried it), a line from each run's first iteration (○) to its final submission (●).",
    ),
]
METRICS = ("score", "true_reward", "hack_rate", "clean_reward")


def run_row(run: dict) -> dict:
    held, dev = run["held_out"]["mean"], run["dev"].get("mean", {})
    return {
        "run": run["label"],
        "tags": " ".join(run.get("picks", [])),
        **{m: round(held[m], 4) for m in METRICS},
        "dev_score": round(dev["score"], 4) if dev else "",
        "agent_hours": round(run["agent_seconds"] / 3600, 2)
        if run.get("agent_seconds")
        else "",
        "dev_checks": len(run.get("dev_curve") or []),
        "iterations": run.get("iterations"),
        "code_checks": run.get("code_checks"),
        "turns": run.get("turns"),
        "output_tokens": run.get("output_tokens"),
        "submitted": run.get("submitted"),
        "job": run["job"],
    }


def base_row(base: dict) -> dict:
    return {
        "run": base["label"],
        **{m: base["held_out"][m] for m in METRICS},
        "dev_score": base.get("dev", {}).get("score", ""),
    }


def env_rows(groups: list[dict], baselines: list[dict], scope: str) -> list[dict]:
    """Per agent, the mean over its runs in each env of scope, then the baselines."""
    envs = (*(collect.HELD_OUT if scope == "held_out" else collect.DEV), "mean")
    rows = []
    for g in groups:
        row: dict = {"agent": g["label"], "runs": len(g["runs"])}
        for e in envs:
            values = [r[scope][e] for r in g["runs"] if r.get(scope)]
            for m in METRICS:
                row[f"{e} {m}"] = round(sum(v[m] for v in values) / len(values), 4)
        rows.append(row)
    for base in baselines:
        row = {"agent": base["label"], "runs": ""}
        for e in envs:
            m_of = (
                base.get(scope, {}) if e == "mean" else base.get("envs", {}).get(e, {})
            )
            for m in METRICS:
                row[f"{e} {m}"] = m_of.get(m, "")
        rows.append(row)
    return rows


def arrowed(column: str) -> str:
    """A column name with ↑ if higher is better or ↓ if lower is, for the Markdown
    tables (the CSVs keep plain names)."""
    if column.endswith("hack_rate"):
        return column + " ↓"
    if column.endswith(("score", "true_reward", "clean_reward")):
        return column + " ↑"
    return column


def write_table(rows: list[dict], path: Path) -> None:
    """rows as path.csv and path.md (the columns of every row, in first-seen order)."""
    columns = list(dict.fromkeys(k for row in rows for k in row))
    with path.with_suffix(".csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, columns, restval="")
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "| " + " | ".join(map(arrowed, columns)) + " |",
        "|" + " - |" * len(columns),
    ]
    lines += [
        "| " + " | ".join(str(row.get(c, "")) for c in columns) + " |" for row in rows
    ]
    path.with_suffix(".md").write_text("\n".join(lines) + "\n")


def runs_of(groups: list[dict]) -> list[dict]:
    """The groups' runs, best held-out score first."""
    runs = [r for g in groups for r in g["runs"]]
    return sorted(runs, key=lambda r: -r["held_out"]["mean"]["score"])


def save(groups: list[dict], baselines: list[dict], config: dict, site: Path) -> None:
    """Write the extra figures, every table and site/FILES.md."""
    figures.SAVE = site / "figures"
    figures.SAVE.mkdir(parents=True, exist_ok=True)
    listing = ["# Figures and tables", "", "## Figures", ""]
    for name in {**figures.FIGURES, **ideas.PAGE}:  # the page's own, saved as the page builds
        text = config.get("captions", {}).get(name, {}).get("text", "")
        listing.append(f"- `figures/{name}.png`, `.svg` (on the page): {text}")
    for name, draw, options, keep, caption in FIGURES:
        draw(
            tagged(groups, keep) if keep else groups,
            baselines,
            name=name,
            **options,
        )
        listing.append(f"- `figures/{name}.png`, `.svg`: {caption}")

    tables = site / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    bases = [base_row(b) for b in baselines]
    specs = [
        (
            "results",
            [run_row(r) for r in runs_of(tagged(groups, ("best", "worst")))] + bases,
            "Table 1: Opus best@4 and worst@4, Sol, Astra, and GRAFT (held-out means).",
        ),
        (
            "picked",
            [run_row(r) for r in runs_of(tagged(groups, TAGS))] + bases,
            "Each agent's best@N, worst@N and dev-selected@N runs, then the baselines.",
        ),
        (
            "runs",
            [run_row(r) for r in runs_of(tagged(groups))] + bases,
            "Every run, best held-out score first, with its tags; then the baselines.",
        ),
        (
            "held_out",
            env_rows(groups, baselines, "held_out"),
            "Per held-out env, each agent's mean over its runs, and the baselines.",
        ),
        (
            "dev",
            env_rows(groups, baselines, "dev"),
            "Per dev env (the agent's own), each agent's mean over its runs, and the baselines.",
        ),
    ]
    listing += ["", "## Tables", "", "Hack rates are fractions (0.01 = 1%).", ""]
    for name, rows, caption in specs:
        write_table(rows, tables / name)
        listing.append(f"- `tables/{name}.md`, `.csv`: {caption}")
    (site / "FILES.md").write_text("\n".join(listing) + "\n")
