"""Build the post: post/post.toml and its section files, the runs in data/runs.json,
and templates/ (the page, its stylesheet and script) into one self-contained
HTML file, site/index.html; and every figure and table as files beside it
(extras.py: site/figures/, site/tables/, site/FILES.md).

    uv run --with markdown --with matplotlib python blog/build.py            # build
    uv run --with markdown --with matplotlib python blog/build.py --collect  # re-read the jobs first

Section files are Markdown (with raw HTML, and Markdown inside it with
markdown="1"), plus slots, each on a line of its own:

    {{table:results}}      each agent's best@N run, then the baselines
    {{figure:passk}}       pass@k per agent, baselines dashed
    {{figure:tradeoff}}    true reward against hack rate: dev paths, ◆ held-out
    {{figure:transfer}}    dev against held-out score for every submission
    {{prompt}}             the prompt every run received (post.toml's prompt)
    {{traces}}             every run, linking to its trace page (traces.py: site/runs/)
    {{code}}               each agent's best submission, file by file

Math: $...$ inline and $$...$$ on a line of its own, in TeX; KaTeX renders it.
"""

from __future__ import annotations

import argparse
import fnmatch
import glob
import html
import json
import re
import shutil
import sys
import tomllib
from pathlib import Path

import markdown

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import collect  # noqa: E402
import extras  # noqa: E402
import figures  # noqa: E402
import ideas  # noqa: E402

SLOT = re.compile(r"^\{\{(table|figure|prompt|traces|code)(?::([\w-]+))?\}\}\s*$", re.M)
MATH = re.compile(r"\$\$(.+?)\$\$|\$(?!\s)([^$\n]+?)(?<!\s)\$", re.S)
EXTENSIONS = ["tables", "fenced_code", "md_in_html", "attr_list", "toc", "sane_lists"]


# Formatting, as the reference page writes its numbers.
def num(value: float | None, digits: int = 3) -> str:
    return '<span class="na">-</span>' if value is None else f"{value:.{digits}f}"


def pct(value: float | None) -> str:
    return '<span class="na">-</span>' if value is None else f"{100 * value:.2f}%"


def count(value: int | None) -> str:
    return '<span class="na">-</span>' if not value else f"{value:,}"


def duration(seconds: float | None) -> str:
    if not seconds:
        return '<span class="na">-</span>'
    minutes = round(seconds / 60)
    return f"{minutes // 60}h {minutes % 60:02d}m"


def bar(score: float, color: str) -> str:
    width = max(0.0, min(1.0, score)) * 100
    return (
        f'<span class="bar"><span class="track"><span class="fill {color}" '
        f'style="width:{width:.1f}%"></span></span><span class="bv">{score:.3f}</span></span>'
    )


# The runs, grouped by agent.
def load_groups(config: dict, runs: list[dict]) -> list[dict]:
    groups = []
    for agent in config["agents"]:
        mine = [
            r
            for r in runs
            if any(fnmatch.fnmatch(r["job"], g) for g in agent["jobs"])
            and agent.get("model", r.get("model")) == r.get("model")
        ]
        mine.sort(key=lambda r: r["started"] or "")
        for i, run in enumerate(mine, 1):
            number = f"seed {run['seed']}" if "seed" in run else f"#{i}"
            run["label"] = f"{agent['label']} {number}"
            run["slug"] = f"{agent.get('slug', agent['label'].split()[0].lower())}-{i}"
            recovered_path = HERE / "data" / "trace_dev.json"
            recovered = json.loads(recovered_path.read_text()).get(run["slug"]) if recovered_path.exists() else None
            if recovered and "dev_evaluated_versions" not in run:
                run["dev_version_curve"] = recovered["complete_curve"]
                run["dev_evaluated_versions"] = recovered["evaluated_versions"]
            run["color"] = agent["color"]
            run["effort"] = run.get("effort") or agent.get("effort")
            run["source_label"] = run["label"]
            run["source_slug"] = run["slug"]
        if "selected_jobs" in agent:
            mine = [r for r in mine if r["job"] in agent["selected_jobs"]]
            if len(mine) != len(agent["selected_jobs"]):
                raise ValueError(f"Missing selected runs for {agent['label']}")
            mine.sort(key=lambda r: -r["held_out"]["mean"]["score"])
            for i, run in enumerate(mine, 1):
                run["label"] = agent["label"] if len(mine) == 1 else f"{agent['label']} #{i}"
                run["slug"] = agent["slug"] if len(mine) == 1 else f"{agent['slug']}-{i}"
        groups.append({**agent, "runs": mine})
    return [g for g in groups if g["runs"]]


# Tables.
TABLE_HEAD = (
    '<thead><tr><th class="n">#</th><th>Run</th><th class="n sc">Score (Pass@1) ↑</th>'
    '<th class="n">True reward ↑</th><th class="n">Hack rate ↓</th><th class="n">Clean reward ↑</th>'
    '<th class="n">Agent time</th><th class="n">Dev checks</th><th class="n">Turns</th>'
    '<th class="n">Output tokens</th>{extra}</tr></thead>'
)


def ranked_table(
    groups: list[dict],
    baselines: list[dict],
    keep: tuple[str, ...] | None,
    final: bool = False,
) -> str:
    """Runs and baselines in one table, in the reference post's format: # is the
    rank by held-out score among every run of every agent and the baselines, so a
    table of a few rows still says where each stands. With keep (e.g. best, worst),
    only each agent's runs with those tags, named by the tag ("Opus 5.5 best@8");
    without, every run, by its number. final adds the submitted iteration."""
    held_score = {
        r["job"]: r["held_out"]["mean"]["score"] for g in groups for r in g["runs"]
    }
    scores = sorted(
        list(held_score.values()) + [b["held_out"]["score"] for b in baselines],
        reverse=True,
    )
    rank = {}  # the first free place for each score (ties share one)
    for i, s in enumerate(scores, 1):
        rank.setdefault(s, i)
    entries = []
    for g in extras.tagged(groups, keep):
        for run in g["runs"]:
            held = run["held_out"]["mean"]
            number = run["label"].rsplit(" ", 1)[-1]
            if keep:
                name = f"{g['label']} {' '.join(run['picks'])}".strip()
                after = ""
            else:
                name = run["label"]
                after = "".join(
                    f'<span class="badge">{p}</span>'
                    for p in run["picks"]
                    if p.split("@")[0] in ("best", "worst")
                )
            eff = (
                f' <span class="eff">{html.escape(run["effort"])}</span>'
                if run.get("effort")
                else ""
            )
            note = (
                '<div class="rnote">Penalized: '
                + html.escape(", ".join(map(str, run["penalties"])))
                + ".</div>"
                if run.get("penalties")
                else ""
            )
            cells = (
                f'<td class="run k"><span class="rn"><span class="key {run["color"]}"></span>'
                f'<a href="runs/{run["slug"]}.html" style="white-space:nowrap">{html.escape(name)}</a>'
                f"{eff}{after}</span>{note}</td>"
                f'<td class="n sc">{bar(held["score"], run["color"])}</td>'
                f'<td class="n">{num(held["true_reward"])}</td><td class="n">{pct(held["hack_rate"])}</td>'
                f'<td class="n">{num(held["clean_reward"])}</td>'
                f'<td class="n">{duration(run.get("agent_seconds"))}</td>'
                f'<td class="n">{count(run.get("dev_evaluated_versions", len(run.get("dev_curve") or [])))}</td>'
                f'<td class="n">{count(run.get("turns"))}</td>'
                f'<td class="n">{count(run.get("output_tokens"))}</td>'
            )
            if final:
                cells += (
                    f'<td class="mono small">{html.escape(final_iteration(run))}</td>'
                )
            entries.append((rank[held["score"]], cells))
    na = '<td class="n"><span class="na">-</span></td>'
    for base in baselines:
        held = base["held_out"]
        note = (
            f'<div class="rnote">{html.escape(base["note"])}</div>'
            if base.get("note")
            else ""
        )
        cells = (
            f'<td class="run"><span class="rn"><span class="key b"></span>{html.escape(base["label"])}'
            f'</span>{note}</td><td class="n sc">{bar(held["score"], "b")}</td>'
            f'<td class="n">{num(held["true_reward"])}</td><td class="n">{pct(held["hack_rate"])}</td>'
            f'<td class="n">{num(held["clean_reward"])}</td>'
            + na * 4
            + (na if final else "")
        )
        entries.append((rank[held["score"]], cells))
    entries.sort(key=lambda e: e[0])
    rows = "".join(
        f'<tr class="{"best" if r == 1 else ""}"><td class="n">{r}</td>{cells}</tr>'
        for r, cells in entries
    )
    head = TABLE_HEAD.format(extra="<th>Final submission</th>" if final else "")
    return (
        f'<div class="tw"><table class="lb">{head}<tbody>{rows}</tbody></table></div>'
    )


def markdown_html(text: str) -> str:
    """Markdown as the page writes it: each table in a .tw, which scrolls sideways
    on a narrow screen instead of widening the page. Math ($...$, $$...$$) is set
    aside first, so Markdown leaves the TeX alone, and comes back as \\(...\\) and
    \\[...\\] for KaTeX (code spans and blocks keep their dollar signs)."""
    maths: list[tuple[bool, str]] = []

    def keep(m: re.Match) -> str:
        maths.append((m.group(1) is not None, m.group(1) or m.group(2)))
        return f"@@MATH{len(maths) - 1}@@"

    parts = re.split(r"(```.*?```|`[^`\n]*`)", text, flags=re.S)
    text = "".join(p if i % 2 else MATH.sub(keep, p) for i, p in enumerate(parts))
    body = markdown.markdown(text, extensions=EXTENSIONS)

    def back(m: re.Match) -> str:
        display, tex = maths[int(m.group(1))]
        tex = html.escape(tex.strip(), quote=False)
        return f'<div class="math">\\[{tex}\\]</div>' if display else f"\\({tex}\\)"

    body = re.sub(r"<p>(@@MATH\d+@@)</p>", r"\1", body)
    body = re.sub(r"@@MATH(\d+)@@", back, body)
    return re.sub(
        r"(<table>.*?</table>)", r'<div class="tw">\1</div>', body, flags=re.S
    )


def prompt_block(config: dict, out: Path) -> str:
    """The prompt verbatim, in a scrolling box, with a copy beside the page."""
    source = (HERE / config["prompt"]).resolve()
    text = source.read_text()
    shutil.copy(source, out / "prompt.md")
    words = len(text.split())
    return (
        f'<div class="doc"><div class="doc-h"><span class="mono">instruction.md</span>'
        f'<span class="mut">{words:,} words · <a href="prompt.md" download>Download</a></span></div>'
        f"<pre>{html.escape(text)}</pre></div>"
    )


def final_iteration(run: dict) -> str:
    """The iteration a run submitted last: from the run, or else from its
    final-submission idea card (matched to the submitted files)."""
    if run.get("submitted"):
        return str(run["submitted"])
    cards = json.loads((HERE / "data" / "ideas.json").read_text())["cards"]
    final = [
        c["iteration"]
        for c in cards
        if c["run"] == run.get("source_label", run["label"]) and c.get("final_submission")
    ]
    return final[-1] if final else "-"


def traces_block(groups: list[dict], baselines: list[dict]) -> str:
    """Every run, in the results table's format, each linking to its trace page."""
    return ranked_table(groups, baselines, ("best", "worst"), final=True)


def code_block(config: dict, groups: list[dict], out: Path) -> str:
    """Each agent's best run's submitted files, as graded, with copies beside the page."""
    parts = []
    for g in groups:
        best = max(g["runs"], key=lambda r: r["held_out"]["mean"]["score"])
        found = glob.glob(
            f"{config['jobs_root']}/{best['job']}/*/artifacts/app/submission"
        )
        if not found:
            continue
        folder = out / "code" / best["slug"]
        folder.mkdir(parents=True, exist_ok=True)
        files = []
        for f in sorted(
            Path(found[0]).iterdir(), key=lambda f: (f.suffix != ".py", f.name)
        ):
            if not f.is_file():
                continue
            text = f.read_text(errors="replace")
            (folder / f.name).write_text(text)
            files.append(
                f'<details class="file"><summary><span class="mono">{html.escape(f.name)}</span> '
                f'<span class="mut">{text.count(chr(10)):,} lines · '
                f'<a href="code/{best["slug"]}/{html.escape(f.name)}" download>Download</a></span></summary>'
                f"<pre><code>{html.escape(text)}</code></pre></details>"
            )
        held = best["held_out"]["mean"]
        parts.append(
            f'<h3><span class="key {best["color"]}"></span> {html.escape(best["label"])}'
            f'<span class="fname">{html.escape(final_iteration(best))} · held-out score {held["score"]:.3f}'
            f' · <a href="runs/{best["slug"]}.html">trace</a></span></h3>'
            + "".join(files)
        )
    return "".join(parts)


def render_section(
    config: dict, section: dict, groups: list[dict], counters: dict
) -> str:
    text = (HERE / "post" / section["file"]).read_text()
    blocks: list[str] = []
    baselines = config.get("baselines", [])

    out = (HERE / config["output"]).parent

    def slot(match: re.Match) -> str:
        kind, name = match.groups()
        if kind in ("prompt", "traces", "code"):
            blocks.append(
                prompt_block(config, out)
                if kind == "prompt"
                else traces_block(groups, baselines)
                if kind == "traces"
                else code_block(config, groups, out)
            )
            return f"\n\nSLOT{len(blocks) - 1}X\n\n"
        caption = config.get("captions", {}).get(name, {})
        lead = caption.get("lead", kind.capitalize())
        counters[lead] = counters.get(lead, 0) + 1
        figcaption = (
            f'<figcaption><span class="lead">{lead} {counters[lead]}:</span> '
            f"{caption['text']}</figcaption>"
        ) if caption.get("text") else ""
        if kind == "table":  # the only table: {{table:results}}
            block = figcaption + ranked_table(groups, baselines, ("best", "worst"))
        else:
            draw = {**figures.FIGURES, **ideas.PAGE}[name]
            image = draw(groups, baselines)
            block = (
                f'<figure class="fig {caption.get("class", "")}">{figcaption}{image}'
                '</figure>'
            )
        blocks.append(block)
        return f"\n\nSLOT{len(blocks) - 1}X\n\n"

    text = SLOT.sub(slot, text)
    body = markdown_html(text)
    for i, block in enumerate(blocks):
        body = body.replace(f"<p>SLOT{i}X</p>", block)
    return (
        f'<section id="{section["id"]}" class="tab"><h2 class="sec">'
        f"{html.escape(section['title'])}</h2>{body}</section>"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", default=str(HERE / "post" / "post.toml"))
    parser.add_argument(
        "--collect", action="store_true", help="re-read the job folders first"
    )
    parser.add_argument("--local", action="store_true", help="link Code to the local sandbox")
    parser.add_argument("--rebuild-traces", action="store_true", help="rebuild traces from the original job folders")
    args = parser.parse_args()
    config = tomllib.loads(Path(args.config).read_text())
    data = HERE / config["data"]

    if args.collect:
        root = Path(config["jobs_root"])
        jobs = sorted(
            {p for a in config["agents"] for g in a["jobs"] for p in root.glob(g)}
        )
        regrades = root / config.get("regrades", "regrades")
        runs = [r for r in (collect.collect(j, regrades) for j in jobs) if r]
        for agent in config["agents"]:  # a reference method's seeds
            if "rows" in agent:
                runs += collect.seed_runs(
                    Path(agent["rows"]), agent["jobs"][0].split("-")[0]
                )
        data.write_text(json.dumps(runs, indent=1) + "\n")
        print(f"collected {len(runs)} graded runs of {len(jobs)} jobs into {data}")
    runs = json.loads(data.read_text()) if data.exists() else []
    groups = load_groups(config, runs)
    # A baseline whose seeds are in as runs of their own is shown as those.
    config["baselines"] = [
        b
        for b in config.get("baselines", [])
        if b["label"] not in {g["label"] for g in groups}
    ]

    # Every figure and table as files first; it sets where the page's figures go.
    extras.save(
        groups, config.get("baselines", []), config, (HERE / config["output"]).parent
    )
    counters: dict = {}
    sections = "".join(render_section(config, s, groups, counters)
                       for s in config["sections"] if s["id"] not in {"traces", "code"})
    code_href = "../../safety/tasks/transfer/" if args.local else (
        "https://github.com/vballoli/autoresearch-safety/tree/main/safety/tasks/transfer"
    )
    nav = "".join(
        f'<li><a href="#{s["id"]}">{html.escape(s["nav"])}</a></li>'
        for s in config["sections"]
        if s["id"] not in {"traces", "code"}
    )
    nav += f'<li><a href="traces.html">Traces</a></li><li><a href="{code_href}">Code</a></li>'
    nav += "".join(
        f'<li><a href="{html.escape(link["href"])}">{html.escape(link["nav"])}</a></li>'
        for link in config.get("links", [])
    )
    templates = HERE / "templates"
    template = (templates / "page.html").read_text()
    values = {
        "style": (templates / "style.css").read_text().strip(),
        "script": (templates / "script.js").read_text(),
        "title": html.escape(config["title"]),
        "back_href": html.escape(config.get("back_href", "../index.html")),
        "nav": nav,
        "sections": sections,
    }
    page = template
    for key, text in values.items():
        page = page.replace("{{" + key + "}}", text)
    out = HERE / config["output"]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page)
    # Each run's trace page (site/runs/), which the page links to. Imported here,
    # not at the top: traces.py imports this module.
    trace_nav = re.sub(r'href="#([^"]+)"', r'href="index.html#\1"', nav)
    trace_section = next(s for s in config["sections"] if s["id"] == "traces")
    trace_values = {**values, "title": config["title"] + " — Traces", "nav": trace_nav,
                    "sections": render_section(config, trace_section, groups, counters)}
    trace_page = template
    for key, text in trace_values.items():
        trace_page = trace_page.replace("{{" + key + "}}", text)
    (out.parent / "traces.html").write_text(trace_page)
    if not args.rebuild_traces:
        archived = HERE / "data" / "traces"
        trace_dir = out.parent / "runs"
        trace_dir.mkdir(parents=True, exist_ok=True)
        import editorial
        selected = [r for g in groups for r in g["runs"]]
        allowed = {r["slug"] + ".html" for r in selected}
        for stale in trace_dir.glob("*.html"):
            if stale.name not in allowed:
                stale.unlink()
        for group in groups:
            for run in group["runs"]:
                source = archived / (run["source_slug"] + ".html")
                page = editorial.revise(source.read_text(), run["source_slug"])
                page = page.replace(run["source_label"], run["label"])
                tabs = "" if len(group["runs"]) == 1 else (
                    '<nav aria-label="Runs of this model" class="tabs runtabs">' +
                    "".join(f'<a href="{r["slug"]}.html"' +
                            (' aria-current="page"' if r["job"] == run["job"] else '') +
                            f'>{html.escape(r["label"])} <span class="ts">'
                            f'{r["held_out"]["mean"]["score"]:.3f}</span></a>'
                            for r in group["runs"]) + '</nav>')
                page = re.sub(r'<nav[^>]*class="tabs runtabs"[^>]*>.*?</nav>',
                              lambda _: tabs, page, flags=re.S)
                (trace_dir / (run["slug"] + ".html")).write_text(page)
        (trace_dir / "index.json").write_text(json.dumps([
            {"label": r["label"], "slug": r["slug"], "job": r["job"]}
            for r in selected], indent=2) + "\n")
    if args.rebuild_traces:
        if not Path(config["jobs_root"]).is_dir():
            raise FileNotFoundError("Rebuilding traces requires the original jobs_root directory")
        import traces
        traces.main()
    if (HERE / "landing.py").exists():
        import landing
        landing.update(groups, config["title"], out.parent.parent / "index.html")
    print(
        f"wrote {out} ({len(page) // 1024} kB, {sum(len(g['runs']) for g in groups)} runs)"
    )


if __name__ == "__main__":
    main()
