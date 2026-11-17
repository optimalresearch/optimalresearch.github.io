"""The post's figures, drawn with matplotlib as inline SVG in the page's style: IBM
Plex Sans (fonts/, from Google Fonts, so the PNGs use it too; in the page's SVG
the text stays text), the page's ink and grid colors, and its series palette
(--s1 ... --s8 in templates/style.css).

One style for every figure: sentence-case labels with ↑/↓ for the better
direction, score axes to two decimals, the env splits named "training envs" and
"held-out envs", and the legend in a row below the plot: a square per agent (and
baseline) for color, gray markers for what a mark is.

Each function takes the collected runs (collect.py) grouped by agent and returns
an <svg> string. build.py puts each in a <figure class="fig ...">.
"""

from __future__ import annotations

import io
import logging
import re
from itertools import combinations
from pathlib import Path

from uncertainty import pass_interval, score_interval

import matplotlib

matplotlib.use("Agg")
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.ticker import FixedLocator, FormatStrFormatter  # noqa: E402

for _font in sorted((Path(__file__).resolve().parent / "fonts").glob("*.ttf")):
    font_manager.fontManager.addfont(str(_font))
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

HACK_WEIGHT = 10.0  # score = true reward - HACK_WEIGHT x hack rate

# The page's tokens (templates/style.css :root).
INK, INK2, MUTED, LINE, AXIS, BASE = (
    "#111114",
    "#3f3f46",
    "#686871",
    "#e5e5e8",
    "#cdcdd4",
    "#a1a1aa",
)
SERIES = {
    "s1": "#0072b2",
    "s2": "#e69f00",
    "s3": "#009e73",
    "s4": "#d55e00",
    "s5": "#cc79a7",
    "s6": "#56b4e9",
    "s7": "#a63d8f",
    "s8": "#f0e442",
    "b": BASE,  # a reference method (GRAFT) shown as runs of its own
}

plt.rcParams.update(
    {
        "svg.fonttype": "none",
        "font.family": ["IBM Plex Sans", "DejaVu Sans", "sans-serif"],
        "font.size": 9,
        "axes.edgecolor": AXIS,
        "axes.linewidth": 0.6,
        "axes.labelcolor": INK,
        "axes.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.color": LINE,
        "grid.linewidth": 0.4,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "legend.labelcolor": INK2,
    }
)


def score_axis(axis) -> None:
    """A score (or reward) axis: two decimals."""
    axis.set_major_formatter(FormatStrFormatter("%.2f"))


def legend_below(ax, colors: list, kinds: list | None = None) -> None:
    """The legend centered under the figure, outside the axes: the colors (agents,
    baselines) in the first row, what the marks mean in the second."""
    kinds = kinds or []
    ncol = max(len(colors), 1)
    if kinds and len(kinds) > ncol:
        ncol = len(kinds)
    # Legends fill by column: interleave, so each row is one kind of handle.
    rows = [colors + [None] * (ncol - len(colors))]
    if kinds:
        rows.append(kinds + [None] * (ncol - len(kinds)))
    handles = [h for column in zip(*rows, strict=True) for h in column]
    blank = Line2D([], [], ls="none", label=" ")
    ax.figure.legend(
        handles=[h if h is not None else blank for h in handles],
        loc="outside lower center",
        ncol=ncol,
        handlelength=1.6,
        columnspacing=1.8,
    )


def agent_handles(groups: list[dict], baselines: list[dict]) -> list:
    """A colored square per agent, then a gray one per baseline."""
    return [Patch(color=SERIES[g["color"]], label=g["label"]) for g in groups] + [
        Patch(color=BASE, label=b["label"]) for b in baselines
    ]


def marker_handle(label: str, marker: str, filled: bool = True, **kw) -> Line2D:
    """A gray marker for the legend: what a mark means, whatever its color."""
    return Line2D(
        [],
        [],
        ls="none",
        marker=marker,
        ms=kw.pop("ms", 6.5),
        color=MUTED if filled else "white",
        mec="white" if filled else MUTED,
        mew=1.2 if filled else 1.4,
        label=label,
        **kw,
    )


# Where svg() also saves each figure as NAME.png and NAME.svg, to copy from the
# page (build.py sets it to site/figures/); None saves nothing.
SAVE: Path | None = None


def svg(fig, name: str, label: str) -> str:
    """The figure as an inline <svg>: no XML header or metadata, ids unique to the
    figure (several share a page), and an accessible label. Also saved to SAVE: a
    PNG on white, and an SVG with its text as paths, so it looks the same anywhere."""
    if SAVE is not None:
        fig.savefig(
            SAVE / f"{name}.png", dpi=200, facecolor="white", bbox_inches="tight"
        )
        with plt.rc_context({"svg.fonttype": "path"}):
            fig.savefig(
                SAVE / f"{name}.svg",
                facecolor="white",
                bbox_inches="tight",
                metadata={"Date": None},
            )
    plt.rcParams["svg.hashsalt"] = name
    buffer = io.StringIO()
    fig.savefig(buffer, format="svg", metadata={"Date": None, "Creator": None})
    plt.close(fig)
    text = buffer.getvalue()
    text = text[text.index("<svg") :]
    text = re.sub(r"<metadata>.*?</metadata>\s*", "", text, flags=re.S)
    return text.replace("<svg ", f'<svg role="img" aria-label="{label}" ', 1)


# A picked run (build.py's best@N, worst@N, dev-selected@N) keeps its agent's color
# and gets its tag's line style and held-out marker: ◆ best, ▼ worst, ■ dev-selected.
DASHES = {"best": "-", "worst": (0, (4, 2)), "dev-selected": (0, (1, 1.5))}
MARKERS = {"best": "D", "worst": "v", "dev-selected": "s"}


def run_key(group: dict, run: dict) -> tuple[object, str]:
    """A run's line style and held-out marker: by its first tag if picked."""
    if run.get("picks"):
        tag = run["picks"][0].split("@")[0]
        return DASHES[tag], MARKERS[tag]
    return "-", "D"


def picked_legend(ax, groups: list[dict], extra: list | None = None) -> bool:
    """A legend entry per picked run (its line and held-out marker), below the
    plot; False, drawing nothing, if no run is picked."""
    handles = []
    for group in groups:
        for run in group["runs"]:
            if run.get("picks"):
                dash, marker = run_key(group, run)
                handles.append(
                    Line2D(
                        [],
                        [],
                        color=SERIES[group["color"]],
                        ls=dash,
                        lw=1.5,
                        marker=marker,
                        ms=6,
                        mec="white",
                        label=f"{group['label']} {' = '.join(run['picks'])}",
                    )
                )
    if not handles:
        return False
    handles += extra or []
    ax.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.2),
        ncol=2,
        borderaxespad=0,
    )
    return True


def pass_at(scores: list[float], k: int) -> float:
    """The expected best of k runs chosen at random among them (exact)."""
    subsets = list(combinations(scores, k))
    return sum(max(s) for s in subsets) / len(subsets)


def passk(groups: list[dict], baselines: list[dict]) -> str:
    """Opus Pass@1/2/4 and one bar for each single-run agent."""
    rows = []
    for group in groups:
        scores = [r["held_out"]["mean"]["score"] for r in group["runs"]]
        if len(scores) == 1:
            rows.append((group["label"], scores[0], SERIES[group["color"]], pass_interval(group["runs"], 1)))
        else:
            for k in (1, 2, 4):
                if k <= len(scores):
                    rows.append((f"{group['label']} · Pass@{k}", pass_at(scores, k), SERIES[group["color"]], pass_interval(group["runs"], k)))
    rows += [(b["label"], b["held_out"]["score"], BASE, None) for b in baselines]
    fig, ax = plt.subplots(figsize=(7.2, 3.5), layout="constrained")
    for i, (label, score, color, interval) in enumerate(rows):
        ax.barh(i, score, height=.62, color=color, zorder=3)
        if interval:
            lo, hi = interval
            ax.errorbar(score, i, xerr=[[score-lo], [hi-score]], fmt="none",
                        ecolor=INK, elinewidth=1.1, capsize=3, capthick=1.1, zorder=4)
        ax.text((interval[1] if interval else score) + .012, i, f"{score:.3f}", va="center", fontsize=10, color=INK)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows])
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xticks([0, .25, .5, .75, 1])
    score_axis(ax.xaxis)
    ax.grid(True, axis="x")
    ax.grid(False, axis="y")
    ax.set_xlabel("Held-out score ↑")
    return svg(fig, "passk", "Opus Pass@1, Pass@2 and Pass@4, with Sol, Astra and GRAFT scores")


def dev_curves(
    groups: list[dict], baselines: list[dict] | None = None, name: str = "dev"
) -> str:
    """Figure: each run's dev score at each full launcher run, against hours since
    the agent started; ◆ the submission's held-out score, at the agent's end."""
    fig, ax = plt.subplots(figsize=(7.6, 3.3), layout="constrained")
    for group in groups:
        color = SERIES[group["color"]]
        for i, run in enumerate(group["runs"]):
            dash, marker = run_key(group, run)
            xs = [p["hours"] for p in run["dev_curve"]]
            ys = [max(p["score"], -1.0) for p in run["dev_curve"]]
            if xs:
                ax.plot(
                    xs,
                    ys,
                    color=color,
                    lw=1.5,
                    ls=dash,
                    alpha=0.8,
                    marker="o",
                    ms=4,
                    mec="white",
                    mew=1,
                    label=group["label"] if i == 0 else None,
                )
            end = (run["agent_seconds"] or 0) / 3600
            ax.plot(
                [end],
                [run["held_out"]["mean"]["score"]],
                marker=marker,
                ms=6.5,
                color=color,
                mec="white",
                mew=1.2,
                ls="none",
                zorder=4,
            )
    ax.set_xlabel("Hours since the agent started")
    ax.set_ylabel("Dev score (mean of 3 envs) ↑")
    if not picked_legend(ax, groups):
        ax.legend(loc="lower right")
    return svg(fig, name, "Dev score against time, per run")


def tradeoff(
    groups: list[dict],
    baselines: list[dict],
    name: str = "tradeoff",
    metric: str = "true_reward",
) -> str:
    """Figure: true reward (metric "true_reward") against hack rate, the score's two
    parts (score = true reward - HACK_WEIGHT x hack rate, drawn as faint lines of
    equal score), or clean reward ("clean_reward", no such lines). Each faded line
    connects a run's full launcher runs on dev in order; ◆ is the submission's
    held-out result."""
    fig, ax = plt.subplots(figsize=(5.9, 3.5), layout="constrained")
    for group in groups:
        color = SERIES[group["color"]]
        for run in group["runs"]:
            xs = [100 * p["hack_rate"] for p in run["dev_curve"]]
            ys = [p[metric] for p in run["dev_curve"]]
            if xs:  # the dev path faded, as the final marker is what counts
                ax.plot(xs, ys, color=color, lw=1.8, alpha=0.22, zorder=2)
                ax.plot(
                    xs,
                    ys,
                    ls="none",
                    marker="o",
                    ms=4,
                    color=color,
                    alpha=0.3,
                    mec=color,
                    mew=0.5,
                    zorder=2,
                )
            held = run["held_out"]["mean"]
            ax.plot(
                [100 * held["hack_rate"]],
                [held[metric]],
                marker="D",
                ms=7,
                color=color,
                mec="white",
                mew=1.2,
                ls="none",
                zorder=4,
            )
    for base in baselines:
        held = base["held_out"]
        ax.plot(
            [100 * held["hack_rate"]],
            [held[metric]],
            marker="D",
            ms=6.5,
            color=BASE,
            mec="white",
            mew=1.2,
            ls="none",
            zorder=4,
        )
    ax.grid(True, axis="both")
    if metric == "true_reward":
        equal_score_lines(ax)
    ax.set_xlabel("Hack rate (%) ↓")
    ax.set_ylabel(metric.replace("_", " ").capitalize() + " ↑")
    score_axis(ax.yaxis)
    # Color is the agent (or baseline); the marker says what was measured.
    colors = agent_handles(groups, baselines)
    kinds = [
        Line2D(
            [],
            [],
            color=MUTED,
            lw=1.8,
            alpha=0.35,
            marker="o",
            ms=4,
            label="Dev result (training envs)",
        ),
        marker_handle("Final submission (held-out envs)", "D"),
    ]
    legend_below(ax, colors, kinds)
    return svg(fig, name, f"{metric.replace('_', ' ').capitalize()} against hack rate")


def tradeoff_extremes(groups: list[dict], baselines: list[dict], name: str = "tradeoff") -> str:
    """One panel per best/worst run, selected by final held-out score.

    All panels share limits; paths include only available full-environment dev
    records. Held-out points remain separate because they measure other tasks.
    """
    picked = []
    for group in groups:
        ordered = sorted(group["runs"], key=lambda r: r["held_out"]["mean"]["score"])
        choices = (("Best", ordered[-1]), ("Worst", ordered[0])) if len(ordered) > 1 else (("", ordered[0]),)
        picked.extend((group, tag, run) for tag, run in choices)
    # Taller with each agent, so every row keeps the height it has with two.
    rows = (len(picked) + 1) // 2
    fig, axes = plt.subplots(rows, 2, figsize=(7.6, 1.0 + 2.45 * rows),
                             sharex=True, sharey=True, squeeze=False, layout="constrained")
    points = [p for _, _, run in picked for p in
              [*run.get("dev_version_curve", run.get("dev_curve", [])), run["held_out"]["mean"]]]
    points += [b["held_out"] for b in baselines]
    xmax = max(100 * p["hack_rate"] for p in points)
    ymin = min(p["true_reward"] for p in points)
    ymax = max(p["true_reward"] for p in points)
    for unused in list(axes.flat)[len(picked):]: unused.set_visible(False)
    for ax, (group, tag, run) in zip(axes.flat, picked):
        color = SERIES[group["color"]]
        path = sorted(run.get("dev_version_curve", run.get("dev_curve", [])), key=lambda p: p["hours"])
        xs = [100 * p["hack_rate"] for p in path]
        ys = [p["true_reward"] for p in path]
        ax.plot(xs, ys, color=color, lw=1.3, alpha=.65, zorder=2)
        ax.scatter(xs, ys, s=23, facecolors="white", edgecolors=color, linewidths=1.2, zorder=3)
        for i in range(1, len(path)):
            ax.annotate("", xy=(xs[i], ys[i]), xytext=(xs[i-1], ys[i-1]),
                        arrowprops=dict(arrowstyle="->", color=color, lw=.9, shrinkA=4, shrinkB=4), zorder=2)
        for i in sorted({0, len(path)-1}) if path else []:
            ax.annotate(str(path[i].get("check", i+1)), (xs[i], ys[i]), xytext=(5, 5),
                        textcoords="offset points", fontsize=8, color=INK2)
        held = run["held_out"]["mean"]
        ax.scatter([100*held["hack_rate"]], [held["true_reward"]], marker="D", s=65,
                   color=color, edgecolors="white", linewidths=1.0, zorder=5)
        for base in baselines:
            v = base["held_out"]
            ax.scatter([100*v["hack_rate"]], [v["true_reward"]], marker="s", s=35,
                       color=BASE, edgecolors="white", linewidths=.7, zorder=4)
        title = group["label"] + (f" · {tag.lower()}@{len(group['runs'])}" if tag else "")
        ax.set_title(title,
                     loc="left", fontsize=10, fontweight="semibold", pad=10)
        ax.text(.97, .96, f"Held-out score {held['score']:.3f}", transform=ax.transAxes,
                ha="right", va="top", fontsize=9, color=INK2)
        ax.text(.97, .86, f"{run.get('iterations', 0)} iterations · {run.get('dev_evaluated_versions',len(path))} checks",
                transform=ax.transAxes, ha="right", va="top", fontsize=8.5, color=MUTED)
        if not path:
            ax.text(.97, .76, "Dev curve unavailable", transform=ax.transAxes,
                    ha="right", va="top", fontsize=8.5, color=MUTED)
        ax.set_xscale("symlog", linthresh=1, linscale=1)
        ticks = [v for v in [0, .5, 1, 2, 5, 10, 20] if v <= xmax + 1]
        ax.set_xticks(ticks, [str(v) for v in ticks])
        ax.set_xlim(-.10, xmax + 1)
        ax.set_ylim(max(0, ymin-.07), min(1, ymax+.10))
        ax.grid(True, axis="both", alpha=.6)
        score_axis(ax.yaxis)
    for ax in axes[-1]: ax.set_xlabel("Hack rate (%) ↓", fontsize=9)
    for ax in axes[:,0]: ax.set_ylabel("True reward ↑")
    handles = [Line2D([], [], color=MUTED, marker="o", mfc="white", ms=5, lw=1,
                      label="Three-environment dev checks →"),
               marker_handle("Final held-out result", "D"),
               Line2D([], [], color=BASE, marker="s", ls="none", ms=6, label="GRAFT (held-out)")]
    fig.legend(handles=handles, loc="outside lower center", ncol=3, fontsize=8.5)
    return svg(fig, name, "Best and worst runs: dev paths and final held-out reward versus hack rate")


def equal_score_lines(ax, step: float = 0.5) -> None:
    """Faint lines of equal score over true reward (y) against hack rate in % (x),
    every step, their values along the top edge under "Score";
    the axes keep their limits."""
    (x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()
    for i in range(int(-10 / step), int(10 / step)):
        score = i * step
        xs = [100 * (y - score) / HACK_WEIGHT for y in (y0, y1)]
        if xs[1] < x0 or xs[0] > x1:
            continue
        ax.plot(xs, [y0, y1], color=AXIS, lw=0.7, ls=(0, (1, 2)), zorder=1)
        if x0 <= xs[1] <= x1:
            ax.annotate(
                f"{score:.1f}",
                (xs[1], y1),
                xytext=(0, 2),
                textcoords="offset points",
                fontsize=8,
                color=MUTED,
                ha="center",
                va="bottom",
            )
    ax.annotate(  # names the numbers along the top edge
        "Score ↑",
        (0.5, 1.0),
        xycoords="axes fraction",
        xytext=(0, 14),
        textcoords="offset points",
        fontsize=9,
        color=INK,
        ha="center",
        va="bottom",
    )
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)


FLOOR = -1.0  # scores below it (a penalty: -10, -11) are drawn at it


def submissions(
    groups: list[dict], baselines: list[dict], name: str = "submissions"
) -> str:
    """Figure: the held-out score of every iteration each run submitted, against the
    hours since the agent started; ◆ the one graded at the end, ○ the earlier ones
    (regraded). Baselines dashed."""
    fig, ax = plt.subplots(figsize=(7.6, 3.3), layout="constrained")
    for group in groups:
        color = SERIES[group["color"]]
        for i, run in enumerate(group["runs"]):
            points = [p for p in run.get("submissions", []) if p["hours"] is not None]
            xs = [p["hours"] for p in points]
            ys = [max(p["held_out"]["mean"]["score"], FLOOR) for p in points]
            ax.plot(
                xs,
                ys,
                color=color,
                lw=1.4,
                alpha=0.75,
                zorder=3,
                label=group["label"] if i == 0 else None,
            )
            for p, x, y in zip(points, xs, ys, strict=True):
                ax.plot(
                    [x],
                    [y],
                    ls="none",
                    zorder=4,
                    marker="D" if p["final"] else "o",
                    ms=6 if p["final"] else 5,
                    color=color if p["final"] else "white",
                    mec=color,
                    mew=1.2 if p["final"] else 1.4,
                )
    for base in baselines:
        ax.axhline(
            base["held_out"]["score"], color=MUTED, lw=1, ls=(0, (4, 3)), zorder=2
        )
        ax.annotate(
            base["label"],
            (0.0, base["held_out"]["score"]),
            xycoords=("axes fraction", "data"),
            xytext=(2, 3),
            textcoords="offset points",
            fontsize=9,
            color=MUTED,
        )
    ax.set_xlabel("Hours since the agent started (when submitted)")
    ax.set_ylabel("Held-out score ↑")
    ax.legend(loc="lower right")
    return svg(fig, name, "Held-out score of each submission against time")


def transfer(groups: list[dict], baselines: list[dict]) -> str:
    """Final-submission dev and held-out scores, with evaluation intervals."""
    fig, ax = plt.subplots(figsize=(5.3, 4.9), layout="constrained")
    values = []
    for group in groups:
        color = SERIES[group["color"]]
        for run in group["runs"]:
            if not run.get("dev"):
                continue
            x, y = run["dev"]["mean"]["score"], run["held_out"]["mean"]["score"]
            xlo, xhi = score_interval(run["dev"])
            ylo, yhi = score_interval(run["held_out"])
            values += [xlo, xhi, ylo, yhi]
            ax.errorbar(x, y, xerr=[[x-xlo], [xhi-x]], yerr=[[y-ylo], [yhi-y]],
                        fmt="D", ms=6, color=color, mfc=color, mec="white", mew=.9,
                        elinewidth=1, capsize=2.5, capthick=1, zorder=4)
    for base in baselines:
        x, y = base["dev"]["score"], base["held_out"]["score"]
        values += [x, y]
        ax.plot(x, y, marker="D", ms=6.5, color=BASE, mec="white", mew=1.2,
                ls="none", zorder=4)
    if values:
        low, high = min(values), max(values)
        pad = .08 * (high-low or 1)
        low, high = low-pad, high+pad
        ax.plot([low, high], [low, high], color=AXIS, lw=.9, ls=(0, (4, 3)), zorder=1)
        ax.set_xlim(low, high)
        ax.set_ylim(low, high)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, axis="both")
    score_axis(ax.xaxis)
    score_axis(ax.yaxis)
    ax.set_xlabel("Dev score (training envs) ↑")
    ax.set_ylabel("Held-out score ↑")
    legend_below(ax, agent_handles(groups, baselines))
    return svg(fig, "transfer", "Final submissions: dev and held-out scores with approximate 95% evaluation intervals")


FIGURES = {
    "passk": passk,
    "tradeoff": tradeoff_extremes,
    "transfer": transfer,
}
