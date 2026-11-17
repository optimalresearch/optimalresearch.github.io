"""The idea map: every iteration each run tried, placed by its method family.

blog/data/ideas.json holds the families (clusters, in an order where neighbors are
the most similar methods) and one card per iteration: its run, hours since the
agent started, family, an offset toward the neighboring family it resembles, and
whether it was the run's final submission. Subagents judged the families from each
iteration's code diff and the agent's own notes (blog/README.md).

Two figures, saved with the others (extras.FIGURES):
- idea_map: families along x, one row per agent; ● an iteration, ◆ a final submission.
- idea_timeline: per agent, family against hours into the run; one line per run.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
from figures import INK, INK2, LINE, MUTED, SERIES, marker_handle, svg
from matplotlib.lines import Line2D

DATA = Path(__file__).parent / "data" / "ideas.json"


def load(groups: list[dict]) -> tuple[list[dict], list[dict], list[dict]] | None:
    """The clusters (by position), the agents with ideas (each with its runs' cards,
    color and held-out scores), or None without ideas.json."""
    if not DATA.exists():
        return None
    data = json.loads(DATA.read_text())
    clusters = sorted(data["clusters"], key=lambda c: c["position"])
    where = {c["id"]: c["position"] for c in clusters}
    agents = []
    for g in groups:
        runs = []
        for run in g["runs"]:
            cards = [c for c in data["cards"] if c["run"] == run.get("source_label", run["label"])]
            if not cards:
                continue
            for c in cards:
                c["y"] = where[c["cluster"]] + c.get("offset", 0)
            cards.sort(key=lambda c: c["hours"])
            runs.append(
                {
                    "label": run["label"],
                    "number": run["label"].rsplit("#", 1)[-1] if "#" in run["label"] else "",
                    "score": run["held_out"]["mean"]["score"],
                    "cards": cards,
                }
            )
        if runs:
            agents.append(
                {"label": g["label"], "color": SERIES[g["color"]], "runs": runs}
            )
    return clusters, agents, [c for a in agents for r in a["runs"] for c in r["cards"]]


def bands(ax, clusters: list[dict], axis: str) -> None:
    """Alternate light bands behind the families, along axis ("x" or "y")."""
    for c in clusters[::2]:
        p = c["position"]
        span = ax.axvspan if axis == "x" else ax.axhspan
        span(p - 0.5, p + 0.5, color=LINE, alpha=0.35, lw=0, zorder=0)


def idea_map(groups: list[dict], baselines: list[dict], name: str = "idea_map") -> str:
    loaded = load(groups)
    if loaded is None:
        return ""
    clusters, agents, _ = loaded
    rng = random.Random(0)
    n = len(clusters)
    fig, ax = plt.subplots(
        figsize=(8.6, 1.2 + 1.15 * len(agents)), layout="constrained"
    )
    bands(ax, clusters, "x")
    for row, agent in enumerate(reversed(agents)):
        k = len(agent["runs"])
        for i, run in enumerate(agent["runs"]):
            base = row + (i - (k - 1) / 2) * 0.62 / max(k - 1, 1)
            for c in run["cards"]:
                if c.get("final_submission"):
                    continue
                ax.plot(
                    c["y"] + rng.uniform(-0.06, 0.06),
                    base + rng.uniform(-0.03, 0.03),
                    "o",
                    ms=4.2,
                    color=agent["color"],
                    alpha=0.38,
                    mew=0,
                    zorder=2,
                )
            for c in run["cards"]:
                if c.get("final_submission"):
                    ax.plot(
                        c["y"],
                        base,
                        "D",
                        ms=7,
                        color=agent["color"],
                        mec="white",
                        mew=1,
                        zorder=4,
                    )
                    ax.annotate(
                        f"#{run['number']}",
                        (c["y"], base),
                        xytext=(5, 0),
                        textcoords="offset points",
                        va="center",
                        fontsize=6.5,
                        color=INK2,
                        zorder=5,
                    )
    ax.set_yticks(range(len(agents)), [a["label"] for a in reversed(agents)])
    ax.set_ylim(-0.55, len(agents) - 0.45)
    ax.set_xticks(
        [c["position"] for c in clusters],
        [c["label"] for c in clusters],
        rotation=32,
        ha="right",
    )
    ax.set_xlim(0.5, n + 0.5)
    ax.tick_params(axis="y", length=0, labelsize=9.5, labelcolor=INK2)
    ax.grid(False)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("Method family (neighbors are the most similar)", color=MUTED)
    handles = [
        Line2D(
            [],
            [],
            ls="none",
            marker="o",
            ms=5,
            color=MUTED,
            alpha=0.5,
            mew=0,
            label="Iteration tried",
        ),
        Line2D(
            [],
            [],
            ls="none",
            marker="D",
            ms=6.5,
            color=MUTED,
            mec="white",
            label="Final submission (#run)",
        ),
    ]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
    return svg(
        fig, name, "Method families each agent tried, with its final submissions"
    )


def idea_timeline(
    groups: list[dict], baselines: list[dict], name: str = "idea_timeline"
) -> str:
    loaded = load(groups)
    if loaded is None:
        return ""
    clusters, agents, _ = loaded
    hours = max(c["hours"] for a in agents for r in a["runs"] for c in r["cards"])
    fig, axes = plt.subplots(
        1,
        len(agents),
        figsize=(4.4 * len(agents) + 1.4, 4.6),
        sharey=True,
        layout="constrained",
        squeeze=False,
    )
    for ax, agent in zip(axes[0], agents, strict=True):
        bands(ax, clusters, "y")
        k = len(agent["runs"])
        for i, run in enumerate(agent["runs"]):
            lift = (i - (k - 1) / 2) * 0.05  # keep runs in one family apart
            xs = [c["hours"] for c in run["cards"]]
            ys = [c["y"] + lift for c in run["cards"]]
            ax.plot(
                xs,
                ys,
                "-o",
                color=agent["color"],
                alpha=0.3,
                lw=1.4,
                ms=3.6,
                mew=0,
                zorder=2,
            )
            for c in run["cards"]:
                if c.get("final_submission"):
                    ax.plot(
                        c["hours"],
                        c["y"] + lift,
                        "D",
                        ms=7,
                        color=agent["color"],
                        mec="white",
                        mew=1,
                        zorder=4,
                    )
                    ax.annotate(
                        f"#{run['number']}",
                        (c["hours"], c["y"] + lift),
                        xytext=(5, 0),
                        textcoords="offset points",
                        va="center",
                        fontsize=6.5,
                        color=INK2,
                        zorder=5,
                    )
        ax.set_title(agent["label"], fontsize=10, color=INK2, loc="left")
        ax.set_xlim(-0.1, hours + 0.35)
        ax.set_xlabel("Hours since the agent started")
        ax.grid(True, axis="x")
        ax.grid(False, axis="y")
    axes[0][0].set_yticks(
        [c["position"] for c in clusters], [c["label"] for c in clusters]
    )
    axes[0][0].set_ylim(len(clusters) + 0.5, 0.5)
    axes[0][0].tick_params(axis="y", length=0)
    handles = [
        Line2D(
            [],
            [],
            color=MUTED,
            alpha=0.5,
            lw=1.4,
            marker="o",
            ms=3.6,
            mew=0,
            label="One run's iterations, in order",
        ),
        Line2D(
            [],
            [],
            ls="none",
            marker="D",
            ms=6.5,
            color=MUTED,
            mec="white",
            label="Final submission (#run)",
        ),
    ]
    fig.legend(handles=handles, loc="outside upper center", ncol=2)
    for ax in axes[0][1:]:
        ax.tick_params(axis="y", length=0)
    return svg(fig, name, "Method family of each iteration over time, per agent")


# The idea space: every iteration of both agents in one 2-D layout, a blob around
# each method family, one panel per agent, a line from each run's first iteration
# (○) to its final submission (●).
FAMILY_COLORS = [  # by position; the proxy-reward baseline is a neutral gray
    "#8a8a93",
    "#b5452f",
    "#a8792a",
    "#2f4f5a",
    "#2a8a8a",
    "#3b6fb6",
    "#7a4f9a",
    "#3f8f63",
    "#b0607e",
    "#6d6d3f",
    "#4f6f9f",
    "#9f6f4f",
]
PANEL = "#f4f2ee"


def with_references(data: dict) -> dict:
    """data with each reference method (data["references"]: GRAFT) as one more
    card in a family of its own, its judged similarity to every family."""
    data = {
        **data,
        "similarity": {k: dict(v) for k, v in data["similarity"].items()},
        "cards": list(data["cards"]),
    }
    for name, ref in data.get("references", {}).items():
        own = f"ref:{name}"
        data["similarity"][own] = {**ref["similarity"], own: 1.0}
        for c, v in ref["similarity"].items():
            data["similarity"][c][own] = v
        data["cards"].append(
            {
                "run": name,
                "iteration": name,
                "cluster": own,
                "components": ref.get("components", []),
            }
        )
    return data


def layout(data: dict) -> dict[tuple[str, str], tuple[float, float]]:
    """2-D positions of the cards: metric MDS (SMACOF from a classical start) of
    0.65 x (1 - family similarity) + 0.35 x Jaccard distance of their tags."""
    import numpy as np

    cards = data["cards"]
    sim = data["similarity"]
    tags = [set(c["components"]) for c in cards]
    n = len(cards)
    d = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            union = tags[i] | tags[j]
            jac = 1 - len(tags[i] & tags[j]) / len(union) if union else 0
            d[i, j] = d[j, i] = (
                0.65 * (1 - sim[cards[i]["cluster"]][cards[j]["cluster"]]) + 0.35 * jac
            )
    j = np.eye(n) - 1 / n
    b = -0.5 * j @ (d**2) @ j
    w, v = np.linalg.eigh(b)
    x = v[:, -2:] * np.sqrt(np.maximum(w[-2:], 1e-9))
    for _ in range(300):  # SMACOF (Guttman transform, unit weights)
        e = np.sqrt(((x[:, None] - x[None]) ** 2).sum(-1))
        ratio = np.where(e > 1e-9, d / np.maximum(e, 1e-9), 0)
        bm = -ratio
        bm[np.diag_indices(n)] = ratio.sum(1)
        x = bm @ x / n
    x -= x.mean(0)
    _, vecs = np.linalg.eigh(np.cov(x.T))
    x = x @ vecs[:, ::-1]  # principal axis horizontal, to fit a wide panel
    # baseline on the left, as the start of the story
    first = [
        i for i, c in enumerate(cards) if c["cluster"] == data["clusters"][0]["id"]
    ]
    if first and x[first, 0].mean() > 0:
        x[:, 0] *= -1
    rng = np.random.default_rng(0)
    x += rng.normal(0, 0.012 * np.ptp(x[:, 0]), x.shape)  # identical cards apart
    return {(c["run"], c["iteration"]): tuple(p) for c, p in zip(cards, x, strict=True)}


def blob(points, radius: float):
    """A smooth rounded outline around points: the convex hull of a circle around
    each, rounded further by Chaikin corner cutting."""
    import math

    ring = [
        (
            px + radius * math.cos(a * math.pi / 12),
            py + radius * math.sin(a * math.pi / 12),
        )
        for px, py in points
        for a in range(24)
    ]
    ring = sorted(set(ring))

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower, upper = [], []
    for p in ring:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in reversed(ring):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    hull = lower[:-1] + upper[:-1]
    for _ in range(3):
        hull = [
            q
            for a, b in zip(hull, hull[1:] + hull[:1], strict=True)
            for q in (
                (0.75 * a[0] + 0.25 * b[0], 0.75 * a[1] + 0.25 * b[1]),
                (0.25 * a[0] + 0.75 * b[0], 0.25 * a[1] + 0.75 * b[1]),
            )
        ]
    return hull + hull[:1]


def place_labels(
    ax, labels: list, color: dict, radius: float, size: float = 8.5
) -> None:
    """Each family's name beside its blob: above, below, left or right, whichever
    overlaps the names already placed, the other blobs and the panel's title least."""
    fig = ax.figure
    fig.canvas.draw()  # lay the figure out, so display positions are final
    renderer = fig.canvas.get_renderer()
    taken = []  # display-space boxes: placed names and every blob
    blobs = {}
    for c, outline, _ in labels:
        pts = ax.transData.transform(outline)
        blobs[c["id"]] = (
            pts[:, 0].min(),
            pts[:, 1].min(),
            pts[:, 0].max(),
            pts[:, 1].max(),
        )
    title = ax.transAxes.transform([(0, 0.88), (0.62, 1)])
    taken.append((*title[0], *title[1]))

    def overlap(a, b):
        w = min(a[2], b[2]) - max(a[0], b[0])
        h = min(a[3], b[3]) - max(a[1], b[1])
        return max(w, 0) * max(h, 0)

    for c, outline, on in sorted(labels, key=lambda x: -len(x[1])):
        xs = [p[0] for p in outline]
        ys = [p[1] for p in outline]
        cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
        options = [
            ((cx, max(ys)), (0, 4), "center", "bottom"),
            ((cx, min(ys)), (0, -4), "center", "top"),
            ((min(xs), cy), (-5, 0), "right", "center"),
            ((max(xs), cy), (5, 0), "left", "center"),
        ]
        best = None
        for xy, off, ha, va in options:
            text = ax.annotate(
                c["label"],
                xy,
                xytext=off,
                textcoords="offset points",
                ha=ha,
                va=va,
                fontsize=size,
                fontweight="semibold",
                color=color[c["id"]],
                alpha=1 if on else 0.45,
                zorder=6,
            )
            bb = text.get_window_extent(renderer)
            box = (bb.x0, bb.y0, bb.x1, bb.y1)
            panel = ax.get_window_extent(renderer)
            cost = sum(overlap(box, o) for o in taken) + sum(
                overlap(box, b) for k, b in blobs.items() if k != c["id"]
            )
            cost += 1e6 * (
                box[0] < panel.x0
                or box[2] > panel.x1
                or box[1] < panel.y0
                or box[3] > panel.y1
            )
            if best is None or cost < best[0]:
                if best is not None:
                    best[1].remove()
                best = (cost, text, box)
            else:
                text.remove()
        taken.append(best[2])


def label_runs(ax, points: list[tuple[str, float, float]]) -> None:
    """Each final submission's run number beside its dot, on the first side where
    it covers no other final dot and no number already placed."""
    import matplotlib.patheffects as pe

    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    scale = fig.dpi / 72
    dots = [ax.transData.transform((x, y)) for _, x, y in points]
    placed = []
    sides = [
        (7, 3, "left", "bottom"),
        (7, -3, "left", "top"),
        (-7, 3, "right", "bottom"),
        (-7, -3, "right", "top"),
        (0, 8, "center", "bottom"),
        (0, -8, "center", "top"),
    ]
    for (text, x, y), here in zip(points, dots, strict=True):
        best = None
        for dx, dy, ha, va in sides:
            t = ax.annotate(
                text,
                (x, y),
                xytext=(dx, dy),
                textcoords="offset points",
                ha=ha,
                va=va,
                fontsize=7,
                color="#23343b",
                zorder=8,
                path_effects=[pe.withStroke(linewidth=2.2, foreground="white")],
            )
            bb = t.get_window_extent(renderer)
            r = 6 * scale  # a final dot's radius, with a little room
            hits = sum(
                1
                for d in dots
                if d is not here
                and bb.x0 - r < d[0] < bb.x1 + r
                and bb.y0 - r < d[1] < bb.y1 + r
            ) + sum(1 for b in placed if bb.overlaps(b))
            if best is None or hits < best[0]:
                if best is not None:
                    best[1].remove()
                best = (hits, t, bb)
            else:
                t.remove()
            if hits == 0:
                break
        placed.append(best[2])


def idea_space(
    groups: list[dict], baselines: list[dict], name: str = "idea_space"
) -> str:
    import matplotlib.patheffects as pe

    loaded = load(groups)
    if loaded is None:
        return ""
    clusters, agents, cards = loaded
    data = json.loads(DATA.read_text())
    where = layout(data)
    color = {
        c["id"]: FAMILY_COLORS[i % len(FAMILY_COLORS)] for i, c in enumerate(clusters)
    }
    xs = [p[0] for p in where.values()]
    ys = [p[1] for p in where.values()]
    span = max(max(xs) - min(xs), max(ys) - min(ys))
    radius = 0.045 * span
    members = {
        c["id"]: [
            where[(k["run"], k["iteration"])] for k in cards if k["cluster"] == c["id"]
        ]
        for c in clusters
    }
    fig, axes = plt.subplots(
        len(agents),
        1,
        figsize=(9.2, 3.5 * len(agents)),
        layout="constrained",
        squeeze=False,
    )
    for ax, agent in zip(axes[:, 0], agents, strict=True):
        ax.set_facecolor(PANEL)
        ax.set_xlim(min(xs) - 5 * radius, max(xs) + 5 * radius)
        ax.set_ylim(min(ys) - 3.5 * radius, max(ys) + 6 * radius)
        visited = {k["cluster"] for r in agent["runs"] for k in r["cards"]}
        labels = []
        for c in clusters:
            pts = members[c["id"]]
            if not pts:
                continue
            outline = blob(pts, radius)
            on = c["id"] in visited
            ax.fill(
                *zip(*outline, strict=True),
                color=color[c["id"]],
                alpha=0.13 if on else 0.05,
                lw=0,
                zorder=1,
            )
            ax.plot(
                *zip(*outline, strict=True),
                color=color[c["id"]],
                lw=2.0 if on else 1.0,
                alpha=1 if on else 0.35,
                ls="-" if on else (0, (3, 2)),
                zorder=2,
                solid_capstyle="round",
            )
            labels.append((c, outline, on))
        place_labels(ax, labels, color, radius)
        for run in agent["runs"]:
            for k in run["cards"]:
                x, y = where[(k["run"], k["iteration"])]
                ax.plot(
                    x,
                    y,
                    "o",
                    ms=4.6,
                    color=color[k["cluster"]],
                    alpha=0.55,
                    mew=0,
                    zorder=3,
                )
        halo = [pe.Stroke(linewidth=3.6, foreground="white", alpha=0.85), pe.Normal()]
        ends = []
        for run in agent["runs"]:
            start = run["cards"][0]
            end = next(
                (k for k in run["cards"] if k.get("final_submission")), run["cards"][-1]
            )
            (x0, y0), (x1, y1) = (
                where[(start["run"], start["iteration"])],
                where[(end["run"], end["iteration"])],
            )
            ax.plot(
                [x0, x1],
                [y0, y1],
                color="#23343b",
                lw=1.5,
                alpha=0.8,
                zorder=4,
                path_effects=halo,
                solid_capstyle="round",
            )
            ax.plot(x0, y0, "o", ms=7, mfc="white", mec="#23343b", mew=1.6, zorder=5)
            ends.append((run, end, x1, y1))
        for _, end, x1, y1 in ends:
            ax.plot(
                x1,
                y1,
                "o",
                ms=10,
                color=color[end["cluster"]],
                mec="#23343b",
                mew=1.4,
                zorder=7,
            )
        label_runs(ax, [(f"#{run['number']}", x1, y1) for run, _, x1, y1 in ends])
        ax.text(
            0.012,
            0.965,
            agent["label"],
            transform=ax.transAxes,
            va="top",
            fontsize=11,
            fontweight="semibold",
            color="#23343b",
        )
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(False)
    handles = [
        Line2D(
            [],
            [],
            ls="none",
            marker="o",
            ms=5,
            color=MUTED,
            alpha=0.6,
            mew=0,
            label="Iteration",
        ),
        Line2D(
            [],
            [],
            ls="none",
            marker="o",
            ms=7,
            mfc="white",
            mec="#23343b",
            mew=1.6,
            label="Run's first iteration",
        ),
        Line2D(
            [],
            [],
            ls="none",
            marker="o",
            ms=9,
            color=MUTED,
            mec="#23343b",
            mew=1.4,
            label="Final submission (#run)",
        ),
    ]
    fig.legend(handles=handles, loc="outside lower center", ncol=3)
    return svg(fig, name, "Idea space: method families both agents explored, per agent")


# ---------------------------------------------------------------------------
# The four idea figures in their own style: warm colors for the reward-driven
# (RL) families, cool ones for the imitation families, gray for the baseline;
# ★ marks a final submission everywhere.
FAMILY = {
    "C1": "#8c8c8c",
    "C2": "#a4243b",
    "C3": "#d8572a",
    "C4": "#e8a33d",
    "C5": "#7f4f24",
    "C6": "#1d4e89",
    "C7": "#6a4c93",
    "C8": "#2a9d8f",
    "C9": "#5fa8d3",
}
TEXT, SOFT, RULE = INK, MUTED, LINE  # the other figures' ink, muted text and rules
STAR = dict(marker="*", ms=13, mec="white", mew=0.9, zorder=8)


def family_color(cluster_id: str, i: int) -> str:
    return FAMILY.get(cluster_id, FAMILY_COLORS[i % len(FAMILY_COLORS)])


def styled(ax, grid: str | None = None) -> None:
    ax.set_facecolor("white")
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(RULE)
    ax.tick_params(colors=SOFT, length=0)
    ax.grid(False)
    if grid is not None:
        ax.grid(True, axis=grid, color=RULE, lw=0.6)
    ax.set_axisbelow(True)


def family_legend(
    fig,
    clusters: list[dict],
    colors: dict,
    ncol: int = 3,
    y: float = 0.0,
    extra: list | None = None,
) -> None:
    """The families as legend rows under the figure, grouped and in order, then
    any extra handles (what the marks mean)."""
    from matplotlib.patches import Patch

    handles = [Patch(color=colors[c["id"]], label=c["label"]) for c in clusters]
    fig.legend(
        handles=handles + (extra or []),
        loc="outside lower center",
        ncol=ncol,
        handlelength=1.1,
        handleheight=1.1,
        columnspacing=1.6,
    )


def two_lines(text: str, longest: int = 20) -> str:
    """text as is if short, else split at the word boundary that balances the lines."""
    words = text.split()
    if len(text) <= longest or len(words) < 2:
        return text
    splits = [(" ".join(words[:i]), " ".join(words[i:])) for i in range(1, len(words))]
    first, second = min(splits, key=lambda s: max(len(s[0]), len(s[1])))
    return f"{first}\n{second}"


def name_clouds(ax, items: list, colors: dict, avoid: list) -> None:
    """Each family's name next to its own cloud: above it, below, left or right,
    whichever covers the fewest names already placed and other clouds' centers
    (long names wrap to two lines). Names stay readable over the clouds: a family
    the agent tried is named in a darker shade of its color, one it never tried in
    gray."""
    from matplotlib.colors import to_rgb

    def darker(color: str, by: float = 0.35) -> tuple:
        return tuple(v * (1 - by) for v in to_rgb(color))

    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    pad = 20 * fig.dpi / 72  # about a cloud's visible radius
    panel = ax.get_window_extent(renderer)
    taken = [a.get_window_extent(renderer) for a in avoid]
    taken.append(ax.title.get_window_extent(renderer))
    cores = {}
    for c, pts, _ in items:
        d = ax.transData.transform(pts)
        cores[c["id"]] = (
            d[:, 0].min() - 8,
            d[:, 1].min() - 8,
            d[:, 0].max() + 8,
            d[:, 1].max() + 8,
        )

    def overlap(b, o) -> float:
        x0, y0, x1, y1 = (o.x0, o.y0, o.x1, o.y1) if hasattr(o, "x0") else o
        return max(min(b.x1, x1) - max(b.x0, x0), 0) * max(
            min(b.y1, y1) - max(b.y0, y0), 0
        )

    for c, _, on in sorted(items, key=lambda x: -len(x[1])):
        x0, y0, x1, y1 = cores[c["id"]]
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        text = two_lines(c["label"])
        options = [
            ((cx, y1 + pad - 8), "center", "bottom"),
            ((cx, y0 - pad + 8), "center", "top"),
            ((x0 - pad + 8, cy), "right", "center"),
            ((x1 + pad - 8, cy), "left", "center"),
        ]
        best = None
        inverse = ax.transData.inverted()
        for rank, (spot, ha, va) in enumerate(options):
            label = ax.text(
                *inverse.transform(spot),
                text,
                ha=ha,
                va=va,
                fontsize=8,
                fontweight="semibold",
                linespacing=1.05,
                color=darker(colors[c["id"]]) if on else MUTED,
                zorder=12,
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.9),
            )
            b = label.get_window_extent(renderer)
            cost = 20 * sum(overlap(b, o) for o in taken)
            cost += sum(overlap(b, core) for k, core in cores.items() if k != c["id"])
            cost += 1e9 * (
                b.x0 < panel.x0 or b.x1 > panel.x1 or b.y0 < panel.y0 or b.y1 > panel.y1
            )
            cost += 40 * rank  # prefer above, then below
            if best is None or cost < best[0]:
                if best is not None:
                    best[1].remove()
                best = (cost, label, b)
            else:
                label.remove()
        taken.append(best[2])


def draw_clouds(ax, clouds: list, sigma_x: float, sigma_y: float) -> None:
    """Soft clouds, one per family, behind the points: each point drawn as three
    large faint discs in the family's color, so they pile up where points crowd.
    (The SVG reuses one disc symbol, so this stays small.)"""
    del sigma_x, sigma_y
    for pts, color, strength in clouds:
        for size, alpha in ((2600, 0.035), (1300, 0.05), (500, 0.06)):
            ax.scatter(
                *zip(*pts, strict=True),
                s=size,
                color=color,
                lw=0,
                zorder=1,
                alpha=alpha * strength / 0.42,
            )


def landscape(
    groups: list[dict], baselines: list[dict], name: str = "ideas_landscape"
) -> str:
    """Figure: the shared map of ideas, one panel per agent; each family a soft
    cloud, each iteration a dot, an arrow from each run's first idea to its ★."""
    from matplotlib.patches import FancyArrowPatch

    loaded = load(groups)
    if loaded is None:
        return ""
    clusters, agents, cards = loaded
    data = with_references(json.loads(DATA.read_text()))
    where = layout(data)
    references = data.get("references", {})
    lookalike = {
        (g["run"], g["iteration"])
        for ref in references.values()
        for g in ref.get("graft_like", [])
    }
    colors = {c["id"]: family_color(c["id"], i) for i, c in enumerate(clusters)}
    xs = [p[0] for p in where.values()]
    ys = [p[1] for p in where.values()]
    sx, sy = max(xs) - min(xs), max(ys) - min(ys)
    # One agent per row, stacked, so each panel spans the page's width and its
    # labels read at the page's text size.
    fig, axes = plt.subplots(
        len(agents),
        1,
        figsize=(8.4, 5.6 * len(agents)),
        layout="constrained",
        squeeze=False,
    )
    to_name: list = []
    for ax, agent in zip(axes[:, 0], agents, strict=True):
        styled(ax)
        ax.grid(False)
        ax.spines["bottom"].set_visible(False)
        visited = {k["cluster"] for r in agent["runs"] for k in r["cards"]}
        names: list = []
        clouds: list = []
        to_name.append((ax, names, avoid := []))
        ax.set_xlim(min(xs) - 0.12 * sx, max(xs) + 0.12 * sx)
        ax.set_ylim(min(ys) - 0.12 * sy, max(ys) + 0.2 * sy)
        for c in clusters:  # clouds from every card of the family, both agents
            pts = [
                where[(k["run"], k["iteration"])]
                for k in cards
                if k["cluster"] == c["id"]
            ]
            if not pts:
                continue
            on = c["id"] in visited
            clouds.append((pts, colors[c["id"]], 0.42 if on else 0.16))
            names.append((c, pts, on))
        draw_clouds(ax, clouds, 0.05 * sx, 0.07 * sy)
        for run in agent["runs"]:
            for k in run["cards"]:
                x, y = where[(k["run"], k["iteration"])]
                ax.plot(
                    x,
                    y,
                    "o",
                    ms=4,
                    color=colors[k["cluster"]],
                    mec="white",
                    mew=0.5,
                    zorder=4,
                )
        for i, run in enumerate(agent["runs"]):
            start = run["cards"][0]
            end = next(
                (k for k in run["cards"] if k.get("final_submission")), run["cards"][-1]
            )
            a, b = (
                where[(start["run"], start["iteration"])],
                where[(end["run"], end["iteration"])],
            )
            ax.add_patch(
                FancyArrowPatch(
                    a,
                    b,
                    connectionstyle=f"arc3,rad={0.18 if i % 2 else -0.18}",
                    arrowstyle="-|>,head_length=3.5,head_width=2",
                    lw=0.9,
                    color=TEXT,
                    alpha=0.35,
                    shrinkA=3,
                    shrinkB=7,
                    zorder=5,
                )
            )
            ax.plot(*a, "o", ms=5.5, mfc="white", mec=TEXT, mew=1.1, zorder=6)
            ax.plot(*b, color=colors[end["cluster"]], **STAR)
        for run in agent["runs"]:
            for k in run["cards"]:
                if (k["run"], k["iteration"]) in lookalike:
                    ax.plot(
                        *where[(k["run"], k["iteration"])],
                        "s",
                        ms=10,
                        mfc="none",
                        mec=TEXT,
                        mew=1.3,
                        zorder=7,
                    )
        for ref in references:
            x, y = where[(ref, ref)]
            ax.plot(x, y, "D", ms=10, color=TEXT, mec="white", mew=1.2, zorder=11)
            avoid.append(
                ax.annotate(
                    ref,
                    (x, y),
                    xytext=(8, -1),
                    textcoords="offset points",
                    va="center",
                    fontsize=9,
                    fontweight="semibold",
                    color=TEXT,
                    zorder=11,
                    bbox=dict(
                        boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.72
                    ),
                )
            )
        ax.set_title(
            agent["label"],
            loc="left",
            fontsize=10.5,
            fontweight="semibold",
            color=TEXT,
            pad=8,
        )
        ax.set_xlim(min(xs) - 0.12 * sx, max(xs) + 0.12 * sx)
        ax.set_ylim(min(ys) - 0.12 * sy, max(ys) + 0.2 * sy)
        ax.set_xticks([])
        ax.set_yticks([])

    handles = [
        Line2D(
            [],
            [],
            ls="none",
            marker="o",
            ms=5,
            color=SOFT,
            mec="white",
            label="An idea tried (iteration)",
        ),
        Line2D(
            [],
            [],
            ls="none",
            marker="o",
            ms=6,
            mfc="white",
            mec=TEXT,
            mew=1.1,
            label="Run's first idea",
        ),
        Line2D(
            [],
            [],
            ls="none",
            marker="*",
            ms=12,
            color=SOFT,
            mec="white",
            label="Run's final submission",
        ),
    ]
    if references:
        handles += [
            Line2D(
                [],
                [],
                ls="none",
                marker="D",
                ms=8,
                color=TEXT,
                mec="white",
                label="GRAFT (published method)",
            ),
            Line2D(
                [],
                [],
                ls="none",
                marker="s",
                ms=9,
                mfc="none",
                mec=TEXT,
                mew=1.3,
                label="GRAFT-like idea",
            ),
        ]
    fig.legend(
        handles=handles,
        loc="outside lower center",
        ncol=len(handles),
        columnspacing=1.6,
    )
    for ax, names, avoid in to_name:
        name_clouds(ax, names, colors, avoid)
    return svg(
        fig,
        name,
        "Map of the ideas each agent tried, with each run's path to its submission",
    )


def mix(groups: list[dict], baselines: list[dict], name: str = "ideas_mix") -> str:
    """Figure: the share of each agent's iterations in each family, one panel per
    agent side by side, families down the left; ★ one per final submission."""
    loaded = load(groups)
    if loaded is None:
        return ""
    clusters, agents, _ = loaded
    colors = {c["id"]: family_color(c["id"], i) for i, c in enumerate(clusters)}
    fig, axes = plt.subplots(
        1,
        len(agents),
        figsize=(3.3 * len(agents) + 2.6, 0.4 * len(clusters) + 1.3),
        sharey=True,
        layout="constrained",
        squeeze=False,
    )
    shares = []
    for agent in agents:
        cards = [k for r in agent["runs"] for k in r["cards"]]
        share = {
            c["id"]: sum(k["cluster"] == c["id"] for k in cards) / len(cards)
            for c in clusters
        }
        finals = {
            c["id"]: sum(
                bool(k.get("final_submission")) and k["cluster"] == c["id"]
                for k in cards
            )
            for c in clusters
        }
        shares.append((agent, share, finals, len(cards)))
    top = max(v for _, s, _, _ in shares for v in s.values())
    for ax, (agent, share, finals, n) in zip(axes[0], shares, strict=True):
        styled(ax, "x")
        for c in clusters:
            y, v = c["position"], share[c["id"]]
            ax.barh(y, v, height=0.68, color=colors[c["id"]], lw=0, zorder=3)
            if v:
                label = f"{v:.0%}" + (
                    "   " + "★" * finals[c["id"]] if finals[c["id"]] else ""
                )
                ax.annotate(
                    label,
                    (v, y),
                    xytext=(4, 0),
                    textcoords="offset points",
                    va="center",
                    fontsize=8,
                    color=TEXT,
                )
        for i in range(1, len(clusters)):
            if clusters[i].get("group") != clusters[i - 1].get("group"):
                ax.axhline(clusters[i]["position"] - 0.5, color=RULE, lw=1, zorder=1)
        ax.set_xlim(0, top * 1.55)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
        ax.set_title(
            f"{agent['label']}  ·  {n} iterations, {len(agent['runs'])} runs",
            loc="left",
            fontsize=10.5,
            fontweight="semibold",
            color=TEXT,
        )
    first = axes[0][0]
    first.set_yticks(
        [c["position"] for c in clusters], [c["label"] for c in clusters], fontsize=8.5
    )
    for tick, c in zip(first.get_yticklabels(), clusters, strict=True):
        tick.set_color(colors[c["id"]])
        tick.set_fontweight("bold")
    first.set_ylim(len(clusters) + 0.5, 0.5)
    for c0 in clusters:  # the group names, beside the first family of each group
        prev = [c for c in clusters if c["position"] == c0["position"] - 1]
        if not prev or prev[0].get("group") != c0.get("group"):
            first.annotate(
                c0.get("group", "").upper(),
                (0, c0["position"] - 0.5),
                xycoords=("axes fraction", "data"),
                xytext=(-4, -1),
                textcoords="offset points",
                ha="right",
                va="top",
                fontsize=6.5,
                color=SOFT,
            )
    fig.supxlabel(
        "Share of the agent's iterations in each method family   ·   ★ one final submission",
        color=SOFT,
        fontsize=8.5,
    )
    return svg(fig, name, "Share of each agent's iterations per method family")


def ribbons(
    groups: list[dict], baselines: list[dict], name: str = "ideas_timeline"
) -> str:
    """Figure: one ribbon per run over its hours, colored by the family of the idea
    it was on (from one iteration's first run to the next's); ★ when it submitted
    its final answer, a tick at the run's end."""
    loaded = load(groups)
    if loaded is None:
        return ""
    clusters, agents, cards = loaded
    used = {card["cluster"] for card in cards}
    clusters = [cluster for cluster in clusters if cluster["id"] in used]
    clusters = [{**c, "label": FAMILY_LABELS.get(c["id"], c["label"])} for c in clusters]
    colors = {c["id"]: family_color(c["id"], i) for i, c in enumerate(clusters)}
    hours_of = {
        r["label"]: (r.get("agent_seconds") or 0) / 3600
        for g in groups
        for r in g["runs"]
    }
    rows = []
    for agent in agents:
        rows += [(agent["label"], run) for run in agent["runs"]]
        rows.append(None)  # a gap between agents
    rows = rows[:-1]
    fig, ax = plt.subplots(figsize=(8.6, 0.3 * len(rows) + 2.2), layout="constrained")
    styled(ax, "x")
    longest = 0.0
    labels = []
    for y, row in enumerate(rows):
        if row is None:
            continue
        agent_label, run = row
        cards = run["cards"]
        end = max(hours_of.get(run["label"], 0), cards[-1]["hours"] + 0.05)
        longest = max(longest, end)
        for k, nxt in zip(cards, cards[1:] + [None], strict=True):
            x1 = nxt["hours"] if nxt else end
            ax.barh(
                y,
                max(x1 - k["hours"], 0.012),
                left=k["hours"],
                height=0.62,
                color=colors[k["cluster"]],
                lw=0,
                zorder=3,
            )
            ax.plot(
                [k["hours"]] * 2, [y - 0.31, y + 0.31], color="white", lw=0.7, zorder=4
            )
        final = next((k for k in cards if k.get("final_submission")), None)
        if final:
            ax.plot(
                final["hours"],
                y - 0.48,
                color=colors[final["cluster"]],
                **{**STAR, "ms": 11},
            )
        ax.plot([end, end], [y - 0.36, y + 0.36], color=TEXT, lw=1, zorder=4)
        labels.append((y, run["label"]))
    ax.set_yticks(
        [y for y, _ in labels], [t for _, t in labels], fontsize=9, color=INK2
    )
    ax.set_ylim(len(rows) - 0.3, -0.9)
    ax.set_xlim(0, longest + 0.1)
    ax.set_xlabel("Hours since the agent started")
    marks = [
        marker_handle("Final method first tried", "*", ms=10),
        Line2D(
            [], [], ls="none", marker="|", ms=9, mew=1.2, color=TEXT, label="End of run"
        ),
    ]
    family_legend(fig, clusters, colors, ncol=4, extra=marks)
    return svg(fig, name, "Each run's ideas over time, colored by method family")


def similarity(
    groups: list[dict], baselines: list[dict], name: str = "ideas_similarity"
) -> str:
    """Figure: how alike the method families are (the judged similarity, 0 to 1),
    lower triangle, families in their order; colored swatches name them."""

    import numpy as np
    from matplotlib.colors import LinearSegmentedColormap

    if not DATA.exists():
        return ""
    data = json.loads(DATA.read_text())
    clusters = sorted(data["clusters"], key=lambda c: c["position"])
    colors = {c["id"]: family_color(c["id"], i) for i, c in enumerate(clusters)}
    ids = [c["id"] for c in clusters]
    n = len(ids)
    m = np.array([[data["similarity"][a][b] for b in ids] for a in ids], dtype=float)
    shown = np.where(np.tril(np.ones((n, n)), -1) > 0, m, np.nan)
    cmap = LinearSegmentedColormap.from_list(
        "sim", ["#ffffff", "#c9d6e3", "#5b7ea4", "#1f3b5c"]
    )
    fig, ax = plt.subplots(figsize=(7.4, 6.2), layout="constrained")
    styled(ax)
    ax.grid(False)
    ax.spines["bottom"].set_visible(False)
    image = ax.imshow(shown, cmap=cmap, vmin=0, vmax=1)
    for i in range(n):
        for j in range(i):
            v = m[i, j]
            ax.text(
                j,
                i,
                f"{v:.2f}".lstrip("0") if v < 1 else "1",
                ha="center",
                va="center",
                fontsize=7.5,
                color="white" if v > 0.55 else TEXT,
            )
    names = [c["label"] for c in clusters]
    ax.set_yticks(range(n), names, fontsize=8.5)
    ax.set_xticks(range(n), names, rotation=40, ha="right", fontsize=8.5)
    for tick, c in zip(ax.get_yticklabels(), clusters, strict=True):
        tick.set_color(colors[c["id"]])
        tick.set_fontweight("bold")
    for tick, c in zip(ax.get_xticklabels(), clusters, strict=True):
        tick.set_color(colors[c["id"]])
        tick.set_fontweight("bold")
    # where one kind of approach ends and the next begins
    kinds = [c.get("group", "") for c in clusters]
    for i in range(1, n):
        if kinds[i] != kinds[i - 1]:
            ax.plot([-0.5, i - 0.5], [i - 0.5, i - 0.5], color=TEXT, lw=0.8, alpha=0.5)
            ax.plot(
                [i - 0.5, i - 0.5], [i - 0.5, n - 0.5], color=TEXT, lw=0.8, alpha=0.5
            )
    bar = fig.colorbar(image, ax=ax, shrink=0.6, pad=0.02)
    bar.outline.set_visible(False)
    bar.ax.tick_params(colors=SOFT, length=0, labelsize=8)
    bar.set_label(
        "How alike (judged, 0 = unrelated, 1 = same method)", color=SOFT, fontsize=8.5
    )
    ax.set_xlim(-0.5, n - 1.5)
    ax.set_ylim(n - 0.5, 0.5)
    return svg(fig, name, "Judged similarity between the method families")


FAMILY_LABELS = {
    "C1": "RL on raw reward", "C2": "Trusted-derived reward cap",
    "C3": "RL + extra hack detectors", "C4": "RL on trusted + monitor elsewhere",
    "C5": "RL on trusted prompts only", "C6": "Imitation + RL",
    "C7": "Imitation + negative loss", "C8": "Imitation only",
    "C9": "Imitation + pseudo-labels",
}

def coverage(groups: list[dict], baselines: list[dict], name: str = "ideas_landscape") -> str:
    """Discrete run-by-family coverage. No inferred spatial geometry or paths."""
    from matplotlib.patches import Rectangle
    loaded = load(groups)
    if loaded is None:
        return ""
    clusters, agents, cards = loaded
    used = {card["cluster"] for card in cards}
    clusters = [cluster for cluster in clusters if cluster["id"] in used]
    # Wider with each agent, so every agent's columns stay as wide as with two.
    fig, ax = plt.subplots(figsize=(2.6 + 2.9 * len(agents), 0.4 * len(clusters) + 1.3), layout="constrained")
    col = 0
    for agent in agents:
        first = col
        for run in sorted(agent["runs"], key=lambda run: -run["score"]):
            visits = {}
            previous, step = None, 0
            for card in run["cards"]:
                if card["cluster"] != previous:
                    step += 1
                    visits.setdefault(card["cluster"], []).append(step)
                    previous = card["cluster"]
            visited = set(visits)
            final = next((c["cluster"] for c in run["cards"] if c.get("final_submission")), None)
            for row, cluster in enumerate(clusters):
                ax.add_patch(Rectangle((col-.44,row-.40),.88,.80,
                                      facecolor="#f5f5f7",edgecolor="none",zorder=0))
                if cluster["id"] in visited:
                    ax.add_patch(Rectangle((col-.44,row-.40),.88,.80,
                                          facecolor=agent["color"],alpha=.16,edgecolor="none",zorder=1))
                    nums = visits[cluster["id"]]
                    label = "\n".join(",".join(map(str, nums[i:i+2])) for i in range(0,len(nums),2))
                    ax.text(col,row,label,ha="center",va="center",fontsize=9.5,
                            color=INK,linespacing=1.1,zorder=3)
                if cluster["id"] == final:
                    ax.add_patch(Rectangle((col-.44,row-.40),.88,.80,
                                          facecolor="none",edgecolor=INK,linewidth=1.2,zorder=4))
            ax.text(col,len(clusters)+.15,f"{run['score']:.3f}",ha="center",va="center",fontsize=8,color=INK2)
            col += 1
        ax.text((first+col-1)/2,-1.1,two_lines(agent["label"], 12),ha="center",va="center",fontsize=12,fontweight="semibold",color=INK)
        col+=.65
    ax.text(-.7,len(clusters)+.15,"Held-out score",ha="right",va="center",fontsize=11,color=MUTED)
    ax.set_xticks([])
    ax.set_yticks(range(len(clusters)),[FAMILY_LABELS.get(c["id"],c["label"]) for c in clusters],fontsize=11)
    ax.tick_params(axis="y",length=0,pad=8)
    ax.set_xlim(-.55,col-1.1);ax.set_ylim(len(clusters)+.6,-.65)
    ax.grid(False)
    for spine in ax.spines.values():spine.set_visible(False)
    from matplotlib.patches import Patch
    handles=[Patch(facecolor="white",edgecolor=INK,label="Final method’s family"),
             Patch(facecolor="#f5f5f7",edgecolor="none",label="No recorded attempt")]
    fig.legend(handles=handles,loc="outside lower center",ncol=2,fontsize=10)
    return svg(fig,name,"Runs sorted by held-out score, with numbered method-family visits and outlined final families")


# The idea figures the page shows ({{figure:NAME}} in post/results.md).
PAGE = {"ideas_landscape": coverage, "ideas_timeline": ribbons}
