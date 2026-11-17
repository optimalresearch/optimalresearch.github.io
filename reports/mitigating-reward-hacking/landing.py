"""Update this task's homepage row using its cached results."""

import html
import math
import re
from pathlib import Path

from uncertainty import evaluation_noise_band


HERE = Path(__file__).resolve().parent
WIDTH, HEIGHT = 320, 114


def chart(title: str, description: str, marks: list[str]) -> str:
    return (f'<figure class="spark"><figcaption>{html.escape(title)}</figcaption>'
            f'<svg viewBox="0 0 {WIDTH} {HEIGHT}" role="img" aria-label="{html.escape(description)}">'
            f'{"".join(marks)}</svg><div class="tip" hidden></div></figure>')


def mean_rewards(groups: list[dict]) -> str:
    left, right, top, bottom = 108, 306, 8, 96
    rows = sorted(groups, key=lambda group: -sum(run["held_out"]["mean"]["score"]
                  for run in group["runs"]) / len(group["runs"]))
    marks, descriptions = [], []
    for tick in (0, .25, .5, .75, 1):
        x = left + tick * (right - left)
        marks.append(f'<line class="grid" x1="{x:.1f}" x2="{x:.1f}" y1="{top}" y2="{bottom}"/>'
                     f'<text class="tick" x="{x:.1f}" y="110" text-anchor="middle">{tick:g}</text>')
    for i, group in enumerate(rows):
        mean = sum(run["held_out"]["mean"]["score"] for run in group["runs"]) / len(group["runs"])
        y = top + (i + .5) * (bottom - top) / len(rows)
        label = html.escape(group["label"])
        logo = (HERE / "templates" / f'{group["slug"]}.svg').read_text().strip()
        marks.append(f'<g transform="translate(0 {y - 7:.1f})">{logo}</g>'
                     f'<text class="st-l" x="20" y="{y + 4:.1f}">{label}</text>'
                     f'<rect class="bar {group["color"]}" x="{left}" y="{y - 7:.1f}" '
                     f'width="{max(0, mean) * (right - left):.1f}" height="14" tabindex="0" '
                     f'data-tip="{label} {mean:.3f}"/>')
        descriptions.append(f'{group["label"]} {mean:.3f}')
    return chart("Held-out score by model", "; ".join(descriptions), marks)


def passk(group: dict) -> str:
    scores = sorted(run["held_out"]["mean"]["score"] for run in group["runs"])
    n = len(scores)
    ks = [k for k in (1, 2, 4, 8) if k <= n]
    values = [sum(score * math.comb(i, k - 1) for i, score in enumerate(scores)) / math.comb(n, k)
              for k in ks]
    band = evaluation_noise_band(group["runs"])
    low = max(0, math.floor((min(values + [lo for _, lo, _ in band]) - .02) * 10) / 10)
    high = min(1, math.ceil((max(values + [hi for _, _, hi in band]) + .02) * 10) / 10)
    left, right, top, bottom = 35, 306, 8, 96
    points = [(left + math.log2(k) * (right - left) / math.log2(ks[-1]),
               bottom - (value - low) * (bottom - top) / (high - low)) for k, value in zip(ks, values)]
    marks = []
    for value in (low, (low + high) / 2, high):
        y = bottom - (value - low) * (bottom - top) / (high - low)
        marks.append(f'<line class="grid" x1="{left}" x2="{right}" y1="{y:.1f}" y2="{y:.1f}"/>'
                     f'<text class="tick" x="{left - 7}" y="{y + 4:.1f}" text-anchor="end">{value:.2f}</text>')
    def band_point(k, value):
        x = left + math.log2(k) * (right - left) / math.log2(ks[-1])
        y = bottom - (value - low) * (bottom - top) / (high - low)
        return f'{x:.1f},{y:.1f}'

    outline = [band_point(k, hi) for k, _, hi in band]
    outline += [band_point(k, lo) for k, lo, _ in reversed(band)]
    marks.append(f'<polygon class="noise-band" points="{" ".join(outline)}" '
                 'fill="var(--base)" fill-opacity=".35" tabindex="0" '
                 'data-tip="Estimated evaluation noise: 5th–95th percentile under an equal-quality Gaussian model; '
                 'noise scale bounded from archived reward and hack metrics."/>')
    label_y = bottom - (band[-1][1] - low) * (bottom - top) / (high - low) + 14
    marks.append(f'<text class="tick" x="{right}" y="{label_y:.1f}" text-anchor="end">'
                 'Baseline from noise</text>')
    line = " ".join(f'{x:.1f},{y:.1f}' for x, y in points)
    marks.append(f'<polyline class="line {group["color"]}" points="{line}"/>')
    for k, value, (x, y) in zip(ks, values, points):
        marks.append(f'<text class="tick" x="{x:.1f}" y="110" text-anchor="middle">{k}</text>'
                     f'<circle class="dot {group["color"]}" cx="{x:.1f}" cy="{y:.1f}" r="4" '
                     f'tabindex="0" data-tip="Pass@{k}: {value:.3f}"/>')
    x, y = points[-1]
    label_y = y - 8 if y >= top + 12 else y + 15
    marks.append(f'<text class="st-v" x="{x + 4:.1f}" y="{label_y:.1f}" text-anchor="end">{values[-1]:.3f}</text>')
    title = f'{group["label"]} pass@k'
    description = title + " (log₂ x-axis), " + ", ".join(f'k={k}: {value:.3f}' for k, value in zip(ks, values))
    description += ". Shaded band estimates the baseline from evaluation noise."
    return chart(title, description, marks)


def update(groups: list[dict], title: str, path: Path) -> None:
    plots = mean_rewards(groups) + passk(next(group for group in groups if group["slug"] == "opus"))
    row = ('<article class="task" id="mitigating-reward-hacking"><div class="t-main">'
           '<a class="t-title" href="mitigating-reward-hacking/index.html">'
           f'<span class="t-label">{html.escape(title)}</span>&#8239;'
           '<svg class="t-action" viewBox="0 0 20 20" aria-hidden="true" focusable="false">'
           '<path d="M11.5 3.5h5v5m0-5-7 7M8.5 4.5h-4a1 1 0 0 0-1 1v10a1 1 0 0 0 1 1h10a1 1 0 0 0 1-1v-4"/>'
           f'</svg></a></div><div class="t-plots">{plots}</div></article>')
    homepage = path.read_text()
    pattern = re.compile(r'<article class="task" id="mitigating-reward-hacking">.*?</article>', re.S)
    if pattern.search(homepage):
        homepage = pattern.sub(lambda _: row, homepage)
    else:
        end = '</article></div></section>'
        assert homepage.count(end) == 1, "Cannot locate the homepage task list"
        homepage = homepage.replace(end, '</article>' + row + '</div></section>')
    path.write_text(homepage)
