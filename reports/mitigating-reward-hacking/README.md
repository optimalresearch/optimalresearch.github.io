# Mitigating Reward Hacking report

Build the report and update its homepage charts from the repository root:

```bash
uv run --no-project --with markdown --with matplotlib python reports/mitigating-reward-hacking/build.py
python3 -m http.server --directory docs 8766
```

The build writes `docs/mitigating-reward-hacking/` and updates this task’s row in `docs/index.html`. GitHub Pages serves `docs/` from `main` at https://posttrain.ai/.

## Editing

- `post/task.md`, `results.md`, and `analysis.md` contain the report text.
- `post/post.toml` configures the run groups, figure captions, and reference results.
- `data/trace_editorial.json` contains each trace’s method summary, scientific process, and experiment descriptions.
- `data/runs.json` holds the recorded metrics, and `data/ideas.json` holds the method classifications and timings.
- `data/instruction.md` preserves the prompt used for these runs.
- `data/traces/` preserves the recorded sessions. The build applies editorial summaries and checks their links into the sessions.

The configured groups produce four Opus trace pages and one each for Sol and Astra. A normal build uses the cached data and traces; `--collect` and `--rebuild-traces` require the original job directories.

`uncertainty.py` computes approximate evaluation intervals from cached aggregate metrics. Its comments document the assumptions and distinguish evaluation uncertainty from variation across research runs.

The environment and its setup instructions live in [vballoli/autoresearch-safety](https://github.com/vballoli/autoresearch-safety).
