# Optimal Research

Research environments for evaluating how AI agents run experiments and improve their methods.

| Task | Report | Sandbox |
| --- | --- | --- |
| Looping Language Models | [Report](https://optimalresearch.github.io/looped-lm-pretrain/) | [Source](src/looped-lm-pretrain/) |
| Learning and Forgetting | [Report](https://optimalresearch.github.io/learning-forgetting/) | [Source](src/learning-forgetting/) |

The sandbox directories contain the agent prompt, task configuration, container build files, and agent-facing code and data. Reproducing held-out rewards requires the sealed evaluator and evaluation datasets.

## Website

Open `docs/index.html` locally or serve the site with `python3 -m http.server --directory docs 8000`. GitHub Pages publishes `docs/` from `main`. Each report's **Code** link points to its sandbox under `src/`.
