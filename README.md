# Optimal Research

Harbor research tasks for evaluating how AI agents run experiments and improve their methods.

| Task | Report | Sandbox |
| --- | --- | --- |
| Looping Language Models | [Report](https://optimalresearch.github.io/looped-lm-pretrain/) | [Quickstart](src/looped-lm-pretrain/README.md#quickstart) |
| Learning without forgetting | [Report](https://optimalresearch.github.io/learning-forgetting/) | [Quickstart](src/learning-forgetting/README.md#quickstart) |
| Mitigating Reward Hacking | [Report](https://posttrain.ai/mitigating-reward-hacking/) | [Setup](https://github.com/vballoli/autoresearch-safety#setup) |

Each sandbox includes the agent prompt, Harbor task configuration, Dockerfile, and the code and data the agent can use. Scored Harbor runs require the separate verifier and evaluation data.

## Website

Open `docs/index.html` locally or serve the site with `python3 -m http.server --directory docs 8000`. GitHub Pages publishes `docs/` from `main`. Each report’s **Code** link points to its sandbox, either under `src/` or in the task’s source repository.
