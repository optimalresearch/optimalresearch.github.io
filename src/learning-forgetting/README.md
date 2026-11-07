# Learning without forgetting

This is the agent sandbox for a Harbor task. See [task.toml](task.toml) for the Harbor configuration.

## Quickstart

Use a Linux x86-64 machine with [uv](https://docs.astral.sh/uv/getting-started/installation/), Docker with NVIDIA GPU support, and one H100 (80 GB).

```bash
git clone --depth 1 https://github.com/optimalresearch/optimalresearch.github.io.git &&
cd optimalresearch.github.io/src/learning-forgetting &&
uvx --from huggingface-hub hf download Qwen/Qwen3.5-4B \
  --revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a \
  --local-dir "$HOME/.cache/optimalresearch/Qwen3.5-4B" &&
docker build -t optimalresearch-learning-forgetting environment &&
docker run --gpus 1 -it \
  --mount "type=bind,src=$HOME/.cache/optimalresearch/Qwen3.5-4B,dst=/task/models/Qwen3.5-4B,readonly" \
  optimalresearch-learning-forgetting bash
```

Docker opens a shell in `/workspace`. The [agent prompt](instruction.md), [sandbox contract](environment/files/task/CONTRACT.md), and [public data](environment/files/task/public/) are included. Dev and held-out scoring require separate evaluator code and data.
