# Looping Language Models

This is the agent sandbox for a Harbor task. See [task.toml](task.toml) for the Harbor configuration.

## Quickstart

Use a Linux x86-64 machine with [uv](https://docs.astral.sh/uv/getting-started/installation/), Docker with NVIDIA GPU support, and one H200 (141 GB). Data preparation downloads a 2.2 GB FineWeb-Edu shard.

```bash
git clone --depth 1 https://github.com/optimalresearch/optimalresearch.github.io.git &&
cd optimalresearch.github.io/src/looped-lm-pretrain &&
uv run prepare_dev_data.py &&
docker build -t optimalresearch-looped-lm-pretrain environment &&
docker run --gpus 1 -it optimalresearch-looped-lm-pretrain bash
```

Docker opens a shell in `/workspace` with the [starter model](environment/files/task/starter/) and tokenized dev data. The [agent prompt](instruction.md) lists the experiment commands. A scored Harbor run requires the separate verifier and held-out data.
