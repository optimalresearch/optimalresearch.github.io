# Looping Language Models

See [the agent prompt](instruction.md) and the [starter submission](environment/files/task/starter/). The sandbox builds from `environment/Dockerfile` after generating the public development tokens.

From this directory, run:

```bash
python3 -m pip install numpy pyarrow tokenizers huggingface_hub
python3 prepare_dev_data.py \
  --out environment/files/task/public \
  --tokenizer environment/files/task/public/tokenizer.json \
  --revision 87f09149ef4734204d70ed1d046ddc9ca3f2b8f9 \
  --train-tokens 250e6 --val-tokens 5e6 --val-groups 10 \
  --cache-dir "$HOME/.cache/optimalresearch/hf"
docker build -t optimalresearch-looped-lm-pretrain environment
```

This prepares the agent-facing FineWeb-Edu split. The published runs used an H200 GPU.
