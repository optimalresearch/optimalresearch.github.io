# Learning and Forgetting

See [the agent prompt](instruction.md) and the [sandbox contract](environment/files/task/CONTRACT.md). The development facts and supporting corpora are in [the public data directory](environment/files/task/public/).

From this directory, build the sandbox with `docker build -t optimalresearch-learning-forgetting environment`. The published runs used an H100 GPU. The sandbox expects the Qwen3.5-4B weights and tokenizer at `/task/models/Qwen3.5-4B`.
