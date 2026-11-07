# The agent image uses PyTorch 2.7.1, the agent CLIs, and the lmenv harness installed from /opt/lmenv-src.
FROM pytorch/pytorch:2.7.1-cuda12.6-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive PYTHONDONTWRITEBYTECODE=1
RUN apt-get update -qq \
 && apt-get install -y -qq --no-install-recommends gcc libc6-dev git curl procps ca-certificates less tmux ripgrep xz-utils \
 && rm -rf /var/lib/apt/lists/*
RUN curl -fsSL https://nodejs.org/dist/v22.12.0/node-v22.12.0-linux-x64.tar.xz | tar -xJ -C /usr/local --strip-components=1 \
 && npm install -g --no-fund --no-audit @openai/codex @anthropic-ai/claude-code

# Task files (contract, validator, harness, starting point, dev text: public/train.bin and val.bin are built
# by prepare_dev_data.py first). The harness is installed from its own copy.
COPY files/task/ /task/
COPY pyproject.toml /opt/lmenv-src/pyproject.toml
RUN test -f /task/public/train.bin && test -f /task/public/val.bin \
 && cp -r /task/lmenv /opt/lmenv-src/lmenv \
 && pip install --no-cache-dir -e /opt/lmenv-src && pip install --no-cache-dir pytest \
 && mkdir -p /logs/agent /logs/verifier && bash /task/make_workspace.sh && chmod -R a+rX /task
WORKDIR /workspace
