# Sandbox image. The verifier reuses it, so evaluation dependencies belong here.
# Python 3.12 from Ubuntu 24.04; the pinned research stack in /opt/venv, first
# on PATH (so `python3` and `pip` are the venv's). Add task-specific packages
# to requirements.txt, never ad hoc here, so the pin set stays in one file.
FROM ubuntu:24.04
ENV DEBIAN_FRONTEND=noninteractive LANG=C.UTF-8 PATH=/opt/venv/bin:/usr/local/bin:/usr/bin:/bin
RUN apt-get update && apt-get install -y --no-install-recommends \
      python3 python3-venv python3-pip curl ca-certificates git procps tmux build-essential \
    && rm -rf /var/lib/apt/lists/*
# Harbor's Singularity backend needs its exec server deps preinstalled; harmless elsewhere.
RUN python3 -m venv /opt/harbor-server && /opt/harbor-server/bin/pip install -q uvicorn fastapi
COPY requirements.txt /opt/requirements.txt
RUN python3 -m venv /opt/venv && /opt/venv/bin/pip install --no-cache-dir \
      --index-url https://download.pytorch.org/whl/cu130 --extra-index-url https://pypi.org/simple \
      -r /opt/requirements.txt
# vLLM for the verifier's GSM8K generation, in its own venv (it pins its own torch).
COPY requirements-vllm.txt /opt/requirements-vllm.txt
RUN python3 -m venv /opt/vllm && /opt/vllm/bin/pip install --no-cache-dir -r /opt/requirements-vllm.txt
# Public task files: contract, validator, budget, dev data.
COPY files/task/ /task/
RUN mkdir -p /workspace/output /logs/agent /logs/verifier && chmod -R a+rX /task
WORKDIR /workspace
