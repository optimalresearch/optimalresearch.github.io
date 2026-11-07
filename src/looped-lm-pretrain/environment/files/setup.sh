#!/bin/bash
# Harbor Singularity backend hook, sourced at container start (environment/files/
# is bound at /staging/env_files). No-op on backends that built the image.
# 1. Public task files, if the image did not bake them.
if [ ! -f /task/CONTRACT.md ]; then
  mkdir -p /task && cp -r /staging/env_files/task/. /task/
  chmod -R a+rX /task 2>/dev/null || true   # read-only binds under /task (models) are fine
fi
find /task -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null || true
mkdir -p /logs/agent /logs/verifier
# The research stack. The Dockerfile bakes /opt/venv and puts it on PATH; on the
# cluster it is bound in, and commands may not inherit this shell's PATH, so make
# the venv's interpreters win through /usr/local/bin (writable tmpfs).
if [ -d /opt/venv/bin ]; then
  export PATH="/opt/venv/bin:$PATH"
  for b in python3 python pip pip3; do
    [ -x "/opt/venv/bin/$b" ] && ln -sfn "/opt/venv/bin/$b" "/usr/local/bin/$b" 2>/dev/null
  done
  # Harbor's in-container server prepends /usr/bin to PATH for every command, so
  # /usr/bin/python3 must see the venv too: the venv was created with that very
  # interpreter, so a .pth in its site-packages is exactly equivalent.
  site="$(/usr/bin/python3 -c 'import site; print(site.getsitepackages()[0])' 2>/dev/null)"
  vsite="$(ls -d /opt/venv/lib/python3.*/site-packages 2>/dev/null | head -1)"
  if [ -n "$site" ] && [ -n "$vsite" ]; then
    mkdir -p "$site" && echo "$vsite" > "$site/zz-opt-venv.pth" 2>/dev/null || true
  fi
fi
# 2. Workspace on the host side. /staging is this sandbox's private host directory
#    (shared filesystem), so files under it can be bound into GPU jobs by the rent
#    bridge. Make /workspace live there instead of in the container's tmpfs.
if [ -d /staging ]; then
  mkdir -p /staging/workspace/output /staging/rent/requests /staging/rent/jobs
  # The shim binds /staging/workspace at /workspace; fall back to a symlink otherwise.
  if ! mountpoint -q /workspace 2>/dev/null; then
    rm -rf /workspace && ln -s /staging/workspace /workspace
  fi
fi
# 3. The layout the agent's prompt describes (submission/, data/lm/, runs/, snapshots/, timer.sh,
#    /opt/lmenv-src); the submission starts as a copy of the starting point.
bash /task/make_workspace.sh || echo "setup.sh: make_workspace.sh failed" >&2
