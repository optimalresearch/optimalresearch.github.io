#!/usr/bin/env python3
"""Write a model's weights as a submission: one safetensors file in the base
checkpoint's layout (every tensor of the manifest, trained ones replaced).

    from export import export_submission
    export_submission(model, "/workspace/output/model.safetensors")

Tensors the model does not hold (vision tower, MTP head) are copied from the
base checkpoint unchanged. Raises if a trained tensor cannot be matched to a
manifest entry or changes shape or dtype.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import contract  # noqa: E402

MANIFEST = json.loads(contract.MANIFEST.read_text())["tensors"]


def checkpoint_key(name: str) -> str | None:
    """Map a transformers state-dict key to the checkpoint key (text-only models drop `language_model.`)."""
    for cand in (name, name.replace("model.", "model.language_model.", 1)):
        if cand in MANIFEST:
            return cand
    return None


def export_submission(model, out_path, strict: bool = True) -> None:
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file
    tensors = {}
    for shard in sorted(contract.BASE_MODEL_DIR.glob("model.safetensors*.safetensors")):
        with safe_open(str(shard), "pt") as f:
            for k in f.keys():
                tensors[k] = f.get_tensor(k)
    unmatched = []
    for name, p in model.state_dict().items():
        if name == "lm_head.weight" and getattr(model.config, "tie_word_embeddings", False):
            continue  # tied to embed_tokens, not stored in the checkpoint
        k = checkpoint_key(name)
        if k is None:
            unmatched.append(name)
            continue
        want = MANIFEST[k]
        if list(p.shape) != want["shape"]:
            raise ValueError(f"{k}: shape {list(p.shape)} != manifest {want['shape']}")
        tensors[k] = p.detach().to(tensors[k].dtype).cpu().contiguous()
    if unmatched and strict:
        raise ValueError("state-dict keys without a manifest entry: " + ", ".join(unmatched[:5]))
    save_file(tensors, str(out_path), metadata={"format": "pt"})
