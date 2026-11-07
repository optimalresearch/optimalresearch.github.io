# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pyarrow", "tokenizers", "huggingface-hub"]
# ///
"""Build the sandbox's dev tokens with `uv run prepare_dev_data.py`.

Downloads one shard (pinned revision), tokenizes it (or trains a byte-level BPE tokenizer with an
8192-token vocabulary on the first documents when --tokenizer is not given) and writes

  tokenizer.json
  train.bin   uint16 token IDs, documents separated by <|endoftext|>
  val.bin     the same, from row groups at the end of the shard (disjoint from train.bin)
  meta.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

REPO = "HuggingFaceFW/fineweb-edu"
SHARD = "sample/10BT/000_00000.parquet"
EOT = "<|endoftext|>"
PUBLIC = Path(__file__).resolve().parent / "environment" / "files" / "task" / "public"


def texts(table_file, row_groups):
    for i in row_groups:
        yield from table_file.read_row_group(i, columns=["text"]).column("text").to_pylist()


def write_tokens(tokenizer, docs, out: Path, target: int) -> int:
    eot = tokenizer.token_to_id(EOT)
    written, chunk = 0, []
    with out.open("wb") as f:
        for doc in docs:
            chunk.append(doc)
            if len(chunk) < 2048:
                continue
            for enc in tokenizer.encode_batch(chunk):
                ids = np.asarray(enc.ids + [eot], dtype=np.uint16)
                f.write(ids.tobytes())
                written += len(ids)
            chunk = []
            if written >= target:
                break
    return written


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--vocab", type=int, default=8192)
    parser.add_argument("--train-tokens", type=float, default=320e6)
    parser.add_argument("--val-tokens", type=float, default=4e6)
    parser.add_argument("--val-groups", type=int, default=4, help="row groups at the end of the shard held out for validation")
    parser.add_argument("--tokenizer-docs", type=int, default=200_000)
    parser.add_argument("--shard", default=SHARD)
    parser.add_argument("--repo", default=REPO, help="Hugging Face dataset, e.g. HuggingFaceFW/fineweb")
    parser.add_argument("--tokenizer", help="reuse this tokenizer.json instead of training one")
    parser.add_argument("--revision", help="dataset revision (commit) to download; default: the latest")
    parser.add_argument("--cache-dir", help="Hugging Face download cache (default: OUT/hf)")
    args = parser.parse_args(sys.argv[1:] or [
        "--out", str(PUBLIC),
        "--tokenizer", str(PUBLIC / "tokenizer.json"),
        "--revision", "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9",
        "--train-tokens", "250e6", "--val-tokens", "5e6", "--val-groups", "10",
        "--cache-dir", str(Path.home() / ".cache" / "optimalresearch" / "hf"),
    ])
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    path = hf_hub_download(args.repo, args.shard, repo_type="dataset", revision=args.revision,
                           cache_dir=args.cache_dir or str(out / "hf"))
    shard = pq.ParquetFile(path)
    groups = list(range(shard.num_row_groups))
    print(f"{shard.metadata.num_rows} documents in {len(groups)} row groups", flush=True)

    if args.tokenizer:
        tokenizer = Tokenizer.from_file(args.tokenizer)
    else:
        t = time.time()
        tokenizer = Tokenizer(models.BPE())
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tokenizer.decoder = decoders.ByteLevel()
        trainer = trainers.BpeTrainer(vocab_size=args.vocab, special_tokens=[EOT], min_frequency=2,
                                      initial_alphabet=pre_tokenizers.ByteLevel.alphabet())

        def first_docs():
            for i, doc in enumerate(texts(shard, groups)):
                if i >= args.tokenizer_docs:
                    return
                yield doc

        tokenizer.train_from_iterator(first_docs(), trainer)
        print(f"tokenizer trained in {time.time() - t:.0f} s", flush=True)
    tokenizer_path = out / "tokenizer.json"
    if not args.tokenizer or Path(args.tokenizer).resolve() != tokenizer_path.resolve():
        tokenizer.save(str(tokenizer_path))

    # Validation documents come from the last row groups, training documents from the first.
    val_groups = groups[-args.val_groups:]
    t = time.time()
    n_val = write_tokens(tokenizer, texts(shard, val_groups), out / "val.bin", int(args.val_tokens))
    n_train = write_tokens(tokenizer, texts(shard, groups[:-args.val_groups]), out / "train.bin", int(args.train_tokens))
    print(f"tokenized {n_train / 1e6:.0f}M train and {n_val / 1e6:.1f}M val tokens in {time.time() - t:.0f} s")
    (out / "meta.json").write_text(json.dumps({"source": f"{args.repo}/{args.shard}", "vocab_size": tokenizer.get_vocab_size(),
                                               "eot": tokenizer.token_to_id(EOT), "train_tokens": n_train,
                                               "val_tokens": n_val}, indent=2))


if __name__ == "__main__":
    main()
