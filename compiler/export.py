#!/usr/bin/env python3
"""Export an existing AESOP token-RNN checkpoint to the TI runtime header."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from compiler.token_rnn import export_header

DEFAULT_CHECKPOINT = Path("aesop/checkpoints/model.npz")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
    )
    parser.add_argument("--out", type=Path, default=Path("runtime/src/generated_model.h"))
    parser.add_argument("--name", default=None)
    args = parser.parse_args()

    data = np.load(args.checkpoint, allow_pickle=True)
    model = {key: data[key] for key in ("emb", "rec", "bh", "out", "bo")}
    if "qscale" in data.files:
        model["qscale"] = data["qscale"]
    vocab = [str(piece) for piece in data["vocab"].tolist()]
    export_header(args.out, model, vocab, args.name or args.checkpoint.stem)
    print(f"header={args.out}")


if __name__ == "__main__":
    main()
