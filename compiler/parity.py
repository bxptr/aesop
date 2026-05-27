#!/usr/bin/env python3
"""Host-side integer parity smoke checks for exported token-RNN checkpoints."""

from __future__ import annotations

import argparse
from pathlib import Path

from aesop.eval_sampler import RuntimeSampler, SamplerConfig

DEFAULT_CHECKPOINT = Path("aesop/checkpoints/model.npz")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "checkpoint",
        nargs="?",
        type=Path,
        default=DEFAULT_CHECKPOINT,
    )
    parser.add_argument("--seed", type=int, default=1000)
    args = parser.parse_args()

    sampler = RuntimeSampler(
        args.checkpoint,
        SamplerConfig(
            name="parity-smoke",
            margin=64,
            repeat_penalty=96,
            weights=(10, 8, 6, 4, 3, 2, 1, 1),
            force_first_once=True,
            local_repairs=True,
        ),
    )
    print(sampler.sample(args.seed, chars=520))


if __name__ == "__main__":
    main()
