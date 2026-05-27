"""Reference model contract for AESOP-style TI token generators.

The compiler-facing contract is deliberately step-based:

    logits, next_state = model(token, state)

`token` is the previous token id and `state` is the recurrent state. The current
runtime ships a quantized tanh RNN matching `compiler.token_rnn`; future models
can keep this API while the compiler grows more lowerings.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TokenRNNConfig:
    vocab_size: int = 384
    hidden_size: int = 96
    qscale: int = 64
    latent_buckets: int = 48
    latent_repeat: int = 4
