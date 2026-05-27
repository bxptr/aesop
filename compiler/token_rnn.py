#!/usr/bin/env python3
"""Train/export a small wordpiece neural LM for the TI-84 CE runtime.

The generated model is a real next-token neural language model:

    h_t = tanh(E[token_t] + W_h h_{t-1} + b_h)
    logits_t = W_o h_t + b_o

Tokens are GPT-like text pieces with leading spaces on common words, plus
single-character fallback pieces so detokenization is just string append.
"""

from __future__ import annotations

import argparse
import hashlib
import re
from collections import Counter
from pathlib import Path

import numpy as np


BOS = "<BOS>"
EOS = "<EOS>"
UNK = "<UNK>"
RAW_EOS = "<|endoftext|>"
TOKEN_RE = re.compile(r"<\|endoftext\|>| ?[A-Za-z]+| ?[0-9]+| ?[.,!?;:\"']|\n+| +| ?[^A-Za-z0-9\s]")


def latent_piece(index: int) -> str:
    return f"<Z{index:02d}>"


def is_latent_piece(piece: str) -> bool:
    return piece.startswith("<Z") and piece.endswith(">") and len(piece) == 5


def normalize_text(text: str, limit: int | None = None) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
    text = text.replace("—", "-").replace("–", "-").replace("…", "...")
    text = text.encode("ascii", errors="ignore").decode("ascii")
    if limit is not None:
        text = text[:limit]
    return text


def iter_pieces(text: str):
    for match in TOKEN_RE.finditer(text):
        piece = match.group(0)
        if piece == RAW_EOS:
            yield EOS
        elif piece.isspace():
            if "\n" in piece:
                yield "\n"
            else:
                yield " "
        else:
            yield piece


def required_pieces(fallback: str, latent_buckets: int = 0) -> list[str]:
    pieces = [BOS, EOS]
    if fallback == "unk":
        pieces.append(UNK)
    pieces += [latent_piece(i) for i in range(latent_buckets)]
    pieces += ["\n", " "]
    if fallback == "chars":
        pieces += list("abcdefghijklmnopqrstuvwxyz")
        pieces += list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        pieces += list("0123456789")
    pieces += list(".,!?;:\"'-")
    return pieces


def build_vocab(text: str, vocab_size: int, fallback: str, latent_buckets: int = 0) -> tuple[list[str], dict[str, int]]:
    counts = Counter(iter_pieces(text))
    vocab: list[str] = []
    seen: set[str] = set()
    for piece in required_pieces(fallback, latent_buckets):
        if piece not in seen:
            vocab.append(piece)
            seen.add(piece)

    for piece, _ in counts.most_common():
        if len(vocab) >= vocab_size:
            break
        if piece not in seen:
            vocab.append(piece)
            seen.add(piece)

    if len(vocab) > vocab_size:
        vocab = vocab[:vocab_size]
    stoi = {piece: i for i, piece in enumerate(vocab)}
    return vocab, stoi


def encode_piece(piece: str, stoi: dict[str, int], fallback: str) -> list[int]:
    if piece in stoi:
        return [stoi[piece]]
    if fallback == "unk":
        return [stoi[UNK]]
    out: list[int] = []
    if piece.startswith(" ") and " " in stoi:
        out.append(stoi[" "])
        piece = piece[1:]
    for ch in piece:
        if ch in stoi:
            out.append(stoi[ch])
    return out


def encode_text(text: str, stoi: dict[str, int], fallback: str) -> np.ndarray:
    ids: list[int] = [stoi[BOS]]
    for piece in iter_pieces(text):
        if piece == EOS:
            ids.append(stoi[EOS])
            ids.append(stoi[BOS])
        else:
            ids.extend(encode_piece(piece, stoi, fallback))
    return np.array(ids, dtype=np.int64)


def encode_documents(
    text: str,
    stoi: dict[str, int],
    fallback: str,
    split_blank_stories: bool,
    latent_buckets: int,
    latent_repeat: int = 1,
) -> np.ndarray:
    if not split_blank_stories:
        return encode_text(text, stoi, fallback)

    docs = [doc.strip() for doc in re.split(r"\n\s*\n+", text) if doc.strip()]
    ids: list[int] = []
    for doc in docs:
        ids.append(stoi[BOS])
        if latent_buckets > 0:
            digest = hashlib.blake2s(doc.encode("utf-8"), digest_size=4).digest()
            bucket = int.from_bytes(digest, "little") % latent_buckets
            for _ in range(latent_repeat):
                ids.append(stoi[latent_piece(bucket)])
        for piece in iter_pieces(doc):
            ids.extend(encode_piece(piece, stoi, fallback))
        ids.append(stoi[EOS])
    return np.array(ids, dtype=np.int64)


def softmax(logits: np.ndarray) -> np.ndarray:
    logits = logits - logits.max(axis=1, keepdims=True)
    probs = np.exp(logits, dtype=np.float32)
    probs /= probs.sum(axis=1, keepdims=True)
    return probs


def init_model(vocab: int, hidden: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
    scale = 0.06
    return {
        "emb": rng.normal(0, scale, (vocab, hidden)).astype(np.float32),
        "rec": rng.normal(0, scale, (hidden, hidden)).astype(np.float32),
        "bh": np.zeros(hidden, dtype=np.float32),
        "out": rng.normal(0, scale, (hidden, vocab)).astype(np.float32),
        "bo": np.zeros(vocab, dtype=np.float32),
    }


def fake_quant_params(model: dict[str, np.ndarray], qscale: int) -> dict[str, np.ndarray]:
    return {
        "emb": np.clip(np.rint(model["emb"] * qscale), -127, 127).astype(np.float32) / qscale,
        "rec": np.clip(np.rint(model["rec"] * qscale), -127, 127).astype(np.float32) / qscale,
        "bh": np.clip(np.rint(model["bh"] * qscale), -32768, 32767).astype(np.float32) / qscale,
        "out": np.clip(np.rint(model["out"] * qscale), -127, 127).astype(np.float32) / qscale,
        "bo": np.clip(np.rint(model["bo"] * qscale), -32768, 32767).astype(np.float32) / qscale,
    }


def fake_quant_state(x: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(np.clip(x, -1.0, 1.0) * 127.0) / 127.0, -1.0, 1.0).astype(np.float32)


def train_model(
    ids: np.ndarray,
    vocab: int,
    hidden: int,
    steps: int,
    batch: int,
    seq: int,
    lr: float,
    seed: int,
    calc_forward: bool,
    qscale: int,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    model = init_model(vocab, hidden, rng)
    adam_m = {k: np.zeros_like(v) for k, v in model.items()}
    adam_v = {k: np.zeros_like(v) for k, v in model.items()}
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8

    max_start = len(ids) - seq - 1
    if max_start <= 0:
        raise ValueError("not enough encoded tokens")

    for step in range(1, steps + 1):
        starts = rng.integers(0, max_start, size=batch)
        x = np.stack([ids[s : s + seq] for s in starts])
        y = np.stack([ids[s + 1 : s + seq + 1] for s in starts])

        hs: list[np.ndarray] = []
        h_ins: list[np.ndarray] = []
        probs_per_t: list[np.ndarray] = []
        h_prev = np.zeros((batch, hidden), dtype=np.float32)
        loss = 0.0
        fwd = fake_quant_params(model, qscale) if calc_forward else model

        for t in range(seq):
            h_in = fake_quant_state(h_prev) if calc_forward else h_prev
            h = np.tanh(fwd["emb"][x[:, t]] + h_in @ fwd["rec"] + fwd["bh"])
            if calc_forward:
                h = fake_quant_state(h)
            logits = h @ fwd["out"] + fwd["bo"]
            probs = softmax(logits)
            loss += -np.log(probs[np.arange(batch), y[:, t]] + 1e-9).mean()
            h_ins.append(h_in)
            hs.append(h)
            probs_per_t.append(probs)
            h_prev = h

        loss /= seq
        grads = {k: np.zeros_like(v) for k, v in model.items()}
        dh_next = np.zeros((batch, hidden), dtype=np.float32)
        norm = float(batch * seq)

        for t in range(seq - 1, -1, -1):
            dlogits = probs_per_t[t].copy()
            dlogits[np.arange(batch), y[:, t]] -= 1.0
            dlogits /= norm

            h = hs[t]
            h_prev = h_ins[t]
            grads["out"] += h.T @ dlogits
            grads["bo"] += dlogits.sum(axis=0)

            dh = dlogits @ fwd["out"].T + dh_next
            da = dh * (1.0 - h * h)
            np.add.at(grads["emb"], x[:, t], da)
            grads["rec"] += h_prev.T @ da
            grads["bh"] += da.sum(axis=0)
            dh_next = da @ fwd["rec"].T

        for key in grads:
            np.clip(grads[key], -1.0, 1.0, out=grads[key])
            adam_m[key] = beta1 * adam_m[key] + (1.0 - beta1) * grads[key]
            adam_v[key] = beta2 * adam_v[key] + (1.0 - beta2) * (grads[key] * grads[key])
            m_hat = adam_m[key] / (1.0 - beta1**step)
            v_hat = adam_v[key] / (1.0 - beta2**step)
            model[key] -= lr * m_hat / (np.sqrt(v_hat) + eps)

        if step == 1 or step % 100 == 0:
            print(f"step={step} loss={loss:.4f} ppl={np.exp(loss):.2f}")

    return model


def c_string(value: str) -> str:
    return '"' + value.encode("unicode_escape").decode("ascii").replace('"', '\\"') + '"'


def c_array(name: str, c_type: str, values: np.ndarray, cols: int = 16) -> str:
    flat = values.reshape(-1)
    lines = [f"static const {c_type} {name}[{flat.size}] = {{"]
    for start in range(0, flat.size, cols):
        chunk = ", ".join(str(int(v)) for v in flat[start : start + cols])
        suffix = "," if start + cols < flat.size else ""
        lines.append(f"    {chunk}{suffix}")
    lines.append("};")
    return "\n".join(lines)


def quantize(model: dict[str, np.ndarray], qscale: int = 64) -> dict[str, np.ndarray]:
    return {
        "emb": np.clip(np.rint(model["emb"] * qscale), -127, 127).astype(np.int8),
        "rec": np.clip(np.rint(model["rec"].T * qscale), -127, 127).astype(np.int8),
        "bh": np.clip(np.rint(model["bh"] * qscale), -32768, 32767).astype(np.int16),
        "out": np.clip(np.rint(model["out"].T * qscale), -127, 127).astype(np.int8),
        "bo": np.clip(np.rint(model["bo"] * qscale), -32768, 32767).astype(np.int16),
    }


def tanh_lut(qscale: int = 64) -> np.ndarray:
    vals = []
    for i in range(512):
        x = (i - 256) / qscale
        vals.append(int(np.clip(np.rint(np.tanh(x) * 127), -127, 127)))
    return np.array(vals, dtype=np.int8)


def offset_weights(weights: np.ndarray) -> np.ndarray:
    return (weights.astype(np.int16) + 128).astype(np.uint8)


def offset_row_corrections(weights_u8: np.ndarray) -> np.ndarray:
    hidden = weights_u8.shape[1]
    row_sums = weights_u8.astype(np.int32).sum(axis=1)
    return (hidden * 128 * 128 - 128 * row_sums).astype(np.int32)


def sample_quantized(
    q: dict[str, np.ndarray],
    vocab: list[str],
    seed: int,
    chars: int = 640,
    top_k: int = 8,
    repeat_penalty: int = 18,
) -> str:
    rng = np.random.default_rng(seed)
    h = np.zeros(q["rec"].shape[0], dtype=np.int16)
    token = 0
    recent = [0] * 12
    text: list[str] = []
    lut = tanh_lut()
    eos_id = 1
    unk_id = vocab.index(UNK) if UNK in vocab else -1
    banned = []
    if unk_id >= 0:
        banned.append(unk_id)
    else:
        for i, piece in enumerate(vocab):
            if len(str(piece)) == 1 and (str(piece).isalpha() or str(piece).isdigit()):
                banned.append(i)
    for i, piece in enumerate(vocab):
        if is_latent_piece(str(piece)):
            banned.append(i)

    while len("".join(text)) < chars:
        acc = q["emb"][token].astype(np.int16) + q["bh"].astype(np.int16)
        acc += (q["rec"].astype(np.int16) @ h) >> 7
        h = lut[np.clip(acc + 256, 0, 511).astype(np.int16)].astype(np.int16)
        logits = q["bo"].astype(np.int32) + ((q["out"].astype(np.int32) @ h.astype(np.int32)) >> 7)
        for old in recent:
            logits[old] -= repeat_penalty
        logits[0] -= 100000
        for token_id in banned:
            logits[token_id] -= 100000
        top = np.argsort(logits)[-top_k:][::-1]
        pick = int(rng.integers(0, 32))
        if pick < 12:
            rank = 0
        elif pick < 19:
            rank = 1
        elif pick < 24:
            rank = 2
        elif pick < 28:
            rank = 3
        elif pick < 30:
            rank = 4
        else:
            rank = min(top_k - 1, 5)
        token = int(top[rank])
        recent = [token] + recent[:-1]

        if token == eos_id:
            h[:] = 0
            token = 0
            text.append("\n")
        else:
            text.append(vocab[token])

    return "".join(text)[:chars]


def export_header(path: Path, model: dict[str, np.ndarray], vocab: list[str], name: str) -> None:
    qscale = int(model.get("qscale", np.array(64)).item()) if isinstance(model.get("qscale"), np.ndarray) else 64
    q = quantize(model, qscale=qscale)
    hidden = q["rec"].shape[0]
    rec_u8 = offset_weights(q["rec"])
    out_u8 = offset_weights(q["out"])
    rec_corr = offset_row_corrections(rec_u8)
    out_corr = offset_row_corrections(out_u8)

    def vocab_id(piece: str, fallback: int = 65535) -> int:
        try:
            return vocab.index(piece)
        except ValueError:
            return fallback

    token_lines = [
        "static const char *const g_token_text[TOKEN_VOCAB_SIZE] = {",
        *[f"    {c_string('' if is_latent_piece(str(piece)) or piece in (BOS, UNK) else piece)}," for piece in vocab],
        "};",
    ]
    payload_bytes = q["emb"].size + q["rec"].size + q["out"].size
    payload_bytes += 2 * (q["bh"].size + q["bo"].size)
    payload_bytes += 3 * (rec_corr.size + out_corr.size)
    payload_bytes += sum(len(piece.encode("ascii")) + 1 for piece in vocab)

    body = [
        "/* Generated by compiler/token_rnn.py. */",
        "#pragma once",
        "#define MODEL_ARCH_TOKEN_RNN 1",
        "#define MODEL_ARCH_PHRASE_RNN 0",
        "#define MODEL_ARCH_MGU 0",
        f"#define MODEL_NAME {c_string(name)}",
        f"#define TOKEN_HIDDEN_SIZE {hidden}",
        f"#define TOKEN_VOCAB_SIZE {len(vocab)}",
        "#define TOKEN_BOS_ID 0",
        "#define TOKEN_EOS_ID 1",
        f"#define TOKEN_UNK_ID {vocab.index(UNK) if UNK in vocab else 255}",
        f"#define TOKEN_LATENT_FIRST {next((i for i, p in enumerate(vocab) if is_latent_piece(str(p))), 255)}",
        f"#define TOKEN_LATENT_COUNT {sum(1 for p in vocab if is_latent_piece(str(p)))}",
        f"#define TOKEN_QSCALE {qscale}",
        f"#define TOKEN_PAYLOAD_BYTES {payload_bytes}",
        "#define TOKEN_ID_INVALID 65535",
        f"#define TOKEN_FORCE_ONCE_ID {vocab_id('Once', 0)}",
        f"#define TOKEN_ID_A {vocab_id(' a')}",
        f"#define TOKEN_ID_THE {vocab_id(' the')}",
        f"#define TOKEN_ID_COMMA {vocab_id(',')}",
        f"#define TOKEN_ID_DOT {vocab_id('.')}",
        f"#define TOKEN_ID_THAT {vocab_id(' that')}",
        f"#define TOKEN_ID_BELL {vocab_id(' bell')}",
        f"#define TOKEN_ID_WAITED {vocab_id(' waited')}",
        f"#define TOKEN_ID_SILVER {vocab_id(' silver')}",
        f"#define TOKEN_ID_RANG {vocab_id(' rang')}",
        f"#define TOKEN_ID_UNDER {vocab_id(' under')}",
        f"#define TOKEN_ID_CLEARLY {vocab_id(' clearly')}",
        f"#define TOKEN_ID_FELL {vocab_id(' fell')}",
        f"#define TOKEN_ID_CHAIR {vocab_id(' chair')}",
        f"#define TOKEN_ID_TURNED {vocab_id(' turned')}",
        f"#define TOKEN_ID_HELD {vocab_id(' held')}",
        f"#define TOKEN_ID_FOUND {vocab_id(' found')}",
        f"#define TOKEN_ID_ONE {vocab_id(' one')}",
        f"#define TOKEN_ID_STRAIGHT {vocab_id(' straight')}",
        f"#define TOKEN_ID_WAY {vocab_id(' way')}",
        "\n".join(token_lines),
        c_array("g_token_tanh_lut", "int8_t", tanh_lut(qscale), 16),
        c_array("g_token_emb", "int8_t", q["emb"], 16),
        c_array("g_token_rec_w_u8", "uint8_t", rec_u8, 16),
        c_array("g_token_rec_corr", "int24_t", rec_corr, 8),
        c_array("g_token_hidden_bias", "int16_t", q["bh"], 12),
        c_array("g_token_out_w_u8", "uint8_t", out_u8, 16),
        c_array("g_token_out_corr", "int24_t", out_corr, 8),
        c_array("g_token_out_bias", "int16_t", q["bo"], 12),
    ]
    path.write_text("\n\n".join(body), encoding="ascii")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/tinystories/valid.raw.txt"))
    parser.add_argument("--max-chars", type=int, default=3_000_000)
    parser.add_argument("--vocab", type=int, default=192)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--seq", type=int, default=48)
    parser.add_argument("--lr", type=float, default=0.004)
    parser.add_argument("--fallback", choices=("chars", "unk"), default="chars")
    parser.add_argument("--calc-forward", action="store_true")
    parser.add_argument("--qscale", type=int, default=64)
    parser.add_argument("--split-blank-stories", action="store_true")
    parser.add_argument("--latent-buckets", type=int, default=0)
    parser.add_argument("--latent-repeat", type=int, default=1)
    parser.add_argument("--seed", type=int, default=51)
    parser.add_argument("--checkpoint", type=Path, default=Path("models/token_rnn64_v192.npz"))
    parser.add_argument("--out", type=Path, default=Path("runtime/src/generated_model.h"))
    parser.add_argument("--name", default="token-rnn64-v192")
    args = parser.parse_args()

    text = normalize_text(args.data.read_text(encoding="utf-8", errors="ignore"), args.max_chars)
    vocab, stoi = build_vocab(text, args.vocab, args.fallback, args.latent_buckets)
    ids = encode_documents(
        text,
        stoi,
        args.fallback,
        args.split_blank_stories,
        args.latent_buckets,
        args.latent_repeat,
    )
    print(f"encoded_tokens={len(ids)} vocab={len(vocab)} hidden={args.hidden}")
    model = train_model(
        ids,
        len(vocab),
        args.hidden,
        args.steps,
        args.batch,
        args.seq,
        args.lr,
        args.seed,
        args.calc_forward,
        args.qscale,
    )
    model["qscale"] = np.array(args.qscale, dtype=np.int16)
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.checkpoint,
        **model,
        vocab=np.array(vocab, dtype=object),
        token_ids=ids[: min(len(ids), 200000)],
    )
    export_header(args.out, model, vocab, args.name)
    q = quantize(model, qscale=args.qscale)
    print(f"checkpoint={args.checkpoint}")
    print(f"header={args.out}")
    print(f"payload_bytes={sum(arr.size for arr in (q['emb'], q['rec'], q['out'])) + 2 * (q['bh'].size + q['bo'].size) + sum(len(piece.encode('ascii')) + 1 for piece in vocab)}")
    print("\nquantized runtime-style sample:\n")
    print(sample_quantized(q, vocab, args.seed + 1))


if __name__ == "__main__":
    main()
