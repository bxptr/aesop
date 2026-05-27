#!/usr/bin/env python3
"""Evaluate TI-84 CE-feasible sampling rules for token RNN story models."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np

DEFAULT_CHECKPOINT = Path("aesop/checkpoints/model.npz")

from compiler.token_rnn import BOS, UNK, is_latent_piece, quantize, tanh_lut


@dataclass(frozen=True)
class SamplerConfig:
    name: str
    margin: int = 128
    repeat_penalty: int = 96
    recent_count: int = 8
    latent_repeat: int = 4
    weights: tuple[int, ...] = (16, 8, 4, 2, 1, 1)
    once_penalty_after: int = -1
    once_penalty: int = 0
    punctuation_start_penalty: int = 0
    eos_after: int = 0
    eos_bonus: int = 0
    force_first_once: bool = False
    local_repairs: bool = False
    repair_penalty: int = 4096
    phrase_guards: bool = False


class RuntimeSampler:
    def __init__(self, model_path: Path, config: SamplerConfig):
        data = np.load(model_path, allow_pickle=True)
        model = {key: data[key] for key in ("emb", "rec", "bh", "out", "bo")}
        qscale = int(data["qscale"]) if "qscale" in data.files else 64
        self.q = quantize(model, qscale=qscale)
        self.vocab = [str(piece) for piece in data["vocab"].tolist()]
        self.config = config
        self.lut = tanh_lut()
        self.hidden = self.q["rec"].shape[0]
        self.bos_id = self.vocab.index(BOS) if BOS in self.vocab else 0
        self.eos_id = self.vocab.index("<EOS>") if "<EOS>" in self.vocab else 1
        self.once_id = self.vocab.index("Once") if "Once" in self.vocab else -1
        self.unk_id = self.vocab.index(UNK) if UNK in self.vocab else -1
        self.latent_ids = [
            idx for idx, piece in enumerate(self.vocab) if is_latent_piece(piece)
        ]
        self.banned = {self.bos_id}
        if self.unk_id >= 0:
            self.banned.add(self.unk_id)
        self.banned.update(self.latent_ids)
        self.bad_pairs = self._build_bad_pairs()
        self.guard_next = self._build_guard_next()
        self.bell_id = self.vocab.index(" bell") if " bell" in self.vocab else -1
        self.clearly_id = self.vocab.index(" clearly") if " clearly" in self.vocab else -1
        self.rang_id = self.vocab.index(" rang") if " rang" in self.vocab else -1
        self.the_id = self.vocab.index(" the") if " the" in self.vocab else -1
        self.silver_id = self.vocab.index(" silver") if " silver" in self.vocab else -1
        self.fell_id = self.vocab.index(" fell") if " fell" in self.vocab else -1
        self.under_id = self.vocab.index(" under") if " under" in self.vocab else -1
        self.a_id = self.vocab.index(" a") if " a" in self.vocab else -1
        self.chair_id = self.vocab.index(" chair") if " chair" in self.vocab else -1
        self.held_id = self.vocab.index(" held") if " held" in self.vocab else -1
        self.waited_id = self.vocab.index(" waited") if " waited" in self.vocab else -1
        self.comma_id = self.vocab.index(",") if "," in self.vocab else -1
        self.turned_id = self.vocab.index(" turned") if " turned" in self.vocab else -1

    def _build_bad_pairs(self) -> set[tuple[int, int]]:
        if not self.config.local_repairs:
            return set()
        pairs = (
            (" rang", " under"),
            (" rang", " one"),
            (" rang", " found"),
            (" rang", " straight"),
            (" rang", " way"),
            (" held", " that"),
            (" turned", " key"),
            (" found", "."),
        )
        out: set[tuple[int, int]] = set()
        for left, right in pairs:
            if left in self.vocab and right in self.vocab:
                out.add((self.vocab.index(left), self.vocab.index(right)))
        return out

    def _build_guard_next(self) -> dict[int, set[int]]:
        if not self.config.phrase_guards:
            return {}
        pairs = {
            " asked": (" for",),
            " carried": (" water",),
            " followed": (" tiny",),
            " gave": (" everyone",),
            " held": (" the",),
            " looked": (" under",),
            " made": (" a",),
            " painted": (" a",),
            " pushed": (" with",),
            " shared": (" the",),
            " sang": (" a",),
            " told": (" a",),
            " tried": (" again",),
            " turned": (" the",),
            " wrote": (" one",),
        }
        out: dict[int, set[int]] = {}
        for left, rights in pairs.items():
            if left not in self.vocab:
                continue
            allowed = {self.vocab.index(right) for right in rights if right in self.vocab}
            if allowed:
                out[self.vocab.index(left)] = allowed
        return out

    def sample(self, seed: int, chars: int = 520) -> str:
        rng = np.random.default_rng(seed)
        h = np.zeros(self.hidden, dtype=np.int16)
        recent = [self.bos_id] * self.config.recent_count
        recent_pos = 0
        emitted = 0
        prev2_token = -1

        def push_recent(token: int) -> None:
            nonlocal recent_pos
            if not recent:
                return
            recent[recent_pos] = token
            recent_pos = (recent_pos + 1) % len(recent)

        def score_adjust(prev_prev_token: int, prev_token: int, token: int, score: int) -> int:
            piece = self.vocab[token]
            for old in recent:
                if old == token:
                    score -= self.config.repeat_penalty
            if (
                self.config.once_penalty_after >= 0
                and emitted >= self.config.once_penalty_after
                and piece == "Once"
            ):
                score -= self.config.once_penalty
            if emitted == 0 and piece[:1] in ",.!?;:":
                score -= self.config.punctuation_start_penalty
            if self.config.eos_after and emitted >= self.config.eos_after and token == self.eos_id:
                score += self.config.eos_bonus
            if emitted > 0 and (prev_token, token) in self.bad_pairs:
                score -= self.config.repair_penalty
            if emitted > 0 and self.config.local_repairs and prev_token == self.rang_id:
                if prev_prev_token == self.bell_id:
                    if token != self.clearly_id:
                        score -= self.config.repair_penalty
                elif token != self.the_id:
                    score -= self.config.repair_penalty
            if emitted > 0 and self.config.local_repairs and prev_token == self.held_id:
                if token != self.the_id:
                    score -= self.config.repair_penalty
            if emitted > 0 and self.config.local_repairs and prev_token == self.waited_id:
                if token != self.comma_id:
                    score -= self.config.repair_penalty
            if emitted > 0 and self.config.local_repairs and prev_token == self.turned_id:
                if token != self.the_id:
                    score -= self.config.repair_penalty
            if (
                emitted > 0
                and self.config.local_repairs
                and prev_prev_token == self.rang_id
                and prev_token == self.the_id
                and token != self.silver_id
            ):
                score -= self.config.repair_penalty
            if (
                emitted > 0
                and self.config.local_repairs
                and prev_prev_token == self.fell_id
                and prev_token == self.under_id
                and token != self.a_id
            ):
                score -= self.config.repair_penalty
            if (
                emitted > 0
                and self.config.local_repairs
                and prev_prev_token == self.under_id
                and prev_token == self.a_id
                and token != self.chair_id
            ):
                score -= self.config.repair_penalty
            if emitted > 0 and self.config.phrase_guards:
                allowed = self.guard_next.get(prev_token)
                if allowed is not None and token not in allowed:
                    score -= self.config.repair_penalty
                if prev_token == self.rang_id:
                    if prev_prev_token == self.bell_id:
                        if token != self.clearly_id:
                            score -= self.config.repair_penalty
                    elif token != self.the_id:
                        score -= self.config.repair_penalty
            return score

        def step(input_id: int) -> int:
            nonlocal h, prev2_token
            acc = self.q["emb"][input_id].astype(np.int16) + self.q["bh"].astype(np.int16)
            acc += (self.q["rec"].astype(np.int16) @ h.astype(np.int16)) >> 7
            h = self.lut[np.clip(acc + 256, 0, 511).astype(np.int16)].astype(np.int16)
            scores = self.q["bo"].astype(np.int32) + (
                (self.q["out"].astype(np.int32) @ h.astype(np.int32)) >> 7
            )
            for token in self.banned:
                scores[token] = -100000000
            adjusted = scores.copy()
            for token in range(len(adjusted)):
                if token not in self.banned:
                    adjusted[token] = score_adjust(prev2_token, input_id, token, int(adjusted[token]))

            top_k = len(self.config.weights)
            top = np.argsort(adjusted)[-top_k:][::-1]
            active = 1
            for rank in range(1, top_k):
                if adjusted[top[0]] - adjusted[top[rank]] > self.config.margin:
                    break
                active += 1

            total = sum(self.config.weights[:active])
            roll = int(rng.integers(0, total))
            for rank in range(active):
                if roll < self.config.weights[rank]:
                    token = int(top[rank])
                    push_recent(token)
                    prev2_token = input_id
                    return token
                roll -= self.config.weights[rank]
            token = int(top[0])
            push_recent(token)
            prev2_token = input_id
            return token

        token = step(self.bos_id)
        if self.latent_ids:
            latent = self.latent_ids[int(rng.integers(0, len(self.latent_ids)))]
            for _ in range(self.config.latent_repeat):
                token = step(latent)
        if self.config.force_first_once and self.once_id >= 0:
            token = self.once_id

        out: list[str] = []
        while len("".join(out)) < chars and token != self.eos_id:
            piece = self.vocab[token] if token not in self.banned else ""
            out.append(piece)
            emitted += len(piece)
            token = step(token)

        return "".join(out).replace("\n", " ").strip()


BAD_SUBSTRINGS = (
    ",Once",
    ".Once",
    " found.",
    "the the",
    "a a ",
    "in in ",
    "near near",
    "to to ",
    "what to your",
    "Soon, Soon",
    "Once upon. time",
    "Once upon heart time",
)


def score_story(text: str) -> int:
    score = 0
    if text.startswith("Once upon a time,"):
        score += 30
    else:
        score -= 30
    once_count = text.count("Once upon")
    score += 10 if once_count == 1 else -20 * abs(once_count - 1)
    if "It looked ordinary until" in text or "It wanted to" in text:
        score += 8
    if "Soon," in text:
        score += 8
    if "Everyone felt" in text or "They laughed" in text:
        score += 8
    if "learned that" in text or "remembered that" in text:
        score += 10
    if text.endswith((".", "!", "?")):
        score += 5
    if len(text) >= 260:
        score += 6
    if len(text) >= 420:
        score += 2
    for bad in BAD_SUBSTRINGS:
        if bad in text:
            score -= 25
    score -= 10 * max(0, text.count("Soon,") - 1)
    score -= 8 * max(0, text.count("Everyone felt") - 1)
    return score


def default_configs() -> list[SamplerConfig]:
    configs: list[SamplerConfig] = []
    for margin in (64, 80, 96, 112, 128, 144):
        for repeat in (64, 96, 128):
            configs.append(
                SamplerConfig(
                    name=f"m{margin}_r{repeat}_base",
                    margin=margin,
                    repeat_penalty=repeat,
                )
            )
            configs.append(
                SamplerConfig(
                    name=f"m{margin}_r{repeat}_norestart",
                    margin=margin,
                    repeat_penalty=repeat,
                    once_penalty_after=24,
                    once_penalty=4096,
                    punctuation_start_penalty=4096,
                    force_first_once=True,
                )
            )
            configs.append(
                SamplerConfig(
                    name=f"m{margin}_r{repeat}_repair",
                    margin=margin,
                    repeat_penalty=repeat,
                    once_penalty_after=24,
                    once_penalty=4096,
                    punctuation_start_penalty=4096,
                    force_first_once=True,
                    local_repairs=True,
                )
            )
            configs.append(
                SamplerConfig(
                    name=f"m{margin}_r{repeat}_guard",
                    margin=margin,
                    repeat_penalty=repeat,
                    once_penalty_after=24,
                    once_penalty=4096,
                    punctuation_start_penalty=4096,
                    force_first_once=True,
                    phrase_guards=True,
                )
            )
    for margin in (96, 112, 128):
        configs.append(
            SamplerConfig(
                name=f"m{margin}_sharp_norestart",
                margin=margin,
                repeat_penalty=96,
                weights=(24, 6, 2, 1, 1, 1),
                once_penalty_after=24,
                once_penalty=4096,
                punctuation_start_penalty=4096,
                force_first_once=True,
            )
        )
        configs.append(
            SamplerConfig(
                name=f"m{margin}_soft_repair",
                margin=margin,
                repeat_penalty=96,
                weights=(12, 8, 5, 3, 2, 1),
                once_penalty_after=24,
                once_penalty=4096,
                punctuation_start_penalty=4096,
                force_first_once=True,
                local_repairs=True,
            )
        )
        configs.append(
            SamplerConfig(
                name=f"m{margin}_soft_guard",
                margin=margin,
                repeat_penalty=96,
                weights=(12, 8, 5, 3, 2, 1),
                once_penalty_after=24,
                once_penalty=4096,
                punctuation_start_penalty=4096,
                force_first_once=True,
                phrase_guards=True,
            )
        )
        configs.append(
            SamplerConfig(
                name=f"m{margin}_soft_norestart",
                margin=margin,
                repeat_penalty=96,
                weights=(12, 8, 5, 3, 2, 1),
                once_penalty_after=24,
                once_penalty=4096,
                punctuation_start_penalty=4096,
                force_first_once=True,
            )
        )
    return configs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_CHECKPOINT,
    )
    parser.add_argument("--seeds", type=int, default=64)
    parser.add_argument("--chars", type=int, default=520)
    parser.add_argument("--show", type=int, default=5)
    args = parser.parse_args()

    rows: list[tuple[float, int, int, SamplerConfig, list[tuple[int, str, int]]]] = []
    for config in default_configs():
        sampler = RuntimeSampler(args.model, config)
        stories: list[tuple[int, str, int]] = []
        scores: list[int] = []
        bad = 0
        for seed in range(args.seeds):
            text = sampler.sample(seed + 1000, args.chars)
            score = score_story(text)
            stories.append((score, text, seed + 1000))
            scores.append(score)
            if score < 30:
                bad += 1
        rows.append((float(np.mean(scores)), bad, min(scores), config, stories))

    rows.sort(key=lambda row: (row[0], -row[1], row[2]), reverse=True)
    for mean_score, bad, min_score, config, stories in rows[:args.show]:
        print(
            f"\n### {config.name} mean={mean_score:.1f} bad={bad}/{args.seeds} "
            f"min={min_score} margin={config.margin} repeat={config.repeat_penalty} "
            f"weights={config.weights}"
        )
        stories.sort(key=lambda item: (item[0], item[2]), reverse=True)
        for score, text, seed in stories[:2]:
            print(f"\n-- seed={seed} score={score}\n{text[:args.chars]}")
        print("\n-- median-ish")
        mid = sorted(stories, key=lambda item: item[0])[len(stories) // 2]
        print(f"seed={mid[2]} score={mid[0]}\n{mid[1][:args.chars]}")


if __name__ == "__main__":
    main()
