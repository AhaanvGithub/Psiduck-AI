"""The two solvers compared in the eval.

BaselineSolver   one model, English chain-of-thought, greedy, capped at max_new_tokens.

LatentSolver     four specialist delegates run sequentially on one shared KV cache.
                 Each delegate reads its prompt, then takes `latent_steps` latent
                 steps: the last hidden state is fed back as the next input instead
                 of decoding a token. Nothing is decoded until the final decoder,
                 which reads its prompt on top of the full cache (every delegate's
                 prompt and latent thoughts) and generates the answer greedily under
                 the same max_new_tokens cap as the baseline.

                 Optional early exit: a delegate stops its latent steps once two
                 consecutive hidden states have cosine similarity >= early_exit_cos.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

from psiduck.answers import extract_answer
from psiduck.roles import DELEGATES, Delegate, baseline_prompt, decoder_prompt, delegate_prompt


@dataclass
class SolveResult:
    text: str
    prediction: Optional[str]
    wall_s: float
    compute: dict
    detail: dict = field(default_factory=dict)


def greedy_generate(backbone, session, prompt_ids: list[int], max_new_tokens: int) -> list[int]:
    h = backbone.feed_tokens(session, prompt_ids)
    backbone.meter.prompt_tokens += len(prompt_ids)
    out: list[int] = []
    while len(out) < max_new_tokens:
        tok = backbone.next_token(h)
        if tok in backbone.eos_ids:
            break
        out.append(tok)
        backbone.meter.generated_tokens += 1
        if len(out) < max_new_tokens:
            h = backbone.feed_tokens(session, [tok])
    return out


class BaselineSolver:
    name = "baseline"

    def __init__(self, backbone, max_new_tokens: int):
        self.backbone = backbone
        self.max_new_tokens = max_new_tokens

    def solve(self, question: str) -> SolveResult:
        b = self.backbone
        b.meter.reset()
        t0 = time.perf_counter()
        s = b.new_session()
        ids = greedy_generate(b, s, b.chat_turn_ids(baseline_prompt(question)), self.max_new_tokens)
        text = b.decode(ids)
        return SolveResult(
            text, extract_answer(text), time.perf_counter() - t0, b.meter.snapshot(),
            {"hit_token_cap": len(ids) >= self.max_new_tokens},
        )


class LatentSolver:
    name = "latent"

    def __init__(
        self,
        backbone,
        max_new_tokens: int,
        latent_steps: int = 30,
        early_exit_cos: Optional[float] = None,
        delegates: tuple[Delegate, ...] = DELEGATES,
    ):
        self.backbone = backbone
        self.max_new_tokens = max_new_tokens
        self.latent_steps = latent_steps
        self.early_exit_cos = early_exit_cos
        self.delegates = delegates

    def _think(self, session, h) -> int:
        """Run up to latent_steps latent steps from hidden state h; return steps taken."""
        b = self.backbone
        for step in range(1, self.latent_steps + 1):
            h_next = b.feed_latent(session, h)
            if self.early_exit_cos is not None and b.cosine(h, h_next) >= self.early_exit_cos:
                return step
            h = h_next
        return self.latent_steps

    def solve(self, question: str) -> SolveResult:
        b = self.backbone
        b.meter.reset()
        t0 = time.perf_counter()
        s = b.new_session()
        steps_per_delegate = []
        for i, d in enumerate(self.delegates):
            ids = b.chat_turn_ids(delegate_prompt(d, question, i))
            h = b.feed_tokens(s, ids)
            b.meter.prompt_tokens += len(ids)
            steps_per_delegate.append(self._think(s, h))
            b.feed_tokens(s, b.turn_end_ids)
            b.meter.prompt_tokens += len(b.turn_end_ids)
        ids = greedy_generate(b, s, b.chat_turn_ids(decoder_prompt(question)), self.max_new_tokens)
        text = b.decode(ids)
        return SolveResult(
            text, extract_answer(text), time.perf_counter() - t0, b.meter.snapshot(),
            {
                "steps_per_delegate": steps_per_delegate,
                "context_length": s.length,
                "hit_token_cap": len(ids) >= self.max_new_tokens,
            },
        )


def build_solver(config: str, backbone, settings):
    if config == "baseline":
        return BaselineSolver(backbone, settings.max_new_tokens)
    if config == "latent":
        return LatentSolver(backbone, settings.max_new_tokens, settings.latent_steps)
    if config == "latent-early-exit":
        if settings.early_exit_cos is None:
            raise ValueError("latent-early-exit needs PSIDUCK_EARLY_EXIT_COS")
        return LatentSolver(backbone, settings.max_new_tokens, settings.latent_steps, settings.early_exit_cos)
    raise KeyError(f"unknown config {config!r}")
