"""Solvers and the head-to-head comparison.

``LatentDelegationSolver`` — the Psiduck-AI approach. One shared backbone:
    1. encode the problem                -> context latent   (no generation)
    2. 4 delegates exchange latent msgs  -> 4 latent messages (no generation)
    3. controller fuses + decodes ONLY the final integer answer (short generation)

``SingleModelSolver`` — the baseline. One model reasons in English (long
generation) and we parse the boxed integer.

``compare()`` runs both on a problem and returns answers + compute stats so we
can test the thesis that latent communication uses fewer generated tokens.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List, Optional

from .aime import extract_answer
from .backbone import Backbone, LatentState
from .metrics import ComputeStats
from .roles import DELEGATE_ROLES


@dataclass
class SolveResult:
    approach: str
    answer: Optional[int]
    raw_text: str
    compute: ComputeStats
    wall_time_s: float
    detail: dict


class LatentDelegationSolver:
    def __init__(self, backbone: Backbone, answer_max_new_tokens: int = 24) -> None:
        self.backbone = backbone
        self.answer_max_new_tokens = answer_max_new_tokens

    @property
    def num_agents(self) -> int:
        return 1 + len(DELEGATE_ROLES) + 1  # encoder/controller + delegates + decoder

    def solve(self, problem: str) -> SolveResult:
        before = self.backbone.meter.snapshot()
        t0 = time.time()

        context = self.backbone.encode(problem)

        messages: List[LatentState] = []
        for role in DELEGATE_ROLES:
            messages.append(self.backbone.delegate_message(context, role))

        fused = self.backbone.combine([context, *messages])
        raw = self.backbone.decode_answer(fused, self.answer_max_new_tokens)

        elapsed = time.time() - t0
        compute = self.backbone.meter.snapshot().minus(before)
        return SolveResult(
            approach="latent-4-agent",
            answer=extract_answer(raw),
            raw_text=raw,
            compute=compute,
            wall_time_s=elapsed,
            detail={
                "shared_backbone_id": id(self.backbone),
                "num_agents": self.num_agents,
                "delegates": len(DELEGATE_ROLES),
                "latent_messages": len(messages),
            },
        )


class SingleModelSolver:
    def __init__(self, backbone: Backbone, max_new_tokens: int = 256) -> None:
        self.backbone = backbone
        self.max_new_tokens = max_new_tokens

    def solve(self, problem: str) -> SolveResult:
        before = self.backbone.meter.snapshot()
        t0 = time.time()
        raw = self.backbone.baseline_generate(problem, self.max_new_tokens)
        elapsed = time.time() - t0
        compute = self.backbone.meter.snapshot().minus(before)
        return SolveResult(
            approach="single-llm-english",
            answer=extract_answer(raw),
            raw_text=raw,
            compute=compute,
            wall_time_s=elapsed,
            detail={},
        )


@dataclass
class Comparison:
    problem: str
    expected: Optional[int]
    baseline: SolveResult
    latent: SolveResult

    @property
    def generated_token_ratio(self) -> Optional[float]:
        b = self.baseline.compute.generated_tokens
        if b == 0:
            return None
        return self.latent.compute.generated_tokens / b


def compare(
    backbone: Backbone,
    problem: str,
    *,
    expected: Optional[int] = None,
    baseline_max_new_tokens: int = 256,
    answer_max_new_tokens: int = 24,
) -> Comparison:
    baseline = SingleModelSolver(backbone, baseline_max_new_tokens).solve(problem)
    latent = LatentDelegationSolver(backbone, answer_max_new_tokens).solve(problem)
    return Comparison(
        problem=problem, expected=expected, baseline=baseline, latent=latent
    )
