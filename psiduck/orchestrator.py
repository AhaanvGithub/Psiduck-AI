"""The latent-delegation orchestrator.

Flow for a single AIME problem:

    controller  -> produces a short plan
    4 delegates -> each proposes a full solution + candidate integer answer
    aggregator  -> reconciles the candidates into one final integer

All five agents share the exact same backbone instance (one set of weights, one
latent space, one load). If the aggregator fails to emit a parseable answer we
fall back to a majority vote over the delegates.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from .agents import (
    AGGREGATOR_SYSTEM,
    CONTROLLER_SYSTEM,
    DELEGATE_SPECS,
    Agent,
)
from .aime import extract_answer, majority_vote
from .backbone import Backbone


@dataclass
class Candidate:
    agent: str
    text: str
    answer: Optional[int]


@dataclass
class SolveResult:
    problem: str
    plan: str
    candidates: List[Candidate]
    aggregator_text: str
    final_answer: Optional[int]
    shared_backbone_id: int
    backbone_name: str
    votes: dict = field(default_factory=dict)


class LatentDelegationOrchestrator:
    def __init__(self, model: Backbone) -> None:
        self.model = model
        self.controller = Agent("controller", CONTROLLER_SYSTEM, model)
        self.delegates = [
            Agent(name, prompt, model) for name, prompt in DELEGATE_SPECS
        ]
        self.aggregator = Agent("aggregator", AGGREGATOR_SYSTEM, model)

        # Core invariant: every agent must reference the identical backbone.
        shared = [self.controller, *self.delegates, self.aggregator]
        assert all(a.model is model for a in shared), (
            "all agents must share one backbone instance (single latent space)"
        )

    @property
    def agent_count(self) -> int:
        return 1 + len(self.delegates) + 1  # controller + delegates + aggregator

    def _plan(self, problem: str, max_new_tokens: Optional[int]) -> str:
        task = (
            "Problem:\n"
            f"{problem}\n\n"
            "Write the short plan now."
        )
        return self.controller.respond(task, max_new_tokens=max_new_tokens)

    def _delegate(
        self, delegate: Agent, problem: str, plan: str, max_new_tokens: Optional[int]
    ) -> Candidate:
        task = (
            "Problem:\n"
            f"{problem}\n\n"
            "Team plan (for reference):\n"
            f"{plan}\n\n"
            "Now produce your solution from your specialty's perspective."
        )
        text = delegate.respond(task, max_new_tokens=max_new_tokens)
        return Candidate(agent=delegate.name, text=text, answer=extract_answer(text))

    def _aggregate(
        self, problem: str, candidates: List[Candidate], max_new_tokens: Optional[int]
    ) -> str:
        blocks = []
        for cand in candidates:
            blocks.append(
                f"[{cand.agent}] proposed answer: {cand.answer}\n{cand.text}"
            )
        task = (
            "Problem:\n"
            f"{problem}\n\n"
            "Candidate solutions:\n"
            + "\n\n".join(blocks)
            + "\n\nDecide the final integer answer now."
        )
        return self.aggregator.respond(task, max_new_tokens=max_new_tokens)

    def solve(
        self,
        problem: str,
        *,
        plan_max_new_tokens: Optional[int] = None,
        delegate_max_new_tokens: Optional[int] = None,
        aggregator_max_new_tokens: Optional[int] = None,
    ) -> SolveResult:
        plan = self._plan(problem, plan_max_new_tokens)

        candidates: List[Candidate] = [
            self._delegate(d, problem, plan, delegate_max_new_tokens)
            for d in self.delegates
        ]

        aggregator_text = self._aggregate(
            problem, candidates, aggregator_max_new_tokens
        )
        final = extract_answer(aggregator_text)

        vote = majority_vote([c.answer for c in candidates])
        if final is None:
            final = vote

        votes: dict = {}
        for c in candidates:
            if c.answer is not None:
                votes[c.answer] = votes.get(c.answer, 0) + 1

        return SolveResult(
            problem=problem,
            plan=plan,
            candidates=candidates,
            aggregator_text=aggregator_text,
            final_answer=final,
            shared_backbone_id=id(self.model),
            backbone_name=self.model.name,
            votes=votes,
        )
