"""Agents are thin roles layered over one shared backbone.

Every agent holds a reference to the *same* `Backbone` instance. Constructing an
agent does not load any weights; it only stores a system prompt that specializes
the shared model's behavior for a particular sub-task.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .backbone import Backbone, Message


@dataclass
class Agent:
    name: str
    system_prompt: str
    model: Backbone

    def respond(self, task: str, max_new_tokens: Optional[int] = None) -> str:
        messages: list[Message] = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": task},
        ]
        return self.model.chat(messages, max_new_tokens=max_new_tokens)


CONTROLLER_SYSTEM = (
    "You are the controller of a team solving a competition math (AIME) problem. "
    "AIME answers are integers from 0 to 999. Do NOT solve the problem yourself. "
    "Instead, write a short, concrete plan (3-5 bullet points) describing the key "
    "sub-steps and which mathematical tools are relevant."
)

AGGREGATOR_SYSTEM = (
    "You are the aggregator for a team solving an AIME problem. You are given "
    "several candidate solutions and their proposed integer answers. Decide the "
    "single most likely correct integer answer (0-999). Briefly justify, then end "
    "with a line of the exact form: \\boxed{N} where N is the integer answer."
)


# Four specialist delegates. Each one is a distinct *role* over the shared model.
DELEGATE_SPECS: tuple[tuple[str, str], ...] = (
    (
        "algebraist",
        "You are an expert in algebra and equation manipulation. Solve the AIME "
        "problem using algebraic techniques. Show concise steps and end with "
        "\\boxed{N} (an integer 0-999).",
    ),
    (
        "combinatorialist",
        "You are an expert in combinatorics and counting. Solve the AIME problem "
        "using counting arguments where relevant. Show concise steps and end with "
        "\\boxed{N} (an integer 0-999).",
    ),
    (
        "number_theorist",
        "You are an expert in number theory and geometry. Solve the AIME problem "
        "using modular arithmetic, divisibility, or geometric insight as relevant. "
        "Show concise steps and end with \\boxed{N} (an integer 0-999).",
    ),
    (
        "verifier",
        "You are a careful verifier. Re-derive the answer independently, checking "
        "edge cases and arithmetic. Show concise steps and end with \\boxed{N} (an "
        "integer 0-999).",
    ),
)
