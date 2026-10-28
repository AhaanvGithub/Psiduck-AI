"""Prompts for the four specialist delegates, the final decoder, and the baseline.

The decoder and the baseline end with the identical answer instruction, so the
only difference between the two configs is the latent context the decoder sees.
"""

from __future__ import annotations

from dataclasses import dataclass

ANSWER_INSTRUCTION = (
    "Solve the problem step by step, then give the final answer as a single "
    "number in \\boxed{}."
)


@dataclass(frozen=True)
class Delegate:
    name: str
    focus: str


DELEGATES = (
    Delegate(
        "algebra",
        "Translate the problem into equations or expressions and work out the "
        "quantities they determine.",
    ),
    Delegate(
        "counting",
        "Identify anything that must be counted, enumerated or tracked step by "
        "step, and count it carefully.",
    ),
    Delegate(
        "number-theory",
        "Check divisibility, remainders, integrality and units, and whether the "
        "answer must be a whole number.",
    ),
    Delegate(
        "geometry-verify",
        "Handle any geometric or spatial content, then verify the team's "
        "reasoning so far and look for mistakes.",
    ),
)


def delegate_prompt(d: Delegate, question: str, position: int) -> str:
    team = (
        "You are the first member of a team of specialists."
        if position == 0
        else "You are on a team of specialists. Your teammates' earlier thoughts are in your context."
    )
    return (
        f"{team} You are the {d.name} specialist.\n\n"
        f"Problem: {question}\n\n"
        f"Your job: {d.focus} Think silently; do not write the final answer."
    )


def decoder_prompt(question: str) -> str:
    return (
        "You are the final solver on a team of specialists. Their thoughts about "
        "this problem are in your context; use them where they help.\n\n"
        f"Problem: {question}\n\n{ANSWER_INSTRUCTION}"
    )


def baseline_prompt(question: str) -> str:
    return f"Problem: {question}\n\n{ANSWER_INSTRUCTION}"
