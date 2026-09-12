"""CLI entry point for the Psiduck-AI latent-delegation solver.

Examples:
    python main.py --list
    python main.py --problem-index 0
    python main.py --problem "Find the remainder when 7^100 is divided by 1000."
"""

from __future__ import annotations

import argparse
import sys
import time
from textwrap import shorten

from psiduck.config import get_settings
from psiduck.data import SAMPLE_PROBLEMS, get_problem
from psiduck.orchestrator import LatentDelegationOrchestrator


def _print_header(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def _build_orchestrator():
    settings = get_settings()
    print(f"Loading shared backbone: {settings.model_name} (CPU, fp32) ...")
    from psiduck.backbone import SharedModel

    t0 = time.time()
    model = SharedModel(
        settings.model_name,
        max_new_tokens=settings.max_new_tokens,
        temperature=settings.temperature,
        seed=settings.seed,
        torch_threads=settings.torch_threads,
    )
    print(f"Model loaded in {time.time() - t0:.1f}s")
    return LatentDelegationOrchestrator(model), settings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Psiduck-AI AIME solver")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--problem", type=str, help="A custom AIME problem statement.")
    group.add_argument(
        "--problem-index",
        type=int,
        default=0,
        help="Index into the built-in sample problems (default: 0).",
    )
    parser.add_argument(
        "--list", action="store_true", help="List built-in sample problems and exit."
    )
    args = parser.parse_args(argv)

    if args.list:
        for i, p in enumerate(SAMPLE_PROBLEMS):
            print(f"[{i}] {p.problem_id}: {shorten(p.statement, width=80)}")
        return 0

    expected = None
    if args.problem:
        problem_text = args.problem
    else:
        problem = get_problem(args.problem_index)
        problem_text = problem.statement
        expected = problem.answer

    orchestrator, settings = _build_orchestrator()

    _print_header("PROBLEM")
    print(problem_text)

    t0 = time.time()
    result = orchestrator.solve(
        problem_text,
        plan_max_new_tokens=settings.controller_max_new_tokens,
        delegate_max_new_tokens=settings.max_new_tokens,
        aggregator_max_new_tokens=settings.aggregator_max_new_tokens,
    )
    elapsed = time.time() - t0

    _print_header("CONTROLLER PLAN")
    print(result.plan)

    for cand in result.candidates:
        _print_header(f"DELEGATE: {cand.agent}  (answer={cand.answer})")
        print(cand.text)

    _print_header("AGGREGATOR")
    print(result.aggregator_text)

    _print_header("RESULT")
    print(f"Backbone            : {result.backbone_name}")
    print(f"Shared backbone id  : {result.shared_backbone_id}")
    print(f"Agents (shared model): {orchestrator.agent_count} "
          f"(1 controller + {len(orchestrator.delegates)} delegates + 1 aggregator)")
    print(f"Delegate votes      : {result.votes}")
    print(f"FINAL ANSWER        : {result.final_answer}")
    if expected is not None:
        verdict = "CORRECT" if result.final_answer == expected else "incorrect"
        print(f"Expected answer     : {expected}  -> {verdict}")
    print(f"Wall time           : {elapsed:.1f}s")

    return 0


if __name__ == "__main__":
    sys.exit(main())
