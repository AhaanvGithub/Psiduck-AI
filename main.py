"""Try one problem with the baseline and the latent pipeline.

    python main.py --problem "Find the remainder when 7^100 is divided by 1000."
    python main.py --sample 0
    python main.py --gsm8k 5          # the 6th problem of GSM8K test (unshuffled)

The full benchmark lives in run_overnight.ps1 / scripts/run_eval.py.
"""

from __future__ import annotations

import argparse
import sys

from psiduck import data
from psiduck.answers import is_correct
from psiduck.config import Settings
from psiduck.pipeline import build_solver


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Psiduck-AI: baseline vs latent on one problem")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--problem", type=str)
    g.add_argument("--sample", type=int, default=0)
    g.add_argument("--gsm8k", type=int)
    ap.add_argument("--configs", default="baseline,latent")
    args = ap.parse_args(argv)

    if args.problem:
        question, gold = args.problem, None
    elif args.gsm8k is not None:
        p = data.load("gsm8k")[args.gsm8k]
        question, gold = p.question, p.answer
    else:
        p = data.SAMPLE_PROBLEMS[args.sample]
        question, gold = p.question, p.answer

    settings = Settings.from_env()
    from psiduck.backbone import SharedModel

    print(f"Loading {settings.model_name} ({settings.quantization}) ...")
    model = SharedModel(settings.model_name, settings.quantization, settings.seed)
    print(f"\nProblem: {question}\nGold: {gold}\n")
    for cfg in args.configs.split(","):
        r = build_solver(cfg, model, settings).solve(question)
        verdict = "" if gold is None else ("  CORRECT" if is_correct(r.prediction, gold) else "  wrong")
        print(f"=== {cfg} ===\n{r.text}\n")
        print(f"prediction={r.prediction}{verdict}  wall={r.wall_s:.1f}s  {r.compute}  {r.detail}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
