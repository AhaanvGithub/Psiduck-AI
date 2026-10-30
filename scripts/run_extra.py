"""Resumable run of both configs on one extra dataset (Experiment 2 in EVAL_PLAN.md).

    python scripts/run_extra.py amc23 --max-new-tokens 4096

Writes results/<dataset>/{latent,baseline}.jsonl in the same format as the main
runner, so analysis.dataset_section can score it. Re-running skips finished problems.
"""

from __future__ import annotations

import argparse
import sys
import traceback
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from psiduck import analysis, data, system  # noqa: E402
from psiduck.answers import is_correct  # noqa: E402
from psiduck.config import Settings  # noqa: E402
from psiduck.hf_env import REPO_ROOT  # noqa: E402
from psiduck.pipeline import build_solver  # noqa: E402
from psiduck.runner import append_jsonl, problem_seed  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("--max-new-tokens", type=int, required=True)
    args = ap.parse_args()

    settings = replace(Settings.from_env(), max_new_tokens=args.max_new_tokens)
    topic = system.ntfy_topic(REPO_ROOT / "results" / "ntfy_topic.txt")
    problems = data.load(args.dataset)
    from psiduck.backbone import SharedModel

    model = SharedModel(settings.model_name, settings.quantization, settings.seed, thinking=settings.thinking,
                        temperature=settings.temperature, top_p=settings.top_p)
    for cfg in ("latent", "baseline"):
        path = REPO_ROOT / "results" / args.dataset / f"{cfg}.jsonl"
        done = analysis.read_jsonl(path)
        solver = build_solver(cfg, model, settings)
        for idx, p in enumerate(problems):
            if idx in done:
                continue
            base = {"idx": idx, "problem_id": p.problem_id, "gold": p.answer}
            try:
                r = solver.solve(p.question, seed=problem_seed(settings.seed, args.dataset, idx))
                rec = {**base, "prediction": r.prediction, "correct": is_correct(r.prediction, p.answer),
                       "wall_s": round(r.wall_s, 3), **r.compute, **r.detail, "text": r.text}
            except Exception as e:
                rec = {**base, "prediction": None, "correct": False, "error": f"{type(e).__name__}: {e}"[:500]}
                print(traceback.format_exc(), flush=True)
                model.free()
            append_jsonl(path, rec)
            done[idx] = rec
            print(f"{args.dataset}/{cfg} {len(done)}/{len(problems)}", flush=True)
        s = analysis.config_stats(list(done.values()))
        system.notify(topic, f"{args.dataset}/{cfg} finished: {s['correct']}/{s['n']} correct, "
                             f"mean {s['gen_tokens']:.0f} tokens, {s['cap_hits']} hit the cap.")
    system.notify(topic, f"{args.dataset} run DONE. Stop the pod after the results are downloaded.", priority="high")
    return 0


if __name__ == "__main__":
    sys.exit(main())
