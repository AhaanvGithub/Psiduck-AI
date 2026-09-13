"""CLI: compare the latent 4-agent pipeline against baselines.

Examples:
    python main.py --list
    python main.py --problem-index 2
    python main.py --problem "Find the remainder when 7^100 is divided by 1000."
    python main.py --all
    python main.py --all --dataset aime24 --self-consistency 4
    python main.py --all --dataset aime24 --dashboard   # live terminal table, for demos
    python main.py --all --ablation-scramble            # does the decoder use the latent at all?
    python main.py --all --decode-reasoning             # give the decoder room to work, not just blurt a digit
    python main.py --all --latent-variant shallow       # skip delegate re-pooling (1 hop, not 5)
    python main.py --all --latent-variant minimal       # no delegates at all -- just encode -> decode

Baselines:
    single-llm-english   greedy, temperature 0 (the original headline baseline)
    self-consistency-Nx  N sampled solo attempts + majority vote (compute-matched;
                          only meaningful with PSIDUCK_TEMPERATURE > 0, since at
                          temperature 0 every sample is identical)

Latent variants (--latent-variant; see psiduck/pipeline.py for the full rationale):
    full     original design: 4 delegates, each re-pools to a single vector (default)
    shallow  1 compression hop; delegate role tags are literal embeddings, never re-pooled
    minimal  0 delegates; just encode(problem) -> decode_answer()
"""

from __future__ import annotations

import argparse
import sys
import time
from textwrap import shorten
from typing import Optional

from psiduck.config import get_settings
from psiduck.data import SAMPLE_PROBLEMS, get_problem, load_problem_set
from psiduck.pipeline import Comparison, compare


def _hr(title: str) -> None:
    print("\n" + "=" * 74)
    print(title)
    print("=" * 74)


def _fmt_flops(n: float) -> str:
    return f"{n:.2e}"


def _build_backbone():
    settings = get_settings()
    print(f"Loading shared backbone: {settings.model_name} (device={settings.device}, dtype={settings.dtype}) ...")
    from psiduck.backbone import SharedModel

    t0 = time.time()
    model = SharedModel(
        settings.model_name,
        temperature=settings.temperature,
        seed=settings.seed,
        torch_threads=settings.torch_threads,
        latent_layer=settings.latent_layer,
        latent_pooling=settings.latent_pooling,
        dtype=settings.dtype,
        device=settings.device,
    )
    print(
        f"Model loaded in {time.time() - t0:.1f}s on {model.device} "
        f"(hidden size {model.hidden_size}, dtype={settings.dtype!r}, "
        f"latent_layer={settings.latent_layer!r}, latent_pooling={settings.latent_pooling!r}, "
        f"target_embed_norm={model.target_embed_norm:.2f})"
    )
    print(
        f"Latent config: variant={settings.latent_variant!r}, "
        f"decode_reasoning={settings.decode_reasoning}, "
        f"answer_max_new_tokens={settings.answer_max_new_tokens}"
    )
    return model, settings


def _print_comparison(cmp: Comparison, num_params: int) -> None:
    b, l = cmp.baseline, cmp.latent
    sc = cmp.self_consistency

    _hr("BASELINE (single LLM, greedy, reasons in English)")
    print(shorten(b.raw_text.replace("\n", " "), width=600))

    if sc is not None:
        _hr(f"SELF-CONSISTENCY ({sc.detail['n_samples']}x sampled + majority vote)")
        print(f"Votes: {sc.detail['votes']}")

    _hr("LATENT 4-AGENT (agents exchange hidden-state vectors, no English between them)")
    print(f"Decoded final answer text: {l.raw_text!r}")
    print(f"Shared backbone id: {l.detail['shared_backbone_id']}  "
          f"agents: {l.detail['num_agents']}  "
          f"latent messages exchanged: {l.detail['latent_messages']}")

    _hr("RESULT")

    def verdict(ans: Optional[int]) -> str:
        if cmp.expected is None:
            return "n/a"
        return "CORRECT" if ans == cmp.expected else "wrong"

    def row(res) -> tuple:
        return (
            res.approach,
            str(res.answer),
            verdict(res.answer),
            str(res.compute.generated_tokens),
            str(res.compute.forward_passes),
            _fmt_flops(res.compute.estimate_flops(num_params)),
            f"{res.wall_time_s:.1f}",
        )

    rows = [
        ("approach", "answer", "correct", "gen_tokens", "fwd_passes", "~flops", "wall_s"),
        row(b),
    ]
    if sc is not None:
        rows.append(row(sc))
    rows.append(row(l))
    if cmp.scrambled is not None:
        rows.append(row(cmp.scrambled))

    widths = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
    for r_i, r in enumerate(rows):
        print("  ".join(cell.ljust(widths[i]) for i, cell in enumerate(r)))
        if r_i == 0:
            print("  ".join("-" * w for w in widths))

    if cmp.expected is not None:
        print(f"\nExpected answer: {cmp.expected}")

    ratio = cmp.generated_token_ratio
    if ratio is not None:
        print(
            f"Latent pipeline generated {l.compute.generated_tokens} tokens vs "
            f"{b.compute.generated_tokens} for the greedy baseline "
            f"({ratio:.1%} as many; lower = cheaper English articulation)."
        )
    flops_b = b.compute.estimate_flops(num_params)
    flops_l = l.compute.estimate_flops(num_params)
    if flops_b:
        print(
            f"Estimated FLOPs (2*params*tokens, counts the delegate forward passes too) -- "
            f"baseline: {_fmt_flops(flops_b)}, latent: {_fmt_flops(flops_l)} "
            f"({flops_l / flops_b:.1%} as many) -- this is the harder-to-game comparison."
        )
    sc_ratio = cmp.latent_vs_self_consistency_ratio
    if sc_ratio is not None:
        print(
            f"Latent pipeline generated {l.compute.generated_tokens} tokens vs "
            f"{sc.compute.generated_tokens} for self-consistency "
            f"({sc_ratio:.1%} as many) -- this is the compute-matched comparison."
        )
    if cmp.scrambled is not None:
        match = cmp.scramble_matches_latent
        print(
            f"Scramble ablation: real latent answer {l.answer!r} vs scrambled "
            f"{cmp.scrambled.answer!r} -- {'SAME' if match else 'different'} "
            "(one problem isn't conclusive; check the aggregate agreement rate)."
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Psiduck-AI latent vs baselines")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--problem", type=str, help="A custom problem statement.")
    group.add_argument("--problem-index", type=int, default=0)
    group.add_argument("--all", action="store_true", help="Run all problems in --dataset.")
    parser.add_argument("--list", action="store_true", help="List problems in --dataset.")
    parser.add_argument(
        "--dataset",
        choices=["sample", "aime24", "aime25", "aime24+25"],
        default="sample",
        help="'sample' is 4 built-in problems (no network). The aime* options "
        "pull the real ~30-problem AIME sets from Hugging Face (needs the "
        "'datasets' package + network) -- see psiduck/data.py.",
    )
    parser.add_argument(
        "--self-consistency",
        type=int,
        default=None,
        metavar="N",
        help="Also run an N-sample self-consistency baseline (majority vote). "
        "Overrides PSIDUCK_SELF_CONSISTENCY_SAMPLES. Needs PSIDUCK_TEMPERATURE "
        "> 0 to be meaningful -- at temperature 0 every sample is identical.",
    )
    parser.add_argument(
        "--dashboard",
        action="store_true",
        help="Render a live-updating rich terminal table + summary instead of "
        "the verbose per-problem log (good for demos). Requires 'rich'.",
    )
    parser.add_argument(
        "--ablation-scramble",
        action="store_true",
        help="Also run a message-scramble control: the same fused latent "
        "context but with its soft tokens randomly permuted before decoding. "
        "If its answers agree with the real latent answers most of the time, "
        "that's evidence the decoder isn't actually using the latent content.",
    )
    parser.add_argument(
        "--latent-variant",
        choices=["full", "shallow", "minimal", "hybrid"],
        default=None,
        help="'full' (default): 4 delegates, each re-pools to a single vector "
        "(5 lossy compression hops total). 'shallow': 1 compression hop only "
        "-- delegate role tags are literal embeddings, never re-pooled. "
        "'minimal': 0 delegates -- just encode(problem) -> decode_answer(). "
        "'hybrid': decoder sees the REAL problem text (uncompressed, same as "
        "baseline) PLUS the 4 delegates' latent contributions appended -- "
        "tests whether latent specialist input helps/hurts on an intact "
        "problem, instead of asking compression to carry the whole problem. "
        "Overrides PSIDUCK_LATENT_VARIANT.",
    )
    parser.add_argument(
        "--decode-reasoning",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Let decode_answer work step-by-step before \\boxed{N} instead "
        "of being forced straight to a bare digit with zero computation "
        "room. This trades away the token/FLOPs efficiency story (reasoning "
        "output runs close to the baseline's token budget) for a shot at "
        "accuracy -- use --no-decode-reasoning to force it off regardless of "
        "PSIDUCK_DECODE_REASONING left set in your shell from an earlier run.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="Only run the first N problems from --dataset (applies to --all "
        "and --list). Useful to keep a demo run short on a larger set like "
        "aime24 without waiting through all ~30 problems.",
    )
    parser.add_argument(
        "--answer-max-new-tokens",
        type=int,
        default=None,
        metavar="N",
        help="Max tokens for the latent pipeline's final decode. Overrides "
        "PSIDUCK_ANSWER_MAX_NEW_TOKENS (default 256).",
    )
    args = parser.parse_args(argv)

    problems = SAMPLE_PROBLEMS if args.dataset == "sample" else load_problem_set(args.dataset)
    if args.limit is not None:
        # No difficulty ranking exists in the fetched data (AimeProblem.difficulty
        # is just "easy" | "aime", not a numeric score) -- this is the first N
        # problems from the set as HF returns them, not "the hardest N". Say so
        # if you present it: "10 real AIME problems", not "the 10 hardest".
        problems = problems[: args.limit]

    if args.list:
        for i, p in enumerate(problems):
            print(f"[{i}] ({p.difficulty}) {p.problem_id}: "
                  f"{shorten(p.statement, width=70)}")
        return 0

    # CLI flags override env-derived settings when explicitly passed.
    latent_variant = args.latent_variant if args.latent_variant is not None else None
    decode_reasoning = args.decode_reasoning if args.decode_reasoning is not None else None
    answer_max_new_tokens_override = args.answer_max_new_tokens

    model, settings = _build_backbone()
    if latent_variant is not None or decode_reasoning is not None or answer_max_new_tokens_override is not None:
        from dataclasses import replace

        settings = replace(
            settings,
            latent_variant=latent_variant if latent_variant is not None else settings.latent_variant,
            decode_reasoning=decode_reasoning if decode_reasoning is not None else settings.decode_reasoning,
            answer_max_new_tokens=(
                answer_max_new_tokens_override
                if answer_max_new_tokens_override is not None
                else settings.answer_max_new_tokens
            ),
        )
        print(
            f"CLI overrides applied: variant={settings.latent_variant!r}, "
            f"decode_reasoning={settings.decode_reasoning}, "
            f"answer_max_new_tokens={settings.answer_max_new_tokens}"
        )

    n_self_consistency = (
        args.self_consistency
        if args.self_consistency is not None
        else settings.self_consistency_samples
    )
    if n_self_consistency > 0 and settings.temperature <= 0.0:
        print(
            "\nWARNING: --self-consistency requires sampling, but "
            "PSIDUCK_TEMPERATURE is 0. Every sample will be identical and the "
            "vote will be a no-op. Set PSIDUCK_TEMPERATURE (e.g. 0.7) to get a "
            "real self-consistency baseline.\n"
        )

    def run_one(problem_text: str, expected: Optional[int]) -> Comparison:
        _hr("PROBLEM")
        print(problem_text)
        if expected is not None:
            print(f"(known answer: {expected})")
        cmp = compare(
            model,
            problem_text,
            expected=expected,
            baseline_max_new_tokens=settings.baseline_max_new_tokens,
            answer_max_new_tokens=settings.answer_max_new_tokens,
            n_self_consistency=n_self_consistency,
            run_scramble_ablation=args.ablation_scramble,
            latent_variant=settings.latent_variant,
            decode_reasoning=settings.decode_reasoning,
        )
        _print_comparison(cmp, model.num_parameters)
        return cmp

    if args.all:
        selected: list[tuple[str, Optional[int]]] = [(p.statement, p.answer) for p in problems]
    elif args.problem:
        selected = [(args.problem, None)]
    else:
        p = problems[args.problem_index] if args.dataset != "sample" else get_problem(args.problem_index)
        selected = [(p.statement, p.answer)]

    if args.dashboard:
        from psiduck.dashboard import run_dashboard

        comparisons = run_dashboard(
            model,
            selected,
            baseline_max_new_tokens=settings.baseline_max_new_tokens,
            answer_max_new_tokens=settings.answer_max_new_tokens,
            n_self_consistency=n_self_consistency,
            num_params=model.num_parameters,
            latent_variant=settings.latent_variant,
            decode_reasoning=settings.decode_reasoning,
            run_scramble_ablation=args.ablation_scramble,
        )
    else:
        comparisons = [run_one(statement, expected) for statement, expected in selected]

    if len(comparisons) > 1:
        _hr("AGGREGATE COMPUTE")
        n = sum(1 for c in comparisons if c.expected is not None)
        num_params = model.num_parameters

        def agg(pick) -> tuple:
            tokens = sum(pick(c).compute.generated_tokens for c in comparisons)
            flops = sum(pick(c).compute.estimate_flops(num_params) for c in comparisons)
            correct = sum(
                1 for c in comparisons if c.expected is not None and pick(c).answer == c.expected
            )
            return tokens, flops, correct

        tot_b, flops_b, corr_b = agg(lambda c: c.baseline)
        tot_l, flops_l, corr_l = agg(lambda c: c.latent)

        print(f"Problems: {len(comparisons)} (scored: {n})")
        print(f"Baseline (greedy) correct: {corr_b}/{n} | tokens: {tot_b} | ~flops: {_fmt_flops(flops_b)}")
        if n_self_consistency > 0:
            tot_sc, flops_sc, corr_sc = agg(lambda c: c.self_consistency)
            print(
                f"Self-consistency ({n_self_consistency}x) correct: {corr_sc}/{n} | "
                f"tokens: {tot_sc} | ~flops: {_fmt_flops(flops_sc)}"
            )
        print(f"Latent (4-agent) correct: {corr_l}/{n} | tokens: {tot_l} | ~flops: {_fmt_flops(flops_l)}")
        print(
            f"Latent vs greedy baseline: {tot_l / tot_b:.1%} as many generated tokens, "
            f"{flops_l / flops_b:.1%} as many estimated FLOPs"
            if tot_b and flops_b
            else "Latent vs greedy baseline: n/a (baseline generated 0 tokens)"
        )
        if n_self_consistency > 0 and tot_sc and flops_sc:
            print(
                f"Latent vs self-consistency: {tot_l / tot_sc:.1%} as many generated tokens, "
                f"{flops_l / flops_sc:.1%} as many estimated FLOPs"
            )
        if args.ablation_scramble:
            checked = [c for c in comparisons if c.scramble_matches_latent is not None]
            if checked:
                agree = sum(1 for c in checked if c.scramble_matches_latent)
                rate = agree / len(checked)
                print(
                    f"\nScramble ablation agreement rate: {agree}/{len(checked)} "
                    f"({rate:.0%}) of problems had scrambled-latent answer == "
                    "real-latent answer."
                )
                if rate >= 0.5:
                    print(
                        "  -> HIGH agreement: the decoder's answer barely depends on the "
                        "latent content at all -- evidence the channel isn't carrying "
                        "usable signal, independent of layer/pooling/model-size choices."
                    )
                else:
                    print(
                        "  -> LOW agreement: the decoder's answer does depend on the "
                        "actual latent content -- some real signal is flowing, even if "
                        "it isn't yet accurate."
                    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
