# Psiduck-AI v2 evaluation plan (pre-registered)

Written on 2026-10-27, before any smoke test or full run. Nothing below may be changed after results are seen. The only field filled in later is N, which is computed mechanically from smoke-test timings by the rule in §5, before any accuracy is computed. The runner writes it into the marked block automatically.

**Amendment, 2026-10-27 (later the same day), before any smoke test or full run.** A rented GPU became available, so the official run moves from the laptop to it. Hardware, precision and time budget changed (§1, §5); nothing else did. The laptop plan (RTX 5060, 4-bit NF4, 13 h) was never run.

**Amendment 2, 2026-10-28, before any accuracy was viewed.** Decoding now matches the LatentMAS reference code (`run.py` defaults and `models.py`, checked directly): Qwen3 thinking mode on (the chat-template default, which LatentMAS never disables), temperature 0.6, top-p 0.95 (§3). The reason is to make the baseline comparable to the published LatentMAS numbers. Nothing else changed. A 1-hour test of the previous settings (greedy, thinking off) was started on the pod and stopped before any config finished. Its files were moved to `results_superseded/` and are not part of the reported results.

**Amendment 3, 2026-10-28, before any accuracy was viewed.** The A6000 pod stopped and its disk was lost before any config finished, so no results exist from it. The run moves to a single RunPod **A40 (48 GB)**, on-demand, still with Qwen3-8B in bf16. The GSM8K time budget stays at **12 h** (about $0.59/h, within the remaining rental money). Nothing else changed. If an A40 is unavailable, any GPU with ≥ 24 GB may be used instead; the GPU actually used is recorded in `results/plan_lock.json`.

## 1. Model and precision
- **Primary:** `Qwen/Qwen3-8B` in **bf16** (no quantization), on one NVIDIA A40 (48 GB), rented on RunPod and launched with `run_runpod.sh`.
- **Fallback:** `Qwen/Qwen3-4B` in bf16. It is used only if the primary hits a CUDA out-of-memory error during the smoke test, or if the primary's smoke timings give N < 200 ("unusably slow"). The decision is automatic and is made from timing or OOM only, never from accuracy. It is recorded in `results/plan_lock.json` (`fallback_reason`).
- Every config uses the same model and precision. The model actually used is in `results/plan_lock.json` and at the top of `SUMMARY.md`.

## 2. Configs
| config | description |
| --- | --- |
| `baseline` | Single model, one user turn, English chain-of-thought, final answer in `\boxed{}`. |
| `latent` | Four sequential delegates (algebra → counting → number-theory → geometry-verify) on one shared KV cache. Each delegate reads its prompt and then takes **30 latent steps**: the last hidden state is rescaled to the mean input-embedding norm and fed back as the next input embedding. The final decoder prompt is appended, and the answer is generated with the same decoding settings as the baseline. |

The decoder and the baseline use the identical answer instruction (`psiduck/roles.py`). No other configs are part of this run. The early-exit variant (`latent-early-exit`) is implemented but **not** part of this run; if it is run later, it will be registered separately with a fixed threshold.

## 3. Decoding
- Qwen3 **thinking mode on** (`enable_thinking=True`, the chat-template default) for every prompt, as in LatentMAS. Thinking text counts toward the token budget.
- **Sampling at temperature 0.6, top-p 0.95**, as in LatentMAS. One run per problem, batch size 1. The sampling seed for problem `idx` is `42 × 1,000,003 + idx`, with an offset of 500,000 for AIME. Both configs use the same seed for the same problem, and a resumed run reproduces it.
- **Token budget:** `max_new_tokens = 2048` for the baseline and for the latent decoder, on both GSM8K and AIME. This matches the budget in the LatentMAS reference GSM8K commands. With thinking on, some outputs will hit the cap before the final answer; that count is reported per config. Latent steps do not count against this budget; they are reported separately as forward passes.
- Generation stops at `<|im_end|>` or `<|endoftext|>`.

## 4. Datasets and answers
- **GSM8K:** `openai/gsm8k`, config `main`, test split, 1,319 problems. Gold is the number after `####` with commas removed.
- **AIME:** `HuggingFaceH4/aime_2024` (30) + `MathArena/aime_2025` (30) = 60 problems, in that fixed order.
- Each dataset is downloaded once and cached in `data/*.jsonl`. Runs read only the cache.
- **Prediction:** the last `\boxed{}` containing a number. Fallbacks, in order: `####`, then "answer is", then the last number in the output. Commas, `$`, `%` and LaTeX text wrappers are stripped. A prediction is correct if it is numerically equal to gold.
- A problem that raises an error, or kills the process twice, is recorded as an error and counted **incorrect**. Error counts are reported.

## 5. Problem subset rule
1. Shuffle the GSM8K test split with `random.Random(42).shuffle`.
2. Smoke test: run the first 3 shuffled problems for each config, after one untimed warm-up generation, and record **timing only**. Smoke records contain no predictions or correctness.
3. N = floor(12 h ÷ (sum over configs of mean seconds per problem) ÷ 50) × 50, capped at 1,319, with a minimum of 50. The 12 h budget keeps the whole run (setup, GSM8K, AIME) at about 15 h, finishing before the submission deadline.
4. Both configs run the same first N shuffled problems.

<!-- N:START -->
**N = 500** (locked automatically at 2026-10-28T21:30:30, before any accuracy was computed). Smoke-test seconds/problem: latent 29.9, baseline 52.8; sum 82.7s; budget 12 h.
<!-- N:END -->

**Run order:** GSM8K latent → GSM8K baseline → AIME latent → AIME baseline. AIME runs only after GSM8K finishes ("if time allows"). AIME results are reported only if both configs complete all 60 problems; otherwise, for example if the pod is stopped first, they are listed as incomplete.

## 6. Metrics
For each config: accuracy (k/N), mean generated tokens per problem, mean forward passes per problem, mean wall-clock seconds per problem, error count, and number of outputs that hit the token cap.

Paired comparison, latent vs baseline, on the same problems:
- 2×2 table of correct and incorrect results.
- Accuracy difference in percentage points.
- **Exact two-sided McNemar p-value** (binomial on the discordant pairs), with α = 0.05.
- Token reduction %: 1 − (latent generated tokens ÷ baseline generated tokens).
- Forward-pass ratio.

`SUMMARY.md` states plainly whether latent is worse than, equal to or better than the baseline, and whether the difference is significant.

## 7. Integrity rules
- No re-running, tuning, prompt edits or dropped configs after any accuracy is seen. Runs resume after crashes; they are never restarted from scratch to get different numbers.
- Every number in the README comes from `SUMMARY.md`, which is generated by `analyze.py` from `results/`.
- Any deviation from this plan, including crashes, a fallback model, or AIME not finishing, is reported in `SUMMARY.md` or alongside it.

## Experiment 2: AMC 2023 at a 4,096-token cap (added 2026-10-29, after Experiment 1 finished)
Experiment 1 (above) is complete and its results stand unchanged; this is an additional, separate run.
- **Data:** `math-ai/amc23`, test split, all 40 problems (numeric answers), in dataset order.
- **Configs and settings:** identical to Experiment 1 (Qwen3-8B bf16 on the same A40, thinking on, temperature 0.6, top-p 0.95, per-problem seeds), except **`max_new_tokens = 4096`** for both configs. The higher cap was chosen because AIME at 2,048 tokens truncated every output.
- **Disclosure:** before this section was written, an exploratory smoke test was run on the first 2 problems (both configs answered both correctly; nothing else was looked at). It was stopped and is not counted; the full run re-solves those problems with their own seeds.
- **Order and metrics:** latent on all 40, then baseline on all 40; same metrics and exact McNemar test as Experiment 1. Run with `scripts/run_extra.py amc23 --max-new-tokens 4096`.

## Experiment 3: GSM8K at a 4,096-token cap (added 2026-10-30, before it was run)
Experiments 1 and 2 are complete and their results stand unchanged. This run tests one question raised by Experiment 1: does latent's GSM8K advantage persist when the baseline has room to finish (Experiment 1's baseline hit the 2,048-token cap on 185 of 500 problems)?
- **Problems:** the first 200 problems of the same seed-42 shuffled GSM8K test order as Experiment 1, with the same per-problem seeds.
- **Configs and settings:** identical to Experiment 1 (Qwen3-8B bf16, thinking on, temperature 0.6, top-p 0.95), except **`max_new_tokens = 4096`** for both configs.
- **Order:** interleaved (latent then baseline on each problem), so a run cut short by budget still leaves complete pairs. Only problems finished by both configs enter the paired comparison; the number of completed pairs is reported.
- **Metrics:** same as Experiment 1 (accuracy, generated tokens, forward passes, wall-clock time, cap hits, exact McNemar). No accuracy from this run was viewed before this section was written.
- Run with `scripts/run_extra.py gsm8k --max-new-tokens 4096 --shuffled --n 200 --interleave --out gsm8k_4096`.
