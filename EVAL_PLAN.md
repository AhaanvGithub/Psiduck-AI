# Psiduck-AI v2 evaluation plan (pre-registered)

Written on 2026-10-27, before any smoke test or full run. Nothing below may be changed after results are seen. The only field filled in later is N, which is computed mechanically from smoke-test timings by the rule in §5, before any accuracy is computed. The runner writes it into the marked block automatically.

**Amendment, 2026-10-27 (later the same day), before any smoke test or full run.** A rented GPU became available, so the official run moves from the laptop to it. Hardware, precision and time budget changed (§1, §5); nothing else did. The laptop plan (RTX 5060, 4-bit NF4, 13 h) was never run.

## 1. Model and precision
- **Primary:** `Qwen/Qwen3-8B` in **bf16** (no quantization), on one NVIDIA RTX A6000 (48 GB), rented on RunPod and launched with `run_runpod.sh`.
- **Fallback:** `Qwen/Qwen3-4B` in bf16. It is used only if the primary hits a CUDA out-of-memory error during the smoke test, or if the primary's smoke timings give N < 200 ("unusably slow"). The decision is automatic and is made from timing or OOM only, never from accuracy. It is recorded in `results/plan_lock.json` (`fallback_reason`).
- Every config uses the same model and precision. The model actually used is in `results/plan_lock.json` and at the top of `SUMMARY.md`.

## 2. Configs
| config | description |
| --- | --- |
| `baseline` | Single model, one user turn, English chain-of-thought, final answer in `\boxed{}`. |
| `latent` | Four sequential delegates (algebra → counting → number-theory → geometry-verify) on one shared KV cache. Each delegate reads its prompt and then takes **30 latent steps**: the last hidden state is rescaled to the mean input-embedding norm and fed back as the next input embedding. The final decoder prompt is appended, and the answer is generated greedily. |

The decoder and the baseline use the identical answer instruction (`psiduck/roles.py`). No other configs are part of this run. The early-exit variant (`latent-early-exit`) is implemented but **not** part of this run; if it is run later, it will be registered separately with a fixed threshold.

## 3. Decoding
- Qwen3 non-thinking mode (`enable_thinking=False`) for every prompt.
- Greedy decoding (argmax, equivalent to temperature 0), seed 42, batch size 1.
- **Token budget:** `max_new_tokens = 2048` for the baseline and for the latent decoder, on both GSM8K and AIME. This matches the budget in the LatentMAS reference GSM8K commands. Latent steps do not count against this budget; they are reported separately as forward passes.
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
3. N = floor(12 h ÷ (sum over configs of mean seconds per problem) ÷ 50) × 50, capped at 1,319, with a minimum of 50. The 12 h budget comes from the $20 rental budget at about $0.90/h. The pod is hard-stopped 20 h after launch.
4. Both configs run the same first N shuffled problems.

<!-- N:START -->
**N = _pending_.** Locked automatically from smoke timings before the full runs.
<!-- N:END -->

**Run order:** GSM8K latent → GSM8K baseline → AIME latent → AIME baseline. AIME runs only after GSM8K finishes ("if time allows"). AIME results are reported only if both configs complete all 60 problems; otherwise, for example if the 20 h hard stop hits first, they are listed as incomplete.

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
