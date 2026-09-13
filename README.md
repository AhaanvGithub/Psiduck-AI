# Psiduck-AI

**Can four LLMs that talk to each other in *latent space* solve problems a single LLM can't — while using *less* generation compute?**

Psiduck-AI is a research scaffold for exactly that question. One open-weights
model is loaded **once** and shared by an encoder/controller, four specialist
delegates, and a decoder. The agents communicate by exchanging **latent vectors**
(pooled hidden states injected into the next stage as "soft tokens" via
`inputs_embeds`) — **not** English text. The only natural language ever
*generated* is the final integer answer.

```
 problem ─(embed)─▶ encode ──▶ context latent
                                   │
        ┌──────────┬──────────────┼───────────────┬───────────┐
        ▼          ▼              ▼                ▼
     algebra   counting     number-theory     geometry/verify     (4 delegates)
        │          │              │                │
        └──────────┴──────┬───────┴────────────────┘
                          ▼  (latent messages — zero English generated)
                    fuse + decode ──▶ final integer (short generation)
```

## Why this could be cheaper than one LLM

The expensive part of inference is **autoregressive generation**: every output
token is one sequential forward step. A single LLM solving a hard problem
"thinks out loud" in English for hundreds of tokens. Here, the agents exchange
fixed-size latent vectors (one forward pass each, **zero generated tokens**), and
only the final answer is decoded. So the multi-agent pipeline generates far
fewer tokens than an English chain-of-thought — see the `gen_tokens` column in
the demo output.

> **Honest scope.** This scaffold demonstrates the *mechanism* (latent AI-to-AI
> communication over a single shared latent space) and the *compute profile*
> (few generated tokens). It does **not** claim a small, untrained model beats a
> baseline on hard AIME problems — closing the accuracy gap is the research this
> scaffold is built to enable. Point `PSIDUCK_MODEL` at a heavier model on
> capable hardware to study accuracy at scale.

## Requirements

- Python 3.10+ (developed on 3.12), CPU is fine for the default 0.5B model
- Linux/macOS; `scripts/install.sh` creates a local virtualenv

## Setup

```bash
bash scripts/install.sh        # venv at .venv + deps + model prefetch into .hf_cache
```

This is exactly what the Cloud Agent environment runs on `install`
(`.cursor/environment.json`). Everything is installed **inside the repo** (`.venv`,
`.hf_cache`) so it persists on the workspace volume.

## Usage

Use the virtualenv's Python (or `source .venv/bin/activate` first):

```bash
.venv/bin/python main.py --list                 # list sample problems
.venv/bin/python main.py --problem-index 2       # compare baseline vs latent on one problem
.venv/bin/python main.py --all                   # run all samples + aggregate compute
.venv/bin/python main.py --problem "Find the remainder when 7^100 is divided by 1000."

.venv/bin/python scripts/demo_shared_latent.py   # show the shared latent space
```

Each run prints both approaches side by side:

| approach | answer | correct | gen_tokens | fwd_passes | wall_s |
| --- | --- | --- | --- | --- | --- |
| single-llm-english | ... | ... | ~256 | 0 | ... |
| latent-4-agent | ... | ... | ~a few | 5 | ... |

## Configuration (environment variables)

| Variable | Default | Meaning |
| --- | --- | --- |
| `PSIDUCK_MODEL` | `Qwen/Qwen2.5-0.5B-Instruct` | Hugging Face model id (swap in a heavier model) |
| `PSIDUCK_BASELINE_MAX_NEW_TOKENS` | `256` | Baseline English chain-of-thought budget |
| `PSIDUCK_ANSWER_MAX_NEW_TOKENS` | `24` | Tokens the latent decoder may emit for the answer |
| `PSIDUCK_TEMPERATURE` | `0.0` | >0 enables sampling |
| `PSIDUCK_TORCH_THREADS` | `0` (auto) | CPU threads for torch |
| `HF_HOME` | `<repo>/.hf_cache` | Hugging Face cache location |

## Tests

```bash
.venv/bin/python -m pytest
```

Unit tests use a dependency-free `FakeBackbone` (no torch, no downloads) that
implements the same latent API and updates a real `ComputeMeter`, so they assert
both the orchestration wiring and the core thesis (latent communication
generates far fewer tokens than the English baseline).

## Project layout

```
psiduck/
  backbone.py     # SharedModel: one HF model + latent-communication primitives
  pipeline.py     # LatentDelegationSolver, SingleModelSolver, compare()
  metrics.py      # ComputeMeter / ComputeStats (generated tokens, forward passes)
  roles.py        # the four delegate role tags (embedded, never generated)
  aime.py         # answer extraction + normalization to [0, 999]
  data.py         # sample problems (easy + AIME)
  config.py       # env-driven settings
  hf_env.py       # point HF cache into the repo
main.py           # CLI: baseline vs latent comparison
scripts/          # install.sh, prefetch_model.py, demo_shared_latent.py
tests/            # pytest suite (FakeBackbone)
```
