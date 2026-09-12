# Psiduck-AI

A single-latent-space, multi-agent solver for [AIME](https://en.wikipedia.org/wiki/American_Invitational_Mathematics_Examination) problems.

One open-weights language model is loaded **exactly once** and shared by a
controller, four specialist delegates, and an aggregator. Because every agent
routes through the *same* weights, they all operate in the same latent space,
and there are **no per-delegate API credits** — the model is local and loaded a
single time.

```
                 ┌──────────────┐
   problem  ──▶  │  controller  │  writes a short plan
                 └──────┬───────┘
                        ▼
      ┌──────────┬──────────┬────────────────┬───────────┐
      │ algebra  │ counting │ number-theory  │ verifier  │  4 delegates
      └────┬─────┴────┬─────┴───────┬────────┴─────┬──────┘
           └──────────┴─────────────┴──────────────┘
                        ▼
                 ┌──────────────┐
                 │ aggregator   │  reconciles → final integer (0–999)
                 └──────────────┘

All five agents share ONE backbone instance (one set of weights, one latent space).
```

## Why "same latent space"?

Every `Agent` holds a reference to the **same** `Backbone` object
(`psiduck/backbone.py`). Specialization comes from each agent's system prompt,
not from a separate model. The invariant is enforced in
`LatentDelegationOrchestrator.__init__` and demonstrated by
`scripts/demo_shared_latent.py`, which prints the shared object id for all
agents and reads a pooled hidden-state vector from the shared backbone.

## Requirements

- Python 3.10+ (developed on 3.12)
- CPU is fine — the default model is `Qwen/Qwen2.5-0.5B-Instruct` (Apache-2.0)

## Setup

```bash
bash scripts/install.sh      # installs deps + prefetches the model into the HF cache
```

This is exactly what the Cloud Agent environment runs on `install`
(`.cursor/environment.json`).

## Usage

```bash
python main.py --list                 # show built-in sample problems
python main.py --problem-index 0      # solve a sample problem end-to-end
python main.py --problem "Find the remainder when 7^100 is divided by 1000."

python scripts/demo_shared_latent.py  # prove all agents share one latent space
```

Configuration is via environment variables (all optional):

| Variable | Default | Meaning |
| --- | --- | --- |
| `PSIDUCK_MODEL` | `Qwen/Qwen2.5-0.5B-Instruct` | Hugging Face model id |
| `PSIDUCK_MAX_NEW_TOKENS` | `192` | Tokens per delegate answer |
| `PSIDUCK_CONTROLLER_MAX_NEW_TOKENS` | `224` | Tokens for the plan |
| `PSIDUCK_AGGREGATOR_MAX_NEW_TOKENS` | `160` | Tokens for aggregation |
| `PSIDUCK_TEMPERATURE` | `0.0` | >0 enables sampling |
| `PSIDUCK_TORCH_THREADS` | `0` (auto) | CPU threads for torch |

> Note: a 0.5B CPU model is used so the pipeline runs anywhere without a GPU or
> credits. It demonstrates the **architecture** end-to-end; it is not expected to
> reliably solve hard AIME problems. Point `PSIDUCK_MODEL` at a larger model on
> capable hardware for real accuracy.

## Tests

```bash
python -m pytest
```

Unit tests use a dependency-free `FakeBackbone` (no torch, no downloads) to
verify the orchestration wiring, the single-backbone invariant, and AIME answer
parsing/normalization.

## Project layout

```
psiduck/
  backbone.py       # Backbone protocol + SharedModel (one HF model, loaded once)
  agents.py         # Agent role + controller/delegate/aggregator prompts
  orchestrator.py   # LatentDelegationOrchestrator: controller → 4 delegates → aggregator
  aime.py           # answer extraction + normalization to [0, 999]
  data.py           # sample AIME problems
  config.py         # env-driven settings
main.py             # CLI: solve a problem, print the full trace
scripts/
  install.sh        # idempotent env bootstrap (deps + model prefetch)
  prefetch_model.py # warm the HF cache
  demo_shared_latent.py
tests/              # pytest suite (FakeBackbone)
```
