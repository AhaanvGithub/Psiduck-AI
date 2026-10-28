"""Run settings. Defaults are the pre-registered values in EVAL_PLAN.md;
environment variables exist so a rented GPU can change model/quantization
for a *separate*, separately-registered run.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from typing import Optional


def _opt_float(v: Optional[str]) -> Optional[float]:
    return float(v) if v not in (None, "", "none", "None") else None


@dataclass(frozen=True)
class Settings:
    model_name: str = "Qwen/Qwen3-8B"
    quantization: str = "nf4"  # "nf4" (bitsandbytes 4-bit) or "bf16"
    max_new_tokens: int = 2048  # same cap for the baseline and the latent decoder
    latent_steps: int = 30  # per delegate
    early_exit_cos: Optional[float] = None  # None = always run all latent steps
    seed: int = 42

    @classmethod
    def from_env(cls) -> "Settings":
        d = cls()
        return cls(
            model_name=os.environ.get("PSIDUCK_MODEL", d.model_name),
            quantization=os.environ.get("PSIDUCK_QUANT", d.quantization),
            max_new_tokens=int(os.environ.get("PSIDUCK_MAX_NEW_TOKENS", d.max_new_tokens)),
            latent_steps=int(os.environ.get("PSIDUCK_LATENT_STEPS", d.latent_steps)),
            early_exit_cos=_opt_float(os.environ.get("PSIDUCK_EARLY_EXIT_COS")),
            seed=int(os.environ.get("PSIDUCK_SEED", d.seed)),
        )

    def as_dict(self) -> dict:
        return asdict(self)
