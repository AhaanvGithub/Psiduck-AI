"""Runtime configuration, driven by environment variables with safe defaults."""

from __future__ import annotations

import os
from dataclasses import dataclass


DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


@dataclass(frozen=True)
class Settings:
    """Configuration for the shared model and orchestration.

    Everything is overridable through environment variables so the same code
    runs against a tiny CPU model in CI and a larger model on a bigger box.
    """

    model_name: str = DEFAULT_MODEL
    max_new_tokens: int = 192
    controller_max_new_tokens: int = 224
    aggregator_max_new_tokens: int = 160
    temperature: float = 0.0
    seed: int = 0
    torch_threads: int = 0  # 0 => let torch decide / use all cores


def _int_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _float_env(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def get_settings() -> Settings:
    """Build Settings from the environment (all optional)."""
    return Settings(
        model_name=os.environ.get("PSIDUCK_MODEL", DEFAULT_MODEL),
        max_new_tokens=_int_env("PSIDUCK_MAX_NEW_TOKENS", 192),
        controller_max_new_tokens=_int_env("PSIDUCK_CONTROLLER_MAX_NEW_TOKENS", 224),
        aggregator_max_new_tokens=_int_env("PSIDUCK_AGGREGATOR_MAX_NEW_TOKENS", 160),
        temperature=_float_env("PSIDUCK_TEMPERATURE", 0.0),
        seed=_int_env("PSIDUCK_SEED", 0),
        torch_threads=_int_env("PSIDUCK_TORCH_THREADS", 0),
    )
