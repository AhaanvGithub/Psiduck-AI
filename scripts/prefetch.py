"""Download the model weights and all benchmark data once, with retries.

    python scripts/prefetch.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from psiduck import data  # noqa: E402  (also configures HF_HOME)
from psiduck.config import Settings  # noqa: E402
from psiduck.runner import FALLBACK_MODEL  # noqa: E402


def main() -> int:
    from huggingface_hub import snapshot_download

    s = Settings.from_env()
    for name in ("gsm8k", "aime24", "aime25"):
        print(f"{name}: {len(data.load(name))} problems cached in data/{name}.jsonl")
    # The fallback model is fetched too, so a fallback overnight needs no network.
    for model in dict.fromkeys([s.model_name, FALLBACK_MODEL]):
        print(f"Downloading {model} ...")
        path = data.with_retries(
            lambda: snapshot_download(model, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model"]),
            attempts=8,
        )
        print(f"{model} cached at {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
