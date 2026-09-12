"""Download the configured model into the local HF cache.

Run during environment `install` so the weights are baked into the snapshot and
no download is needed at agent runtime. Safe to run repeatedly (idempotent): if
the weights are already cached, this is a fast no-op.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from psiduck.config import get_settings


def main() -> int:
    settings = get_settings()
    model_name = settings.model_name
    print(f"Prefetching model into HF cache: {model_name}")
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as exc:  # pragma: no cover - only hit if deps missing
        print(f"transformers not importable: {exc}", file=sys.stderr)
        return 1

    AutoTokenizer.from_pretrained(model_name)
    AutoModelForCausalLM.from_pretrained(model_name)
    print("Prefetch complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
