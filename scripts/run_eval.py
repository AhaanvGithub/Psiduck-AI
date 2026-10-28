"""Entry point for the unattended eval (called by run_overnight.ps1).

    python scripts/run_eval.py                  # smoke -> lock N -> all runs -> SUMMARY.md
    python scripts/run_eval.py --allow-battery  # don't wait for AC before the smoke test
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from psiduck.config import Settings  # noqa: E402
from psiduck.runner import Runner  # noqa: E402


def make_backbone(settings: Settings):
    from psiduck.backbone import SharedModel

    return SharedModel(settings.model_name, settings.quantization, settings.seed)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--allow-battery", action="store_true")
    args = ap.parse_args()
    return Runner(Settings.from_env(), make_backbone, allow_battery=args.allow_battery).run()


if __name__ == "__main__":
    sys.exit(main())
