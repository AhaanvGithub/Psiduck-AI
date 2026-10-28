"""Build SUMMARY.md (and the README results block) from results/.

    python analyze.py
"""

from __future__ import annotations

import json
import sys

from psiduck.analysis import build_summary, inject_readme
from psiduck.hf_env import REPO_ROOT


def main() -> int:
    results = REPO_ROOT / "results"
    plan_path = results / "plan_lock.json"
    plan = json.loads(plan_path.read_text()) if plan_path.exists() else None
    summary = build_summary(results, plan)
    (REPO_ROOT / "SUMMARY.md").write_text(summary, encoding="utf-8")
    inject_readme(REPO_ROOT / "README.md", summary)
    print(summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
