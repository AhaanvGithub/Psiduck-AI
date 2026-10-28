"""Point the Hugging Face cache into the repo and make hub downloads patient.

Import this before anything that imports huggingface_hub / transformers /
datasets, since those read the environment at import time.
"""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def configure() -> None:
    os.environ.setdefault("HF_HOME", str(REPO_ROOT / ".hf_cache"))
    # The default 10 s read timeout is what killed earlier runs; be generous.
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")
    os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "60")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    # Silence the Windows symlink warning; the cache still works without symlinks.
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")


configure()
