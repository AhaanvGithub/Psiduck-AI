#!/usr/bin/env bash
# Idempotent environment bootstrap for Psiduck-AI.
#
# 1. Install Python dependencies (CPU-only torch, transformers, pytest).
# 2. Prefetch the default model into the Hugging Face cache so agent runtime
#    (and the snapshot built from it) need no network for inference.
#
# Safe to run repeatedly: pip is a no-op when satisfied and the model prefetch
# is skipped when the weights are already cached.
set -euo pipefail

cd "$(dirname "$0")/.."

echo "==> Upgrading pip"
python3 -m pip install --upgrade pip

echo "==> Installing Python dependencies"
python3 -m pip install -r requirements.txt

echo "==> Prefetching model weights into HF cache"
python3 scripts/prefetch_model.py

echo "==> Install complete"
