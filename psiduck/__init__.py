"""Psiduck-AI: a single-latent-space, multi-agent solver for AIME problems.

One open-weights language model is loaded exactly once and shared by a
controller, four specialist delegates, and an aggregator. Because every agent
routes through the *same* weights, they operate in the same latent space and no
extra API credits are spent per delegate.
"""

from .config import Settings, get_settings
from .orchestrator import LatentDelegationOrchestrator, SolveResult

__all__ = [
    "Settings",
    "get_settings",
    "LatentDelegationOrchestrator",
    "SolveResult",
]
