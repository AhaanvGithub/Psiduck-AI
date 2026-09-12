"""Demonstrate that all agents operate in one shared latent space.

Loads the model once, wires up the orchestrator, and shows that:
  * every agent references the identical backbone object (same id), and
  * a latent (pooled hidden-state) vector for a probe string can be read back
    from the shared backbone.

This is the concrete evidence behind "4 delegates staying in the same latent
space" while only paying to load / run one set of weights.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from psiduck.backbone import SharedModel
from psiduck.config import get_settings
from psiduck.orchestrator import LatentDelegationOrchestrator


def main() -> int:
    settings = get_settings()
    print(f"Loading shared backbone: {settings.model_name}")
    model = SharedModel(settings.model_name, torch_threads=settings.torch_threads)
    orch = LatentDelegationOrchestrator(model)

    agents = [orch.controller, *orch.delegates, orch.aggregator]
    ids = {a.name: id(a.model) for a in agents}
    print("\nAgent -> backbone object id:")
    for name, oid in ids.items():
        print(f"  {name:16s} {oid}")

    unique_ids = set(ids.values())
    print(f"\nUnique backbone objects across {len(agents)} agents: {len(unique_ids)}")
    assert len(unique_ids) == 1, "agents do not share a single backbone!"

    vec = model.latent("Find the remainder when 7^100 is divided by 1000.")
    print(f"\nShared latent vector dimension: {len(vec)}")
    print(f"First 8 dims: {[round(x, 4) for x in vec[:8]]}")
    print("\nOK: all agents share exactly one backbone / latent space.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
