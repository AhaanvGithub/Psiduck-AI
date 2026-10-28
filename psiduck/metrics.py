"""Per-problem compute accounting."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class ComputeMeter:
    prompt_tokens: int = 0  # text tokens fed in (prompts)
    generated_tokens: int = 0  # tokens decoded as text output
    latent_steps: int = 0  # hidden states fed back as input (no token decoded)
    forward_passes: int = 0  # model forward calls of any kind

    def reset(self) -> None:
        self.prompt_tokens = self.generated_tokens = self.latent_steps = self.forward_passes = 0

    def snapshot(self) -> dict:
        return asdict(self)
