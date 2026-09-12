"""A deterministic, dependency-free backbone for fast unit tests."""

from __future__ import annotations

from typing import Callable, List, Sequence

from psiduck.backbone import Message


class FakeBackbone:
    """Implements the Backbone protocol without torch/transformers.

    A `responder` maps (system_prompt, user_content) -> assistant text, so tests
    can script exactly what each agent role "says". Every call is recorded.
    """

    def __init__(self, responder: Callable[[str, str], str], name: str = "fake"):
        self._responder = responder
        self._name = name
        self.calls: List[dict] = []

    @property
    def name(self) -> str:
        return self._name

    def chat(self, messages: Sequence[Message], max_new_tokens: int | None = None) -> str:
        system = next((m["content"] for m in messages if m["role"] == "system"), "")
        user = next((m["content"] for m in messages if m["role"] == "user"), "")
        self.calls.append({"system": system, "user": user, "max_new_tokens": max_new_tokens})
        return self._responder(system, user)

    def latent(self, text: str) -> List[float]:
        # Deterministic pseudo-embedding based on char codes.
        return [float((ord(c) % 17) - 8) for c in text[:16]]
