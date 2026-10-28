"""A dependency-free stand-in for SharedModel (no torch, no downloads).

Hidden states are small lists of floats. The fake records every input it is
fed, so tests can assert exactly what the pipeline did to the shared cache.
"""

from __future__ import annotations

import math

from psiduck.backbone import Session
from psiduck.metrics import ComputeMeter

EOS = 0


class FakeBackbone:
    def __init__(self, answer_text: str = "so the answer is \\boxed{42}", converge_after: int | None = None):
        self.meter = ComputeMeter()
        self.eos_ids = {EOS}
        self.turn_end_ids = [7, 8]
        self.answer_ids = [ord(c) for c in answer_text]
        self.converge_after = converge_after  # latent step at which hidden states stop changing
        self.sessions: list[Session] = []

    def chat_turn_ids(self, text: str) -> list[int]:
        return [ord(c) for c in text][:16] + [1, 2]

    def decode(self, ids: list[int]) -> str:
        return "".join(chr(i) for i in ids)

    def new_session(self) -> Session:
        s = Session(cache=[])
        self.sessions.append(s)
        return s

    def _hidden(self, s: Session) -> list[float]:
        if self.converge_after is not None:
            latent = sum(1 for x in s.cache if x[0] == "latent" and x[1] >= self.converge_after)
            if latent:
                return [1.0, 0.0]
        return [math.cos(len(s.cache)), math.sin(len(s.cache))]

    def feed_tokens(self, s: Session, ids: list[int]):
        self.meter.forward_passes += 1
        for t in ids:
            s.cache.append(("tok", t))
        s.length += len(ids)
        s.tokens.extend(ids)
        return self._hidden(s)

    def feed_latent(self, s: Session, h):
        self.meter.forward_passes += 1
        self.meter.latent_steps += 1
        step = sum(1 for x in s.cache if x[0] == "latent") + 1
        s.cache.append(("latent", step))
        s.length += 1
        s.tokens.append(None)
        return self._hidden(s)

    def next_token(self, h) -> int:
        # Emit answer_ids one at a time, counted from tokens generated in the latest session.
        s = self.sessions[-1]
        n = getattr(s, "_gen", 0)
        s._gen = n + 1
        return self.answer_ids[n] if n < len(self.answer_ids) else EOS

    def reseed(self, seed: int) -> None:
        self.last_seed = seed

    @staticmethod
    def cosine(a, b) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(x * x for x in b))
        return dot / (na * nb)

    def free(self) -> None:
        pass
