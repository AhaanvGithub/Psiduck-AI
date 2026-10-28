"""SharedModel.next_token without loading a real model (CPU, tiny fake lm_head)."""

import pytest

torch = pytest.importorskip("torch")

from psiduck.backbone import SharedModel  # noqa: E402
from psiduck.runner import problem_seed  # noqa: E402


class _LM:
    def __init__(self, logits):
        self.lm_head = lambda h: torch.tensor([logits])


def _model(logits, temperature, top_p=0.95, seed=0):
    m = SharedModel.__new__(SharedModel)
    m.torch = torch
    m.model = _LM(logits)
    m.temperature, m.top_p = temperature, top_p
    m.generator = torch.Generator()
    m.generator.manual_seed(seed)
    return m


def test_greedy_when_temperature_zero():
    assert _model([0.0, 5.0, 1.0], 0).next_token(None) == 1


def test_top_p_excludes_tail_tokens():
    # Token 0 holds ~97% of the mass, so top-p 0.95 keeps only token 0.
    m = _model([10.0, 6.5, 0.0, 0.0], 1.0, top_p=0.95)
    assert {m.next_token(None) for _ in range(200)} == {0}


def test_sampling_is_reproducible_per_seed():
    def draws(seed):
        m = _model([1.0, 1.0, 1.0, 1.0], 0.6, top_p=1.0, seed=seed)
        return [m.next_token(None) for _ in range(30)]

    assert draws(7) == draws(7)
    assert len(set(draws(7))) > 1


def test_problem_seed_paired_and_distinct():
    assert problem_seed(42, "gsm8k", 3) == problem_seed(42, "gsm8k", 3)
    assert problem_seed(42, "gsm8k", 3) != problem_seed(42, "gsm8k", 4)
    assert problem_seed(42, "gsm8k", 3) != problem_seed(42, "aime", 3)
