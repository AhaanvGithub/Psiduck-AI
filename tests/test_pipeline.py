from psiduck.pipeline import BaselineSolver, LatentSolver
from psiduck.roles import DELEGATES
from tests.fakes import FakeBackbone


def test_baseline_generates_text_and_extracts():
    b = FakeBackbone()
    r = BaselineSolver(b, max_new_tokens=2048).solve("What is 6*7?")
    assert r.prediction == "42"
    assert r.compute["latent_steps"] == 0
    assert r.compute["generated_tokens"] == len(b.answer_ids)


def test_latent_runs_delegates_sequentially_on_one_cache():
    b = FakeBackbone()
    r = LatentSolver(b, max_new_tokens=2048, latent_steps=30).solve("What is 6*7?")
    assert r.prediction == "42"
    assert len(b.sessions) == 1  # one shared KV cache for every agent
    assert r.detail["steps_per_delegate"] == [30] * len(DELEGATES) == [30, 30, 30, 30]
    assert r.compute["latent_steps"] == 120
    cache = b.sessions[0].cache
    # Layout: [prompt_1, 30 latent, turn_end] x 4, then decoder prompt, then generated tokens.
    kinds = [k for k, _ in cache]
    first_latent = kinds.index("latent")
    assert kinds[first_latent : first_latent + 30] == ["latent"] * 30
    assert kinds.count("latent") == 120
    # No text is generated before the decoder: generated tokens == only the final answer.
    assert r.compute["generated_tokens"] == len(b.answer_ids)
    # Decoder attends to everything: the full context includes all latent steps.
    assert r.detail["context_length"] == len(cache)


def test_token_cap_is_shared():
    b = FakeBackbone(answer_text="x" * 100)
    r = LatentSolver(b, max_new_tokens=10).solve("q")
    assert r.compute["generated_tokens"] == 10
    assert r.detail["hit_token_cap"]
    b2 = FakeBackbone(answer_text="x" * 100)
    r2 = BaselineSolver(b2, max_new_tokens=10).solve("q")
    assert r2.compute["generated_tokens"] == 10


def test_early_exit_stops_when_hidden_state_converges():
    b = FakeBackbone(converge_after=5)
    r = LatentSolver(b, max_new_tokens=64, latent_steps=30, early_exit_cos=0.999).solve("q")
    assert all(s < 30 for s in r.detail["steps_per_delegate"])
    assert r.compute["latent_steps"] == sum(r.detail["steps_per_delegate"])


def test_no_early_exit_by_default():
    b = FakeBackbone(converge_after=5)
    r = LatentSolver(b, max_new_tokens=64, latent_steps=30).solve("q")
    assert r.detail["steps_per_delegate"] == [30, 30, 30, 30]
