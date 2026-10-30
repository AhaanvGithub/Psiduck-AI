import pytest

from psiduck import data


def _fake_gsm8k(n=1319):
    return [{"question": f"q{i}", "answer": f"work\n#### {i:,}"} for i in range(n)]


def test_gsm8k_downloads_once_then_reads_cache(tmp_path):
    calls = []

    def fetch(name):
        calls.append(name)
        return _fake_gsm8k()

    a = data.load("gsm8k", data_dir=tmp_path, fetch=fetch)
    b = data.load("gsm8k", data_dir=tmp_path, fetch=lambda n: pytest.fail("network used twice"))
    assert calls == ["gsm8k"]
    assert len(a) == len(b) == 1319
    assert a[1234].answer == "1234"  # commas stripped


def test_wrong_size_is_rejected(tmp_path):
    with pytest.raises(ValueError):
        data.load("gsm8k", data_dir=tmp_path, fetch=lambda n: _fake_gsm8k(10))
    assert not (tmp_path / "gsm8k.jsonl").exists()


def test_aime_combines_both_years(tmp_path):
    def fetch(name):
        return [{"problem": f"{name} p{i}", "answer": str(i), "id": i} for i in range(30)]

    probs = data.load("aime", data_dir=tmp_path, fetch=fetch)
    assert len(probs) == 60
    assert probs[0].problem_id.startswith("aime24") and probs[-1].problem_id.startswith("aime25")


def test_amc23_uses_question_field(tmp_path):
    def fetch(name):
        return [{"id": i, "question": f"q{i}", "answer": str(i)} for i in range(40)]

    probs = data.load("amc23", data_dir=tmp_path, fetch=fetch)
    assert len(probs) == 40 and probs[3].question == "q3" and probs[3].answer == "3"


def test_retries_with_backoff(monkeypatch):
    monkeypatch.setattr(data.time, "sleep", lambda s: None)
    attempts = []

    def flaky():
        attempts.append(1)
        if len(attempts) < 3:
            raise TimeoutError("ReadTimeout (timeout=30)")
        return "ok"

    assert data.with_retries(flaky, log=lambda m: None) == "ok"
    assert len(attempts) == 3


def test_shuffle_is_deterministic():
    items = list(range(1319))
    assert data.shuffled(items, 42) == data.shuffled(items, 42)
    assert data.shuffled(items, 42) != items
    assert data.shuffled(items, 42)[:5] == data.shuffled(list(range(1319)), 42)[:5]
