import pytest

from psiduck.answers import extract_answer, gsm8k_gold, is_correct, normalize_number


@pytest.mark.parametrize(
    "text, expected",
    [
        ("so we get \\boxed{72}.", "72"),
        ("first \\boxed{3} then actually \\boxed{5}", "5"),
        ("\\boxed{\\frac{1}{2}} no wait \\boxed{1,200}", "1200"),
        ("\\boxed{\\$18.00}", "18"),
        ("Total: 10 + 8 = 18\n#### 18", "18"),
        ("The answer is 1,234 apples.", "1234"),
        ("blah 3 blah 7", "7"),
        ("\\boxed{-4}", "-4"),
        ("\\boxed{2.50}", "2.5"),
        ("\\boxed{50\\%}", "50"),
        ("\\boxed{\\text{12 dollars}} hmm 12", "12"),
        ("no numbers here", None),
        ("", None),
    ],
)
def test_extract_answer(text, expected):
    assert extract_answer(text) == expected


def test_gsm8k_gold():
    assert gsm8k_gold("She has 3+4=<<3+4=7>>7.\n#### 7") == "7"
    assert gsm8k_gold("...\n#### 1,000") == "1000"
    with pytest.raises(ValueError):
        gsm8k_gold("no marker")


def test_is_correct_numeric():
    assert is_correct("18", "18")
    assert is_correct(normalize_number("18.0"), "18")
    assert not is_correct("17", "18")
    assert not is_correct(None, "18")
