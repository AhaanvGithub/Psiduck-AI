"""Answer extraction and normalization for GSM8K and AIME.

Both benchmarks have a single numeric answer (GSM8K: an integer; AIME: an
integer in [0, 999]). We extract the model's final number and compare it to
the gold value numerically, so "1,200", "$1200", "1200.00" and "\\boxed{1200}"
all match a gold of 1200.
"""

from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation
from typing import Optional

_NUM = r"-?\d[\d,]*(?:\.\d+)?|-?\.\d+"
_NUM_RE = re.compile(_NUM)


def _boxed_contents(text: str) -> list[str]:
    """Return the contents of every \\boxed{...} (brace-balanced) in order."""
    out = []
    for m in re.finditer(r"\\boxed\s*\{", text):
        depth, i = 1, m.end()
        while i < len(text) and depth:
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
            i += 1
        if depth == 0:
            out.append(text[m.end() : i - 1])
    return out


def normalize_number(s: str) -> Optional[str]:
    """Canonical string for a number ("1,200.50" -> "1200.5"), or None."""
    if s is None:
        return None
    s = str(s).strip()
    # Strip LaTeX/units noise that commonly wraps a final answer.
    s = re.sub(r"\\(?:text|mathrm|textbf)\{[^}]*\}", "", s)
    s = s.replace("\\$", "").replace("$", "").replace("\\!", "").replace("\\,", "")
    s = s.replace("\\%", "").replace("%", "")
    s = s.replace("{,}", ",")
    m = _NUM_RE.search(s)
    if not m:
        return None
    raw = m.group(0).replace(",", "")
    try:
        d = Decimal(raw)
    except InvalidOperation:
        return None
    if d == d.to_integral_value():
        return str(int(d))
    return format(d.normalize(), "f")


def extract_answer(text: str) -> Optional[str]:
    """Pull the final numeric answer out of a model completion.

    Priority: last \\boxed{}, then "####", then "answer is", then the last
    number in the text.
    """
    if not text:
        return None
    boxed = _boxed_contents(text)
    for content in reversed(boxed):
        n = normalize_number(content)
        if n is not None:
            return n
    m = list(re.finditer(r"####\s*([^\n]+)", text))
    if m:
        n = normalize_number(m[-1].group(1))
        if n is not None:
            return n
    m = list(re.finditer(r"answer\s*(?:is|:)\s*([^\n]+)", text, flags=re.IGNORECASE))
    if m:
        n = normalize_number(m[-1].group(1))
        if n is not None:
            return n
    nums = _NUM_RE.findall(text)
    if nums:
        return normalize_number(nums[-1])
    return None


def gsm8k_gold(answer_field: str) -> str:
    """GSM8K's reference solutions end with '#### <answer>'."""
    m = re.search(r"####\s*(.+)$", answer_field.strip())
    if not m:
        raise ValueError(f"no '####' in GSM8K answer: {answer_field[-80:]!r}")
    n = normalize_number(m.group(1))
    if n is None:
        raise ValueError(f"unparseable GSM8K gold: {m.group(1)!r}")
    return n


def is_correct(prediction: Optional[str], gold: str) -> bool:
    if prediction is None:
        return False
    try:
        return Decimal(prediction) == Decimal(gold)
    except InvalidOperation:
        return False
