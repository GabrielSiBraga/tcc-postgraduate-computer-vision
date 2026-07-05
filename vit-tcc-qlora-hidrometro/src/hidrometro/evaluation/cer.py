"""Character Error Rate para avaliação OCR."""

from __future__ import annotations


def _levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        curr = [i]
        for j, cb in enumerate(b, start=1):
            insert = curr[j - 1] + 1
            delete = prev[j] + 1
            replace = prev[j - 1] + (ca != cb)
            curr.append(min(insert, delete, replace))
        prev = curr
    return prev[-1]


def character_error_rate(reference: str, hypothesis: str) -> float:
    reference = reference.strip()
    hypothesis = hypothesis.strip()
    if not reference:
        return 0.0 if not hypothesis else 1.0
    distance = _levenshtein(reference, hypothesis)
    return distance / len(reference)


def batch_cer(pairs: list[tuple[str, str]]) -> float:
    if not pairs:
        return 0.0
    return sum(character_error_rate(ref, hyp) for ref, hyp in pairs) / len(pairs)
