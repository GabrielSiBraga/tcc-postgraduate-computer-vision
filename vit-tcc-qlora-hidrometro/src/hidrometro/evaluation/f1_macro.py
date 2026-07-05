"""Métricas categóricas F1 macro."""

from __future__ import annotations

from sklearn.metrics import classification_report, f1_score


def macro_f1(references: list[str], predictions: list[str]) -> float:
    return float(f1_score(references, predictions, average="macro", zero_division=0))


def classification_summary(
    references: list[str],
    predictions: list[str],
) -> dict:
    report = classification_report(
        references,
        predictions,
        output_dict=True,
        zero_division=0,
    )
    return report
