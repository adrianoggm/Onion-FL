"""Tests for the metric plugins and report reduction (issue #92).

Logits and labels are hand-picked numbers to check arithmetic; nothing is trained.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from onion_fl.learning.metrics import compute, metrics, reduce_reports

LOGITS = np.array([[2.0, 0.0], [0.0, 1.0], [3.0, 0.0], [0.0, 2.0]])
Y = np.array([0, 1, 1, 1])  # predictions 0 1 0 1


def test_accuracy_and_loss() -> None:
    out = compute(LOGITS, Y, 2, ["loss", "accuracy"])

    expected = -np.mean(
        [
            2 - math.log(math.exp(2) + 1),
            1 - math.log(1 + math.e),
            0 - math.log(math.exp(3) + 1),
            2 - math.log(1 + math.exp(2)),
        ]
    )
    assert out["accuracy"] == pytest.approx(0.75)
    assert out["loss"] == pytest.approx(expected)


def test_loss_is_stable_for_large_logits() -> None:
    out = compute(np.array([[1000.0, 0.0]]), np.array([0]), 2, ["loss"])

    assert out["loss"] == pytest.approx(0.0, abs=1e-9)


def test_macro_f1() -> None:
    # class 0: tp 1 fp 1 fn 0 -> f1 2/3 ; class 1: tp 2 fp 0 fn 1 -> f1 0.8
    out = compute(LOGITS, Y, 2, ["macro_f1"])

    assert out["macro_f1"] == pytest.approx((2 / 3 + 0.8) / 2)


def test_recall_per_class_skips_absent_classes() -> None:
    out = compute(LOGITS, Y, 3, ["recall_per_class"])

    assert out == {"recall.0": 1.0, "recall.1": pytest.approx(2 / 3)}


def test_confusion_matrix_counts() -> None:
    out = compute(LOGITS, Y, 2, ["confusion_matrix"])

    assert out == {
        "confusion.0.0": 1.0,
        "confusion.0.1": 0.0,
        "confusion.1.0": 1.0,
        "confusion.1.1": 2.0,
    }


def test_registry_lists_the_built_ins() -> None:
    assert metrics.names() == [
        "accuracy",
        "confusion_matrix",
        "loss",
        "macro_f1",
        "recall_per_class",
    ]


def test_reports_are_reduced_by_samples_and_confusions_are_summed() -> None:
    reports = [
        ({"accuracy": 1.0, "confusion.0.0": 3.0}, 3),
        ({"accuracy": 0.0, "confusion.0.0": 1.0}, 1),
    ]

    out, samples = reduce_reports(reports)

    assert out == {"accuracy": 0.75, "confusion.0.0": 4.0}
    assert samples == 4


def test_reducing_nothing_gives_nothing() -> None:
    assert reduce_reports([]) == ({}, 0)
    assert reduce_reports([({"accuracy": 1.0}, 0)]) == ({}, 0)
