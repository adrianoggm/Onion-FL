"""Tests for the statistics of the deadline-coupling pilot (issue #213).

Hand-made score tables only: nothing is trained, nothing reads data.
"""

from __future__ import annotations

import importlib.util
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "pilot_deadline_coupling", ROOT / "scripts" / "pilot_deadline_coupling.py"
)
pilot = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(pilot)


def test_a_configuration_always_best_splits_into_agreeing_halves() -> None:
    table = np.array([[1.0, 2.0, 3.0]] * 10) + np.arange(10)[:, None] * 0.01
    out = pilot.split_half(table)
    assert out["splits"] == 126  # C(10, 5) / 2
    assert out["top1"] == 1.0 and out["regret_relative_p95"] == 0.0
    assert out["top1_chance"] == pytest.approx(1 / 3)


def test_halves_that_disagree_regret_the_other_winner() -> None:
    out = pilot.split_half([[1.0, 2.0], [2.0, 1.0]])
    assert out["splits"] == 1 and out["top1"] == 0.0
    assert out["regret_relative_mean"] == pytest.approx(1.0)
    with pytest.raises(ValueError, match="even number"):
        pilot.split_half([[1.0, 2.0]] * 3)


def test_the_relative_regret_is_against_the_best() -> None:
    assert pilot.relative_regret([4.0, 5.0, 2.0], 0) == pytest.approx(1.0)
    assert pilot.relative_regret([4.0, 5.0, 2.0], 2) == 0.0


def test_the_auc_separates_ranks_and_needs_both_classes() -> None:
    assert pilot.auc([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1]) == 1.0
    assert pilot.auc([0.9, 0.8, 0.2, 0.1], [0, 0, 1, 1]) == 0.0
    assert pilot.auc([0.5, 0.5], [0, 1]) == 0.5
    assert pilot.auc([0.1, 0.2], [1, 1]) is None
    low, high = pilot.auc_ci([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1], resamples=200)
    assert low == high == 1.0
    assert pilot.auc_ci([0.1, 0.2], [1, 1], resamples=20) is None


@pytest.mark.parametrize(
    "exceeding, cells, verdict",
    [
        (3, 8, "exists"),  # P(X >= 3 | 8, 0.05) = 0.006
        (2, 8, "inconclusive"),  # 0.057
        (1, 8, "falls"),
        (0, 8, "falls"),
        (3, 4, "exists"),
        (2, 4, "inconclusive"),
        (0, 0, "not identifiable"),
    ],
)
def test_the_gap_verdict_counts_cells_above_the_ceiling(
    exceeding: int, cells: int, verdict: str
) -> None:
    assert pilot.gap_verdict(exceeding, cells) == verdict


def test_overlapping_intervals() -> None:
    assert pilot.overlap([0.1, 0.5], [0.4, 0.9]) is True
    assert pilot.overlap([0.1, 0.3], [0.4, 0.9]) is False
    assert pilot.overlap(None, [0.4, 0.9]) is None


def _config(fog: dict, optimizer, statistic="samples", rho=0.0) -> dict:
    return {
        "topology": {"name": "pilot_dc_two_fogs", "fog": {"defaults": fog}},
        "learning": {"server_optimizer": optimizer},
        "infrastructure": {
            "assignment": {"edges": {"statistic": statistic, "rho": rho}}
        },
    }


def test_a_run_is_classified_by_its_regime_coupling_and_configuration() -> None:
    sync = pilot.classify(_config({}, "replace"))
    assert sync == {
        "regime": "sync",
        "coupling": ("samples", 0.0),
        "configuration": ("pilot_dc_two_fogs", "fedavg"),
    }
    timed = pilot.classify(_config({"deadline": "1s"}, {"name": "fedadam"}))
    assert timed["regime"] == "deadline"
    assert timed["configuration"][1] == "fedadam"
    buffered = {"deadline": "1s", "close_at_quorum": True}
    assert pilot.classify(_config(buffered, "replace", "label_divergence", -0.6)) == {
        "regime": "fedbuff",
        "coupling": ("label_divergence", -0.6),
        "configuration": ("pilot_dc_two_fogs", "fedavg"),
    }


def test_the_steps_run_the_pre_registered_cells() -> None:
    kinds = [
        {"regime": regime, "coupling": coupling}
        for regime in ("sync", *pilot.REGIMES)
        for coupling in (pilot.NAIVE, *pilot.COUPLED)
        for _ in pilot.CONFIGURATIONS
    ]
    counts = Counter()
    for kind in kinds:
        for seed in range(10):
            for step in (0, 1):
                counts[step] += pilot.wanted(step, kind, seed)
    # step 0: the controls with 5 seeds, the naive cells of both regimes with 10
    assert counts[0] == 6 * 5 + 2 * 6 * 10
    # step 1: the four couplings of both regimes with 5 seeds
    assert counts[1] == 2 * 4 * 6 * 5
