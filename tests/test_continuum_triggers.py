"""Triggers, drift detectors and window statistics (continuum C6, issue #161).

The rows are hand-written; nothing is trained (docs/RULES.md).
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from onion_fl.continuum.drift import Reference, detectors, window
from onion_fl.continuum.pace import View
from onion_fl.continuum.triggers import drift_detectors, triggers


def ph(**params):
    return detectors.create("page_hinkley", params)


def test_page_hinkley_ignores_a_flat_series() -> None:
    detector = ph()
    assert all(detector.update(0.2) is None for _ in range(50))


def test_page_hinkley_detects_a_sustained_rise_then_starts_over() -> None:
    detector = ph(threshold=0.5)
    found = [detector.update(x) for x in [0.1] * 10 + [0.9] * 10]

    first = next(i for i, f in enumerate(found) if f is not None)
    assert first >= 10 and found[first] > 0.5  # after the rise, with its statistic
    assert detector.n < 10  # it started over


def test_page_hinkley_waits_for_its_minimum_windows() -> None:
    detector = ph(threshold=0.1, min_samples=5)
    assert [detector.update(x) for x in (0.0, 1.0, 1.0, 1.0)] == [None] * 4


def stream(X, y, history, label_at=None):
    n = len(y)
    data = SimpleNamespace(X=np.asarray(X, float), y=np.asarray(y), n_classes=2)
    label_at = np.zeros(n) if label_at is None else np.asarray(label_at, float)
    return SimpleNamespace(
        data=data,
        history=np.asarray(history, bool),
        available_at=np.zeros(n),
        label_at=label_at,
    )


def test_data_drift_is_the_feature_shift_in_history_deviations() -> None:
    # history at 0 and 2 (std 1); the first window sets the reference at 5
    s = stream([[0.0], [2.0], [5.0], [9.0]], [0, 1, 0, 1], [1, 1, 0, 0])
    first, second = np.array([0, 0, 1, 0], bool), np.array([0, 0, 0, 1], bool)
    reference = Reference(s)

    assert window("data", s, first, first, None, None, reference) == (0.0, 1)
    value, n = window("data", s, second, second, None, None, reference)

    assert (value, n) == (pytest.approx(4.0), 1)


def test_prior_drift_is_the_class_shift_of_the_labels_that_arrived() -> None:
    s = stream([[0.0]] * 6, [0, 0, 0, 1, 1, 1], [1, 1, 0, 0, 0, 0])
    first, second = (
        np.array([0, 0, 1, 1, 0, 0], bool),
        np.array([0, 0, 0, 0, 1, 1], bool),
    )
    reference = Reference(s)

    assert window("prior", s, first, first, None, None, reference) == (0.0, 2)
    value, n = window("prior", s, second, second, None, None, reference)

    assert (value, n) == (pytest.approx(0.5), 2)  # half and half, then all 1


@pytest.mark.parametrize("kind", ["data", "prior"])
def test_the_reference_holds_until_a_drift_moves_it(kind: str) -> None:
    s = stream(
        [[0.0], [0.0], [0.0], [5.0], [5.0], [5.0]],
        [0, 0, 0, 1, 1, 1],
        [1, 1, 0, 0, 0, 0],
    )
    rows = [np.eye(6, dtype=bool)[i] for i in range(2, 6)]
    reference = Reference(s)

    values = [window(kind, s, r, r, None, None, reference)[0] for r in rows[:3]]
    reference.moved(kind)  # the drift was found in the last window
    after = window(kind, s, rows[3], rows[3], None, None, reference)[0]

    assert values[0] == 0.0 and values[1] > 0 and values[2] == values[1]
    assert after == pytest.approx(0.0)


def test_a_gradual_drift_is_detected() -> None:
    ramp = np.arange(21) * 0.1  # ten history deviations over twenty windows
    X = [[-1.0], [1.0]] + [[x] for x in ramp]
    s = stream(X, [0] * 23, [1, 1] + [0] * 21)
    reference, detector = Reference(s), ph()

    found = [
        detector.update(window("data", s, r, r, None, None, reference)[0])
        for r in (np.eye(23, dtype=bool)[i] for i in range(2, 23))
    ]

    assert any(f is not None for f in found)


def test_performance_drift_is_the_error_of_the_stored_predictions() -> None:
    s = stream([[0.0]] * 4, [0, 1, 1, 1], [0, 0, 0, 0])
    logits = np.array([[2.0, 0.0], [2.0, 0.0], [0.0, 2.0], [np.nan, np.nan]])
    predicted = np.array([1, 1, 1, 0], bool)
    labelled = np.ones(4, bool)

    value, n = window("performance", s, labelled, labelled, logits, predicted, None)

    assert (value, n) == (pytest.approx(1 / 3), 3)  # the unpredicted row is left out


def test_a_window_with_nothing_for_its_kind_rests_on_no_rows() -> None:
    s = stream([[0.0]] * 2, [0, 1], [1, 1])
    nothing = np.zeros(2, bool)

    for kind in ("data", "prior", "performance"):
        assert window(kind, s, nothing, nothing, None, nothing, Reference(s))[1] == 0


def fired(spec, **view) -> str | None:
    name, params = (spec, {}) if isinstance(spec, str) else (spec["name"], spec)
    trigger = triggers.create(name, {k: v for k, v in params.items() if k != "name"})
    return trigger.fired(View(**({"now": 0.0, "since": 0.0, "volume": 0.0} | view)))


def test_a_schedule_fires_once_its_data_time_has_passed() -> None:
    every = {"name": "schedule", "every": "5m"}
    assert fired(every, now=299.0) is None
    assert fired(every, now=400.0, since=100.0) == "schedule"


def test_a_volume_trigger_fires_at_its_rows() -> None:
    assert fired({"name": "volume", "samples": 10}, volume=9) is None
    assert fired({"name": "volume", "samples": 10}, volume=10) == "volume"


def test_a_drift_trigger_fires_on_its_kind_only() -> None:
    prior = {"name": "drift", "kind": "prior"}
    assert fired(prior, drift={"data": 2}) is None
    assert fired(prior, drift={"prior": 1}) == "drift/prior"


def test_any_names_the_first_trigger_that_fired() -> None:
    both = {
        "name": "any",
        "of": [{"name": "schedule", "every": 600}, {"name": "volume", "samples": 5}],
    }
    assert fired(both, now=100.0, volume=7) == "volume"
    assert fired(both, now=700.0, volume=7) == "schedule"
    assert fired(both, now=100.0, volume=1) is None


def test_the_detectors_come_from_every_drift_trigger() -> None:
    nested = triggers.create(
        "any",
        {"of": [{"name": "drift", "kind": "prior"}, {"name": "volume", "samples": 1}]},
    )
    stricter = {"name": "page_hinkley", "threshold": 1.0}
    local = triggers.create("drift", {"kind": "data", "detector": stricter})

    found = drift_detectors(nested, local, None)

    assert found == {"prior": "page_hinkley", "data": stricter}


def test_one_kind_cannot_have_two_detectors() -> None:
    stricter = {"name": "page_hinkley", "threshold": 2.0}
    a = triggers.create("drift", {"kind": "prior"})
    b = triggers.create("drift", {"kind": "prior", "detector": stricter})
    with pytest.raises(ValueError, match="prior"):
        drift_detectors(a, b)
