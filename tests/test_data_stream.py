"""A subject's rows as a stream with delayed labels (continuum C3, issue #158).

The rows and times are hand-written: these tests check the schedule (which
row arrives and becomes trainable when); nothing is trained or evaluated.
"""

from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

from onion_fl.data.contract import SubjectData
from onion_fl.data.stream import LabelsConfig, StreamConfig, edge_stream

INF = float("inf")


def rows(t, y=None, subject: str = "swell-1") -> SubjectData:
    n = len(t)
    return SubjectData(
        X=np.arange(n, dtype=np.float32).reshape(n, 1),
        y=np.zeros(n, np.int64) if y is None else np.asarray(y),
        dataset="swell",
        subject=subject,
        task="stress",
        n_classes=2,
        feature_names=["f0"],
        t=np.asarray(t, float),
    )


def stream(**overrides) -> StreamConfig:
    return StreamConfig(**({"bootstrap": 25, "round_every": 10} | overrides))


T = [0, 10, 20, 30, 40, 50, 60]


def test_durations_read_like_the_rest_of_the_config() -> None:
    config = StreamConfig(bootstrap="20m", round_every="15m", start={"staggered": "1h"})

    assert config.bootstrap == 1200 and config.round_every == 900
    assert config.start.staggered == 3600 and config.window is None
    assert LabelsConfig(delay="30m").delay == 1800


@pytest.mark.parametrize(
    "bad", [{"round_every": 0}, {"order": "random"}, {"batch_size": 0}, {"teleport": 1}]
)
def test_a_bad_stream_is_refused(bad: dict) -> None:
    with pytest.raises(ValidationError):
        stream(**bad)


def test_rows_before_the_bootstrap_end_are_history_there_from_the_start() -> None:
    s = edge_stream(rows(T), stream(batch_size=2), LabelsConfig(), seed=0)

    assert s.history.tolist() == [True] * 3 + [False] * 4
    # stream rows at t - 25, in batches of two available at their last row
    assert s.available_at.tolist() == [0, 0, 0, 15, 15, 35, 35]
    assert s.horizon == 35


def test_a_bootstrap_can_count_rows_instead() -> None:
    config = stream(bootstrap={"samples": 2}, batch_size=1)
    s = edge_stream(rows(T), config, LabelsConfig(), seed=0)

    assert s.history.tolist() == [True] * 2 + [False] * 5
    assert s.available_at[2:].tolist() == [0, 10, 20, 30, 40]  # from the third row


def test_rows_arrive_in_time_order() -> None:
    s = edge_stream(rows([30, 0, 10]), stream(bootstrap=5), LabelsConfig(), seed=0)

    assert s.data.t.tolist() == [0, 10, 30] and s.data.X[:, 0].tolist() == [1, 2, 0]


def test_exactly_the_fraction_is_labelled_by_the_seed() -> None:
    data = rows(np.arange(10) * 10.0)

    def mask(seed: int, subject: str = "swell-1") -> list[bool]:
        labels = LabelsConfig(fraction=0.2)
        s = edge_stream(rows(data.t, subject=subject), stream(), labels, seed=seed)
        return np.isfinite(s.label_at).tolist()

    assert sum(mask(0)) == 2 and mask(0) == mask(0)
    assert any(mask(s) != mask(0) for s in range(1, 6))
    assert any(mask(0, f"swell-{i}") != mask(0) for i in range(2, 7))


def test_without_labels_nothing_is_ever_trainable() -> None:
    s = edge_stream(rows(T), stream(), LabelsConfig(fraction=0.0), seed=0)

    assert not s.trainable(-INF, INF).any() and np.isinf(s.label_at).all()


def test_a_label_arrives_after_its_delay_and_only_then_trains() -> None:
    s = edge_stream(rows(T), stream(batch_size=1), LabelsConfig(delay=12), seed=0)

    # history: t - 25 + 12, so labels already due before the start are there
    assert s.label_at.tolist() == [-13, -3, 7, 17, 27, 37, 47]
    assert s.trainable_at.tolist() == [0, 0, 7, 17, 27, 37, 47]
    assert s.trainable(-INF, 0).tolist() == [True, True] + [False] * 5
    assert s.trainable(10, 20).tolist() == [False] * 3 + [True] + [False] * 3
    assert s.arrived(0, 10).tolist() == [False] * 3 + [True] + [False] * 3
    assert s.labelled(0, 10).tolist() == [False, False, True] + [False] * 4


def test_a_staggered_edge_starts_later() -> None:
    s = edge_stream(rows(T), stream(batch_size=1), LabelsConfig(), seed=0, offset=100)

    assert s.available_at.tolist() == [100, 100, 100, 105, 115, 125, 135]
    assert s.horizon == 135


def test_untimed_data_cannot_stream() -> None:
    from dataclasses import replace

    with pytest.raises(ValueError, match="time"):
        edge_stream(replace(rows(T), t=None), stream(), LabelsConfig(), seed=0)
