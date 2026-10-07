"""Replay memories: which rows an edge keeps (continuum C4, issue #159).

Rows are hand-written indices with hand-written labels; nothing is trained
(docs/RULES.md).
"""

from __future__ import annotations

import numpy as np
import pytest

from onion_fl.continuum.memory import memories


def memory(name: str, capacity: int = 4, seed: int = 0):
    kept = memories.create(name, {} if name == "none" else {"capacity": capacity})
    kept.rng = np.random.default_rng(seed)
    return kept


def offer(kept, rows, y=None) -> None:
    rows = np.asarray(rows)
    kept.add(rows, np.zeros(len(rows), int) if y is None else np.asarray(y))


@pytest.mark.parametrize("name", ["fifo", "reservoir", "class_balanced"])
def test_a_memory_never_holds_more_than_its_capacity(name: str) -> None:
    kept = memory(name, capacity=16)
    for start in range(0, 1000, 37):
        rows = np.arange(start, min(start + 37, 1000))
        offer(kept, rows, rows % 3)
        assert len(kept.rows()) <= 16
    assert len(kept.rows()) == 16
    assert len(set(kept.rows().tolist())) == 16  # never the same row twice


def test_none_keeps_nothing() -> None:
    kept = memory("none")
    offer(kept, range(10))

    assert kept.rows().tolist() == [] and kept.sample(3).tolist() == []


def test_fifo_keeps_the_latest_rows() -> None:
    kept = memory("fifo")
    offer(kept, range(6))
    offer(kept, range(6, 10))

    assert kept.rows().tolist() == [6, 7, 8, 9]


def test_a_reservoir_keeps_a_uniform_sample_of_what_it_was_offered() -> None:
    counts = np.zeros(100)
    for seed in range(200):
        kept = memory("reservoir", capacity=10, seed=seed)
        for start in range(0, 100, 7):
            offer(kept, range(start, min(start + 7, 100)))
        counts[kept.rows()] += 1

    # each row is kept with probability 10/100: 20 times in 200 seeds on average
    assert counts.sum() == 2000 and 6 <= counts.min() and counts.max() <= 38
    again = memory("reservoir", capacity=10, seed=3)
    first = memory("reservoir", capacity=10, seed=3)
    offer(again, range(100))
    offer(first, range(100))
    assert again.rows().tolist() == first.rows().tolist()


def test_class_balanced_drops_the_oldest_row_of_the_largest_class() -> None:
    kept = memory("class_balanced", capacity=6)
    offer(kept, range(10), [0] * 8 + [1] * 2)

    assert sorted(kept.rows().tolist()) == [4, 5, 6, 7, 8, 9]


def test_a_sample_is_drawn_without_replacement() -> None:
    kept = memory("fifo", capacity=10)
    offer(kept, range(10))

    sample = kept.sample(5)
    assert len(set(sample.tolist())) == 5 and set(sample) <= set(range(10))
    assert sorted(kept.sample(20).tolist()) == list(range(10))  # at most all


@pytest.mark.parametrize("name", ["fifo", "reservoir", "class_balanced"])
def test_a_saved_memory_goes_on_as_if_it_never_stopped(name: str) -> None:
    straight, first = memory(name, capacity=8), memory(name, capacity=8)
    offer(straight, range(50), np.arange(50) % 2)
    offer(first, range(50), np.arange(50) % 2)
    second = memory(name, capacity=8, seed=99)
    second.load_state(*first.state())

    offer(straight, range(50, 90), np.arange(50, 90) % 2)
    offer(second, range(50, 90), np.arange(50, 90) % 2)
    assert second.rows().tolist() == straight.rows().tolist()
    assert second.sample(3).tolist() == straight.sample(3).tolist()


def plain_reservoir(kept, seen, rows, rng, capacity):
    """Algorithm R, one draw per row offered once the memory is full."""
    kept = list(kept)
    for row in rows:
        seen += 1
        if len(kept) < capacity:
            kept.append(row)
        else:
            j = int(rng.integers(seen))
            if j < capacity:
                kept[j] = row
    return kept, seen


def plain_class_balanced(kept, labels, rows, y, capacity):
    """Append, then drop the oldest row of the largest class (lowest on a tie)."""
    kept, labels = list(kept), list(labels)
    for row, label in zip(rows, y, strict=True):
        kept.append(row)
        labels.append(int(label))
        if len(kept) > capacity:
            oldest = labels.index(int(np.bincount(labels).argmax()))
            del kept[oldest], labels[oldest]
    return kept, labels


@pytest.mark.parametrize("name", ["reservoir", "class_balanced"])
def test_the_memories_keep_what_the_plain_algorithms_keep(name: str) -> None:
    data = np.random.default_rng(3)
    kept, rng = memory(name, capacity=40, seed=11), np.random.default_rng(11)
    rows, labels, seen, start = [], [], 0, 0
    for size in data.integers(0, 60, size=12):
        batch = np.arange(start, start + size)
        y = data.integers(0, 3, size=size)
        start += size
        offer(kept, batch, y)
        if name == "reservoir":
            rows, seen = plain_reservoir(rows, seen, batch.tolist(), rng, 40)
        else:
            rows, labels = plain_class_balanced(rows, labels, batch, y, 40)
        assert kept.rows().tolist() == rows  # the same rows, in the same order
    assert kept.rng.random() == rng.random()  # and the same draws
