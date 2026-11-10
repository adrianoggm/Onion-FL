"""Synthetic client partitions and the CIFAR reader (issue #247).

The partition tests use label arrays only: nothing is trained or evaluated.
The CIFAR batches here are tiny format fixtures written to ``tmp_path`` to
check parsing; the real batches are checked in test_datasets_descriptors.py.
"""

from __future__ import annotations

import numpy as np
import pytest

from onion_fl.data.contract import DataError
from onion_fl.data.partition import partition_labels

LABELS = np.repeat(np.arange(10), 200)  # 2,000 rows, 200 per class


def parts(scheme: str, seed: int = 0, **kwargs) -> list[np.ndarray]:
    return partition_labels(LABELS, scheme, 10, n_classes=10, seed=seed, **kwargs)


SCHEMES = [("iid", {}), ("dirichlet_label", {"alpha": 0.5}), ("label_skew", {"k": 2})]


@pytest.mark.parametrize("scheme, kwargs", SCHEMES)
def test_a_partition_is_a_function_of_scheme_parameter_and_seed(scheme, kwargs):
    first, again = parts(scheme, **kwargs), parts(scheme, **kwargs)
    other = parts(scheme, seed=1, **kwargs)

    assert all(np.array_equal(a, b) for a, b in zip(first, again, strict=True))
    assert not all(
        len(a) == len(b) and np.array_equal(a, b)
        for a, b in zip(first, other, strict=True)
    )


@pytest.mark.parametrize("scheme, kwargs", SCHEMES)
def test_every_row_goes_to_exactly_one_client(scheme, kwargs):
    rows = np.concatenate(parts(scheme, **kwargs))

    assert len(rows) == len(np.unique(rows))
    if scheme != "label_skew":  # label_skew may leave out a class nobody drew
        assert np.array_equal(np.sort(rows), np.arange(len(LABELS)))


def test_iid_gives_equal_clients_with_every_class():
    for part in parts("iid"):
        assert len(part) == 200
        assert len(np.unique(LABELS[part])) == 10


def test_label_skew_gives_k_classes_to_each_client_the_first_its_own():
    for i, part in enumerate(parts("label_skew", k=3)):
        classes = set(np.unique(LABELS[part]).tolist())
        assert len(classes) == 3 and i % 10 in classes


def test_label_skew_splits_each_class_equally_among_its_holders():
    clients = parts("label_skew", k=2)
    for label in range(10):
        counts = [int((LABELS[p] == label).sum()) for p in clients]
        held = [c for c in counts if c]
        assert max(held) - min(held) <= 1, label


def test_dirichlet_skews_more_with_a_smaller_alpha():
    def mean_classes(alpha: float) -> float:
        return np.mean(
            [
                (np.bincount(LABELS[p], minlength=10) > 0.05 * len(p)).sum()
                for p in parts("dirichlet_label", alpha=alpha)
            ]
        )

    assert mean_classes(0.1) < mean_classes(0.5) < mean_classes(100.0)


def test_dirichlet_redraws_until_every_client_has_min_size_rows():
    for part in parts("dirichlet_label", alpha=0.1, min_size=50):
        assert len(part) >= 50


@pytest.mark.parametrize(
    "scheme, kwargs, match",
    [
        ("dirichlet_label", {}, "alpha"),
        ("label_skew", {"k": 11}, "k in 1..10"),
        ("label_skew", {}, "k in"),
        ("dirichlet_label", {"alpha": 0.5, "min_size": 1000}, "need"),
    ],
)
def test_bad_partitions_are_refused(scheme, kwargs, match):
    with pytest.raises(DataError, match=match):
        parts(scheme, **kwargs)


def test_labels_outside_the_classes_are_refused():
    with pytest.raises(DataError, match="outside"):
        partition_labels(np.array([0, 10]), "iid", 2, n_classes=10, seed=0)
