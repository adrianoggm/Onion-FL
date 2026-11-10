"""Synthetic client partitions and the CIFAR reader (issue #247).

The partition tests use label arrays only: nothing is trained or evaluated.
The CIFAR batches here are tiny format fixtures written to ``tmp_path`` to
check parsing; the real batches are checked in test_datasets_descriptors.py.
"""

from __future__ import annotations

import os
import pickle
from pathlib import Path

import numpy as np
import pytest

from onion_fl.data.contract import DataError
from onion_fl.data.ingest import readers
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


# --- the CIFAR reader and the step on a format fixture ------------------------------


def write_batch(path: Path, labels: list[int], shade: int = 0) -> None:
    pixels = np.full((len(labels), 3072), shade, dtype=np.uint8)
    pixels[:, 0] = np.arange(len(labels))  # tells the rows apart
    batch = {
        b"batch_label": path.stem.encode(),
        b"labels": labels,
        b"data": pixels,
        b"filenames": [b"f%d.png" % i for i in range(len(labels))],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(pickle.dumps(batch, protocol=2))  # as the published batches


def test_the_reader_gives_pixels_in_0_1_a_label_and_the_batch(tmp_path: Path):
    write_batch(tmp_path / "data_batch_1", [3, 1, 4], shade=255)

    frame = readers.create("cifar_pickle").read(tmp_path / "data_batch_1")

    assert list(frame.columns[:2]) == ["x_0", "x_1"] and frame.shape == (3, 3074)
    assert frame["y"].tolist() == [3, 1, 4]
    assert set(frame["batch"]) == {"data_batch_1"}
    assert frame["x_1"].tolist() == [1.0, 1.0, 1.0]
    assert frame["x_0"].tolist() == pytest.approx([0, 1 / 255, 2 / 255])


class _Runs:
    def __reduce__(self):
        return (os.getcwd, ())


@pytest.mark.parametrize(
    "payload",
    [
        {b"data": _Runs(), b"labels": []},
        {b"data": np.array([object()], dtype=object), b"labels": [0]},
    ],
)
def test_the_reader_runs_no_code_from_the_file(tmp_path: Path, payload):
    path = tmp_path / "data_batch_1"
    path.write_bytes(pickle.dumps(payload, protocol=2))

    with pytest.raises(DataError, match="not allowed"):
        readers.create("cifar_pickle").read(path)


def test_the_reader_checks_the_shape(tmp_path: Path):
    path = tmp_path / "data_batch_1"
    path.write_bytes(
        pickle.dumps({b"data": np.zeros((2, 5), np.uint8), b"labels": [0, 1]}, 2)
    )

    with pytest.raises(DataError, match="shape"):
        readers.create("cifar_pickle").read(path)
