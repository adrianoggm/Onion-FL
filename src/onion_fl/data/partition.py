from __future__ import annotations

"""Synthetic client partitions of a centralised dataset, as NIID-Bench draws them.

A dataset without natural clients (CIFAR-10) is cut into ``clients`` parts by
its labels only (Li et al., "Federated Learning on Non-IID Data Silos: An
Experimental Study", ICDE 2022, the NIID-Bench ``partition_data``):

=================  ==============================================================
``iid``            a random permutation cut into equal parts
``dirichlet_label``  per class, proportions ~ Dir(α) over the clients (Hsu et al.,
                   2019); a client that already holds N / clients rows takes no
                   more, and the draw repeats until every client holds
                   ``min_size`` rows
``label_skew``     each client holds ``k`` distinct classes (client i always
                   class i mod K); each class is split equally among its holders
=================  ==============================================================

The draw depends on the labels, the scheme, its parameter and its own ``seed``
only, never on the training seed: every scenario compared on it sees the same
clients. It uses a numpy ``default_rng(seed)``, not NIID-Bench's global state,
so the parts are not NIID-Bench's own indices, only drawn the same way.
"""

from typing import Literal

import numpy as np

from onion_fl.data.contract import DataError

Scheme = Literal["iid", "dirichlet_label", "label_skew"]
MAX_DRAWS = 1000


def partition_labels(
    labels: np.ndarray,
    scheme: Scheme,
    clients: int,
    *,
    n_classes: int,
    seed: int,
    alpha: float | None = None,
    k: int | None = None,
    min_size: int = 10,
) -> list[np.ndarray]:
    """The row indices of each client, in the order each client keeps them."""
    y = np.asarray(labels)
    if y.ndim != 1:
        raise DataError("partition: labels must be one-dimensional")
    if clients < 1:
        raise DataError("partition: clients must be at least 1")
    if len(y) and (y.min() < 0 or y.max() >= n_classes):
        raise DataError(f"partition: labels outside 0..{n_classes - 1}")
    y = y.astype(np.int64)
    rng = np.random.default_rng(seed)
    if scheme == "iid":
        return [
            part.astype(np.int64)
            for part in np.array_split(rng.permutation(len(y)), clients)
        ]
    if scheme == "dirichlet_label":
        if alpha is None or alpha <= 0:
            raise DataError("partition: dirichlet_label needs alpha > 0")
        return _dirichlet(y, clients, n_classes, alpha, min_size, rng)
    if scheme == "label_skew":
        if k is None or not 1 <= k <= n_classes:
            raise DataError(f"partition: label_skew needs k in 1..{n_classes}")
        return _label_skew(y, clients, n_classes, k, rng)
    raise DataError(f"partition: unknown scheme {scheme!r}")


def _dirichlet(
    y: np.ndarray,
    clients: int,
    n_classes: int,
    alpha: float,
    min_size: int,
    rng: np.random.Generator,
) -> list[np.ndarray]:
    n = len(y)
    if clients * min_size > n:
        raise DataError(
            f"partition: {clients} clients of at least {min_size} rows need "
            f"{clients * min_size} rows, there are {n}"
        )
    for _ in range(MAX_DRAWS):
        parts: list[list[int]] = [[] for _ in range(clients)]
        for label in range(n_classes):
            rows = np.flatnonzero(y == label)
            rng.shuffle(rows)
            share = rng.dirichlet(np.repeat(alpha, clients))
            # A client already holding its fair share takes no more rows.
            share = share * np.array([len(p) < n / clients for p in parts])
            share = share / share.sum()
            cuts = (np.cumsum(share) * len(rows)).astype(int)[:-1]
            for part, chunk in zip(parts, np.split(rows, cuts), strict=True):
                part += chunk.tolist()
        if min(len(p) for p in parts) >= min_size:
            out = [np.asarray(p, dtype=np.int64) for p in parts]
            for part in out:
                rng.shuffle(part)
            return out
    raise DataError(
        f"partition: no Dir({alpha}) draw in {MAX_DRAWS} gave every client "
        f"{min_size} rows; raise alpha or lower min_size"
    )


def _label_skew(
    y: np.ndarray, clients: int, n_classes: int, k: int, rng: np.random.Generator
) -> list[np.ndarray]:
    held = []
    for i in range(clients):
        classes = [i % n_classes]
        while len(classes) < k:
            label = int(rng.integers(n_classes))
            if label not in classes:
                classes.append(label)
        held.append(classes)
    parts: list[list[int]] = [[] for _ in range(clients)]
    for label in range(n_classes):
        holders = [i for i in range(clients) if label in held[i]]
        rows = np.flatnonzero(y == label)
        rng.shuffle(rows)
        if not holders:  # a class nobody draws is left out, as in NIID-Bench
            continue
        for i, chunk in zip(holders, np.array_split(rows, len(holders)), strict=True):
            parts[i] += chunk.tolist()
    return [np.asarray(p, dtype=np.int64) for p in parts]
