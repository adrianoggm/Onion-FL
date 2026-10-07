from __future__ import annotations

"""Replay memories: which of its rows an edge keeps to train on again (spec §8).

A memory keeps indices into its edge's own stream, never copies of the data,
and only rows a training already consumed successfully (continuum C4). Each
memory has its own random stream, which the runner seeds per edge; ``state``
holds everything needed to go on as if it never stopped.
"""

from typing import Any, Protocol, runtime_checkable

import numpy as np
from pydantic import BaseModel, Field, PositiveInt

from onion_fl.core.context import restore_rng, rng_state
from onion_fl.core.registry import Registry

memories = Registry("memory")


@runtime_checkable
class Memory(Protocol):
    """What an edge uses of a replay memory; every ``memories`` plugin offers it.

    The runner sets ``rng`` to the edge's own stream; ``state`` and
    ``load_state`` go into and come out of the run's bundle.
    """

    capacity: int
    rng: np.random.Generator

    def add(self, rows: np.ndarray, y: np.ndarray) -> None:
        """Offer rows just consumed, in time order, with their labels."""

    def rows(self) -> np.ndarray:
        """The kept rows, as indices into the edge's stream."""

    def sample(self, k: int) -> np.ndarray:
        """Up to ``k`` kept rows, without replacement."""

    def state(self) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        """Arrays and metadata to go on as if the memory never stopped."""

    def load_state(self, arrays: dict[str, np.ndarray], meta: dict[str, Any]) -> None:
        """Go on from what ``state`` returned."""


class CapacityParams(BaseModel):
    capacity: PositiveInt = Field(512, description="Filas que guarda como mucho")


class NoParams(BaseModel):
    pass


class _Memory:
    capacity = 0

    def __init__(self, capacity: int = 0) -> None:
        self.capacity = capacity
        self.kept = np.zeros(0, np.int64)
        self.rng = np.random.default_rng(0)  # the runner gives each edge its own

    def add(self, rows: np.ndarray, y: np.ndarray) -> None:
        raise NotImplementedError

    def rows(self) -> np.ndarray:
        return self.kept

    def sample(self, k: int) -> np.ndarray:
        """Up to ``k`` kept rows, without replacement."""
        k = min(int(k), len(self.kept))
        return self.rng.choice(self.kept, size=k, replace=False) if k else self.kept[:0]

    def state(self) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        return {"rows": self.kept}, {"rng": rng_state(self.rng)}

    def load_state(self, arrays: dict[str, np.ndarray], meta: dict[str, Any]) -> None:
        self.kept = np.asarray(arrays["rows"], np.int64).copy()
        restore_rng(self.rng, meta["rng"])


@memories.register(
    "none",
    title="Sin memoria",
    description="No guarda nada: cada ronda entrena solo con lo reciente.",
    params=NoParams,
)
class NoMemory(_Memory):
    def add(self, rows: np.ndarray, y: np.ndarray) -> None:
        pass


@memories.register(
    "fifo",
    title="FIFO",
    description="Guarda las últimas filas entrenadas, hasta la capacidad.",
    params=CapacityParams,
)
class Fifo(_Memory):
    def add(self, rows: np.ndarray, y: np.ndarray) -> None:
        self.kept = np.concatenate([self.kept, rows]).astype(np.int64)[-self.capacity :]


@memories.register(
    "reservoir",
    title="Reservorio",
    description="Una muestra uniforme de todo lo entrenado (algoritmo R).",
    params=CapacityParams,
    explain="Cada fila ofrecida tiene la misma probabilidad de seguir guardada.",
)
class Reservoir(_Memory):
    def __init__(self, capacity: int = 512) -> None:
        super().__init__(capacity)
        self.seen = 0

    def add(self, rows: np.ndarray, y: np.ndarray) -> None:
        kept = list(self.kept)
        for row in np.asarray(rows, np.int64):
            self.seen += 1
            if len(kept) < self.capacity:
                kept.append(row)
            else:
                j = int(self.rng.integers(self.seen))
                if j < self.capacity:
                    kept[j] = row
        self.kept = np.asarray(kept, np.int64)

    def state(self) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        arrays, meta = super().state()
        return arrays, meta | {"seen": self.seen}

    def load_state(self, arrays: dict[str, np.ndarray], meta: dict[str, Any]) -> None:
        super().load_state(arrays, meta)
        self.seen = int(meta["seen"])


@memories.register(
    "class_balanced",
    title="Equilibrada por clase",
    description="Al llenarse, descarta la fila más antigua de la clase con más filas.",
    params=CapacityParams,
)
class ClassBalanced(_Memory):
    def __init__(self, capacity: int = 512) -> None:
        super().__init__(capacity)
        self.labels = np.zeros(0, np.int64)

    def add(self, rows: np.ndarray, y: np.ndarray) -> None:
        kept, labels = list(self.kept), list(self.labels)
        for row, label in zip(np.asarray(rows, np.int64), np.asarray(y), strict=True):
            kept.append(row)
            labels.append(int(label))
            if len(kept) > self.capacity:
                largest = int(np.bincount(labels).argmax())  # the lowest on a tie
                oldest = labels.index(largest)  # rows come in time order
                del kept[oldest], labels[oldest]
        self.kept = np.asarray(kept, np.int64)
        self.labels = np.asarray(labels, np.int64)

    def state(self) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        arrays, meta = super().state()
        return arrays | {"labels": self.labels}, meta

    def load_state(self, arrays: dict[str, np.ndarray], meta: dict[str, Any]) -> None:
        super().load_state(arrays, meta)
        self.labels = np.asarray(arrays["labels"], np.int64).copy()
