from __future__ import annotations

"""Replay memories: which of its rows an edge keeps to train on again (spec §8).

A memory keeps indices into its edge's own stream, never copies of the data,
and only rows a training already consumed successfully (continuum C4). Each
memory has its own random stream, which the runner seeds per edge; ``state``
holds everything needed to go on as if it never stopped.
"""

from collections import deque
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
    def __init__(self, capacity: int) -> None:
        super().__init__(capacity)
        self.seen = 0

    def add(self, rows: np.ndarray, y: np.ndarray) -> None:
        rows = np.asarray(rows, np.int64)
        room = max(self.capacity - len(self.kept), 0)
        kept = np.concatenate([self.kept, rows[:room]])
        rest, seen = rows[room:], self.seen + room
        if len(rest):  # the i-th row past a full memory replaces slot j < capacity
            # One draw per row, j in [0, seen_i), as algorithm R draws them one by one.
            j = self.rng.integers(np.arange(seen + 1, seen + len(rest) + 1))
            hit = j < self.capacity
            slots, values = j[hit][::-1], rest[hit][::-1]  # the last write wins
            _, last = np.unique(slots, return_index=True)
            kept[slots[last]] = values[last]
        self.seen += len(rows)
        self.kept = kept

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
    def __init__(self, capacity: int) -> None:
        super().__init__(capacity)
        self.labels = np.zeros(0, np.int64)

    def add(self, rows: np.ndarray, y: np.ndarray) -> None:
        # One queue per class, oldest first, of (arrival order, row): dropping the
        # oldest row of the largest class is a popleft, not a scan of the memory.
        queues: dict[int, deque] = {}
        old = zip(self.kept.tolist(), self.labels.tolist(), strict=True)
        for order, (row, label) in enumerate(old):
            queues.setdefault(label, deque()).append((order, row))
        order = size = len(self.kept)
        new = zip(
            np.asarray(rows, np.int64).tolist(), np.asarray(y).tolist(), strict=True
        )
        for row, label in new:
            queues.setdefault(int(label), deque()).append((order, row))
            order, size = order + 1, size + 1
            if size > self.capacity:  # the lowest class on a tie
                most = max(len(q) for q in queues.values())
                largest = min(c for c, q in queues.items() if len(q) == most)
                queues[largest].popleft()
                size -= 1
        merged = sorted((o, r, c) for c, q in queues.items() for o, r in q)
        self.kept = np.array([r for _, r, _ in merged], np.int64)
        self.labels = np.array([c for _, _, c in merged], np.int64)

    def state(self) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        arrays, meta = super().state()
        return arrays | {"labels": self.labels}, meta

    def load_state(self, arrays: dict[str, np.ndarray], meta: dict[str, Any]) -> None:
        super().load_state(arrays, meta)
        self.labels = np.asarray(arrays["labels"], np.int64).copy()
