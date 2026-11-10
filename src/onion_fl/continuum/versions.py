from __future__ import annotations

"""Versions of the global model: champion and challenger (spec §5, §11).

The champion is the model the edges serve. Every round that aggregates
produces a challenger, which the coordinator has the validation streams score
next to the champion, on two sets of the labels that came:

- the **recent** window (plasticity): those since the champion was promoted,
  or within the last ``window`` of data time;
- the **reference** (retention): every earlier one, the history included.

A ``promotions`` plugin (``continuum.promotion``) decides from those scores.
The test subjects never decide. Kept free of imports from the roles, so they
can use it.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from onion_fl.core.ids import sha256

MODELS = ("champion", "challenger")
PARTS = ("recent", "reference")
# Scores of one model on one part: the metrics and their samples.
Scores = Mapping[str, Mapping[str, Mapping[str, float]]]


def digest(state: Mapping[str, np.ndarray]) -> str:
    """SHA-256 of a model state, independent of key order."""
    parts = []
    for key in sorted(state):
        array = np.ascontiguousarray(state[key])
        parts.append(f"{key}:{array.dtype.str}:{array.shape}".encode())
        parts.append(array.tobytes())
    return sha256(b"".join(parts))


@dataclass
class Version:
    """One version of the global model and the server state that goes with it.

    ``status`` is ``candidate`` while it is validated, then ``rejected`` or
    ``active``; an active one becomes ``superseded`` when another is promoted,
    and ``rolled_back`` when a rollback leaves it. ``replaced`` is the champion
    it superseded, where a rollback returns to.
    """

    v: int
    parent: int | None  # the champion it was trained from
    round: int  # the round that produced it (0: the initial model)
    state: dict[str, np.ndarray]
    server: dict[str, np.ndarray] = field(default_factory=dict)
    status: str = "candidate"
    replaced: int | None = None

    @property
    def digest(self) -> str:
        return digest(self.state)

    def record(self) -> dict[str, Any]:
        """What the lineage keeps of it: everything but the arrays."""
        return {
            "v": self.v,
            "parent": self.parent,
            "round": self.round,
            "status": self.status,
            "replaced": self.replaced,
            "digest": self.digest,
        }


@dataclass(frozen=True)
class Decision:
    promote: bool
    reason: str  # promoted, or the first check that failed
    measures: dict[str, float]
