from __future__ import annotations

"""How a continuous federation keeps time (spec §10).

Kept free of imports from the rest of the package, so the roles can use it.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class View:
    """What a trigger sees at its level: data time now and at the last round (or
    update), the rows that became trainable since, and the drift detected since,
    by kind."""

    now: float
    since: float
    volume: float
    drift: Mapping[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class Continuum:
    """A continuous federation's pace, in virtual seconds (``speed`` gives data
    seconds per virtual second)."""

    trigger: Any  # federated: when the coordinator opens a round
    edge_trigger: Any | None  # local: whether an edge has an update for a round
    status_every: float  # virtual seconds between status messages
    speed: float
    until: float  # virtual time of the last label: the final round
    drift: Mapping[str, Any]  # kind -> detector spec, for every node
