from __future__ import annotations

"""What a node can do towards the outside world (spec §5.1).

Each runtime implements ``Context``; nodes only ever talk to it.
"""

import hashlib
from typing import Any, Protocol

import numpy as np

from onion_fl.core.message import Message


class Context(Protocol):
    """The only way a node acts outside itself."""

    node_id: str
    rng: np.random.Generator

    def send(self, msg: Message) -> None:
        """Deliver ``msg`` to ``msg.dst``. ``msg.src`` must be this node."""

    def set_timer(self, delay: float, name: str) -> None:
        """Call ``on_timer(name)`` after ``delay`` seconds; re-arming a name replaces it."""

    def cancel_timer(self, name: str) -> None:
        """Cancel the timer ``name`` if it is armed."""

    def now(self) -> float:
        """Current time in seconds: virtual in simulation, wall clock in real runs."""

    def emit(self, name: str, value: float | None = None, **tags: Any) -> None:
        """Record an instrumentation event tagged with this node."""

    def compute(self, samples: float) -> None:
        """Declare work done in this handler (e.g. training samples x epochs).

        The runtime turns it into busy time with the node's compute model; the
        messages sent in this handler leave when that time has passed.
        """


def node_rng(seed: int, node_id: str) -> np.random.Generator:
    """Random generator of one node, derived from the experiment seed and the node id.

    Uses SHA-256 of the id, never Python's ``hash()``, so every process gets
    the same stream for the same (seed, node).
    """
    digest = hashlib.sha256(node_id.encode("utf-8")).digest()
    words = np.frombuffer(digest[:16], dtype="<u4").tolist()
    return np.random.default_rng(np.random.SeedSequence([seed, *words]))
