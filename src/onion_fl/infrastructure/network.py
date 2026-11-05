from __future__ import annotations

"""The physical network during a run: segment states over time and their queues
(I1 §8).

A segment (a site's LAN or a link between sites) changes as
``network.schedule`` says. With ``network.bandwidth: shared`` each direction of
a segment is one FIFO queue for every logical link that crosses it: a message
waits until all the segments of its path are free, holds each for its size over
that segment's bandwidth, and is through after its size over the narrowest one.
Nothing contends, and a shared path behaves as a dedicated one.

Whether a message is lost to a segment being down is decided when it is sent.
ponytail: a message already in flight when its segment goes down still
arrives; check arrivals against the schedule if that ever matters.
"""

import csv
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from onion_fl.infrastructure.model import Infrastructure, InfrastructureError, patch
from onion_fl.infrastructure.paths import Hop, combine, segment


def _key(between: Sequence[str]) -> str:
    return f"link:{between[0]}|{between[1]}"


def changes(infra: Infrastructure) -> list[tuple[float, str, dict | None, bool | None]]:
    """Every change in time order: (at, segment, fields set, down), traces expanded."""
    out = []
    links = {frozenset(link.between): link.between for link in infra.network.links}
    for change in infra.network.schedule:
        key = _key(links[frozenset(change.between)])
        if change.trace is None:
            out.append((change.at, key, change.set, change.down))
            continue
        path = Path(change.trace)
        if not path.exists():
            raise InfrastructureError(f"network trace not found: {path}")
        with path.open(encoding="utf-8", newline="") as rows:
            for i, row in enumerate(csv.DictReader(rows)):
                fields = {
                    k: v for k, v in row.items() if k != "t" and v not in ("", None)
                }
                try:
                    at = float(row["t"]) + change.at
                    out.append((at, key, patch(fields), None))
                except (KeyError, ValueError):  # no file content in the message
                    raise InfrastructureError(
                        f"{path.name} row {i + 1}: needs a numeric t and link fields "
                        "with units (latency, jitter, bandwidth, loss)"
                    ) from None
    return sorted(out, key=lambda c: c[0])  # stable: same instant, declared order


class PhysicalNetwork:
    """Segment states and, when the bandwidth is shared, when each is free."""

    def __init__(self, infra: Infrastructure) -> None:
        self.infra = infra
        self.shared = infra.network.bandwidth == "shared"
        self.changes = changes(infra)
        self._free: dict[Hop, float] = {}

    def state(self, key: str, t: float) -> tuple[Any, bool]:
        """A segment's fields at ``t`` and whether it is up."""
        update: dict[str, Any] = {}
        up = True
        for at, changed, fields, down in self.changes:
            if at > t:
                break
            if changed == key:
                if down is None:
                    update |= fields
                else:
                    up = not down
        base = segment(self.infra, key)
        return (base.model_copy(update=update) if update else base), up

    def profile(self, hops: Sequence[Hop], t: float) -> dict[str, Any] | None:
        """The profile of a path at ``t``, ``up`` along it; None if a segment is down."""
        states = [(self.state(key, t), forward) for key, forward in hops]
        if not all(up for (_, up), _ in states):
            return None
        return combine([(fields, forward) for (fields, _), forward in states])

    def reserve(
        self, hops: Sequence[Hop], t: float, size_bytes: int
    ) -> tuple[float, float]:
        """Queue a message on every segment of its path: when it starts and when
        it is through. Hops are in the direction of travel."""
        durations = []
        for key, forward in hops:
            fields, _ = self.state(key, t)
            rate = fields.bandwidth_up if forward else fields.bandwidth_down
            durations.append(0.0 if rate is None else size_bytes * 8 / rate)
        start = max([t, *(self._free.get(hop, 0.0) for hop in hops)])
        for hop, duration in zip(hops, durations, strict=True):
            self._free[hop] = start + duration
        return start, start + max(durations, default=0.0)

    def events(self) -> list[tuple[float, str, dict[str, Any]]]:
        """``link.down``, ``link.up`` and ``link.changed`` with their instants."""
        out = []
        for at, key, fields, down in self.changes:
            if down is None:
                out.append((at, "link.changed", {"link": key, **fields}))
            else:
                out.append((at, "link.down" if down else "link.up", {"link": key}))
        return out


def travel(hops: Sequence[Hop], direction: str) -> list[Hop]:
    """The hops of a logical link (listed child -> parent) in one direction."""
    return list(hops) if direction == "up" else [(k, not f) for k, f in reversed(hops)]
