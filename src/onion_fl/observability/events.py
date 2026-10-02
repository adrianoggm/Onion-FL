from __future__ import annotations

"""The event schema and the JSONL sink (spec §10.2, §10.5).

::

    {t_virtual, t_wall, run_id, topology_id, scenario, seed, round,
     level, node, role, kind, name, value, tags}

``kind`` comes from the event name; ``level`` and ``role`` from the federation.
``events.jsonl`` is the source of truth that every other view is built from.
"""

import hashlib
import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA = (
    "t_virtual",
    "t_wall",
    "run_id",
    "topology_id",
    "scenario",
    "seed",
    "round",
    "level",
    "node",
    "role",
    "kind",
    "name",
    "value",
    "tags",
)
KINDS = ("metric", "message", "lifecycle", "diagnostic", "data")
_PREFIXES = (
    (("message.", "link."), "message"),
    (("eval.", "metric.", "round.train_loss", "edge.trained"), "metric"),
    (("diagnostic.",), "diagnostic"),
    (("data.",), "data"),
)
_VOLATILE = ("t_wall", "run_id")  # differ between two runs of the same config


def kind_of(name: str) -> str:
    for prefixes, kind in _PREFIXES:
        if name.startswith(prefixes):
            return kind
    return "lifecycle"


def _plain(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def dumps(event: Mapping[str, Any]) -> str:
    return json.dumps(event, default=_plain, separators=(",", ":"))


class JsonlSink:
    """One JSON event per line."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._file = self.path.open("a", encoding="utf-8", newline="\n")

    def write(self, event: Mapping[str, Any]) -> None:
        self._file.write(dumps(event) + "\n")

    def close(self) -> None:
        self._file.close()


def read_events(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def event_fingerprint(events: str | Path | Iterable[Mapping[str, Any]]) -> str:
    """SHA-256 of the events without ``t_wall`` and ``run_id``: equal for equal runs."""
    if isinstance(events, str | Path):
        events = read_events(events)
    digest = hashlib.sha256()
    for event in events:
        stable = {k: v for k, v in event.items() if k not in _VOLATILE}
        digest.update(json.dumps(stable, sort_keys=True, default=_plain).encode())
        digest.update(b"\n")
    return digest.hexdigest()
