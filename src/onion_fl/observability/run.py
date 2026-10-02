from __future__ import annotations

"""One execution on disk: ``runs/<run_id>/`` (spec §10.1).

=================  ==============================================================
``run.json``       identifiers, config, host, start and end, status, ``run_hash``
``events.jsonl``   every event, in the schema of ``observability.events``
``summary.json``   rounds, traffic, failures and the last scores at the root
=================  ==============================================================

``run_hash`` is the SHA-256 of ``run.json`` (without the hash itself) plus the
digests of the other two files, so any later change to them is detected.
"""

import json
import os
import socket
import time
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from onion_fl.core.ids import code_version, config_id, data_id, new_run_id, sha256
from onion_fl.core.topology import Topology
from onion_fl.observability.events import JsonlSink, kind_of


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="microseconds")


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, default=str) + "\n", encoding="utf-8")


def compute_run_hash(path: str | Path) -> str:
    path = Path(path)
    meta = json.loads((path / "run.json").read_text(encoding="utf-8"))
    meta.pop("run_hash", None)
    parts = [
        json.dumps(meta, sort_keys=True, default=str),
        sha256((path / "events.jsonl").read_bytes()),
        sha256((path / "summary.json").read_bytes()),
    ]
    return sha256("\n".join(parts))


def verify_run(path: str | Path) -> bool:
    """Do the files still match the ``run_hash`` written when the run finished?"""
    meta = json.loads((Path(path) / "run.json").read_text(encoding="utf-8"))
    return meta.get("run_hash") == compute_run_hash(path)


class _Summary:
    def __init__(self) -> None:
        self.rounds = 0
        self.finished = False
        self.sent = self.bytes = self.dropped = self.quorum_failed = 0
        self.final: dict[str, dict[str, Any]] = {}
        self.t_end = 0.0

    def add(self, event: Mapping[str, Any]) -> None:
        name, value, tags = event["name"], event["value"], event["tags"]
        self.t_end = max(self.t_end, event["t_virtual"] or 0.0)
        if name == "round.started":
            self.rounds = max(self.rounds, int(value))
        elif name == "run.finished":
            self.finished = True
        elif name == "message.sent":
            self.sent += 1
            self.bytes += int(value or 0)
        elif name in ("link.dropped", "message.dropped_offline"):
            self.dropped += 1
        elif name == "round.quorum_failed" and event["role"] == "coordinator":
            self.quorum_failed += 1
        elif name.startswith("eval.") and event["role"] == "coordinator":
            key = f"{tags.get('model')}/{tags.get('dataset', '*')}"
            scores = self.final.setdefault(key, {})
            if tags.get("round", 0) >= scores.get("round", 0):
                scores |= {"round": tags.get("round"), name[len("eval.") :]: value}

    def as_dict(self, wall_seconds: float) -> dict[str, Any]:
        return {
            "rounds": self.rounds,
            "finished": self.finished,
            "t_virtual_end": self.t_end,
            "wall_seconds": wall_seconds,
            "messages": {
                "sent": self.sent,
                "bytes": self.bytes,
                "dropped": self.dropped,
            },
            "quorum_failed": self.quorum_failed,
            "final": self.final,
        }


class Run:
    """Records one execution: create it, ``attach`` the federation, run it, ``finish``."""

    def __init__(
        self,
        runs_dir: str | Path,
        *,
        config: Mapping[str, Any],
        topology: Topology,
        seed: int,
        data_ids: Mapping[str, str] | None = None,
        scenario: str = "",
        sinks: Iterable[Any] = (),
    ) -> None:
        self.config_id = config_id(config)
        self.run_id = new_run_id(self.config_id, seed)
        self.path = Path(runs_dir) / self.run_id
        self.path.mkdir(parents=True)
        self.topology, self.seed, self.scenario = topology, seed, scenario
        self.meta: dict[str, Any] = {
            "run_id": self.run_id,
            "topology_id": topology.topology_id,
            "config_id": self.config_id,
            "data_id": data_id(data_ids or {}),
            "code_version": code_version(),
            "seed": seed,
            "scenario": scenario,
            "host": socket.gethostname(),
            "pid": os.getpid(),
            "started_at": _now(),
            "status": "running",
            "config": dict(config),
        }
        _write_json(self.path / "run.json", self.meta)
        self.sinks = [JsonlSink(self.path / "events.jsonl"), *sinks]
        self.nodes: dict[str, tuple[str | None, str | None]] = {
            "experiment": (None, "experiment")
        }
        self.summary = _Summary()
        self._wall = time.perf_counter()
        self._last_t = 0.0

    def attach(self, federation: Any) -> None:
        """Learn each node's level and role and start receiving the runtime's events."""
        levels = self.topology.levels
        coordinator = federation.coordinator
        self.nodes[coordinator.id] = (coordinator.level, "coordinator")
        for node in federation.aggregators.values():
            self.nodes[node.id] = (node.level, "aggregator")
        for node in federation.edges.values():
            self.nodes[node.id] = (levels[-1], "edge" if node.train else "evaluator")
        federation.runtime.listeners.append(self.on_event)

    def on_event(self, raw: Mapping[str, Any]) -> None:
        level, role = self.nodes.get(raw["node"], (None, None))
        tags = dict(raw.get("tags") or {})
        self._last_t = raw["t"]
        event = {
            "t_virtual": raw["t"],
            "t_wall": time.time(),
            "run_id": self.run_id,
            "topology_id": self.topology.topology_id,
            "scenario": self.scenario,
            "seed": self.seed,
            "round": tags.get("round"),
            "level": level,
            "node": raw["node"],
            "role": role,
            "kind": kind_of(raw["name"]),
            "name": raw["name"],
            "value": raw.get("value"),
            "tags": tags,
        }
        for sink in self.sinks:
            sink.write(event)
        self.summary.add(event)

    def record(self, name: str, value: Any = None, **tags: Any) -> None:
        """An event of the experiment itself (placement composition, data cards...)."""
        self.on_event(
            {
                "t": self._last_t,
                "node": "experiment",
                "name": name,
                "value": value,
                "tags": tags,
            }
        )

    def finish(self, status: str = "finished") -> dict[str, Any]:
        for sink in self.sinks:
            sink.close()
        _write_json(
            self.path / "summary.json",
            self.summary.as_dict(time.perf_counter() - self._wall),
        )
        self.meta |= {"finished_at": _now(), "status": status}
        _write_json(self.path / "run.json", self.meta)
        self.meta["run_hash"] = compute_run_hash(self.path)
        _write_json(self.path / "run.json", self.meta)
        return self.meta
