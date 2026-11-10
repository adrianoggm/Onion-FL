from __future__ import annotations

"""The Studio's view of a run's versions, from its events (continuum §11, #203).

- **Versions and lineage.** Each version with its parent, the round that made
  it, its digest and its status, replayed from the ``version.*`` events the way
  the coordinator keeps them (``continuum.versions.Version``).
- **Metrics per version.** The scores of the decision on it (recent window and
  reference, as challenger), and the prequential scores of the rows it served:
  the rows that arrive before round ``r`` are predicted with the model of round
  ``r - 1``, whose version ``round.started`` names.
- **Per edge.** The rows that arrived at each edge's stream, how many came
  labelled, and the drifts it detected.
"""

from collections.abc import Iterable, Mapping
from typing import Any

# A version event's tags that are versions or rounds, not scores.
_NOT_SCORES = ("round", "superseded", "champion", "previous", "parent", "reason")


def _weighted(acc: dict[str, list[float]]) -> dict[str, float]:
    return {k: total / n for k, (total, n) in sorted(acc.items()) if n > 0}


def continuum_view(events: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    versions: dict[int, dict[str, Any]] = {}
    served: dict[int, int] = {}  # round -> the version it served
    champion: int | None = None
    decisions: list[dict[str, Any]] = []
    prequential: list[tuple[int, str, float, float]] = []
    edges: dict[str, dict[str, Any]] = {}

    def version(v: int) -> dict[str, Any]:
        if v not in versions:  # the initial model, or one only named so far
            versions[v] = {"v": v, "parent": None, "round": 0, "digest": None}
            versions[v] |= {"status": "active" if v == 0 else "candidate"}
            versions[v] |= {"replaced": None, "decision": None, "scores": {}}
        return versions[v]

    def edge_of(event: Mapping[str, Any]) -> dict[str, Any]:
        node, tags = event["node"], event.get("tags") or {}
        if node not in edges:
            edges[node] = {"node": node, "dataset": tags.get("dataset")}
            edges[node] |= {"arrived": 0.0, "labelled": 0, "drift": {}}
        return edges[node]

    for event in events:
        name, tags, value = event["name"], event.get("tags") or {}, event.get("value")
        if name == "round.started" and "version" in tags:
            served[int(tags["round"])] = int(tags["version"])
            if champion is None:
                champion = int(tags["version"])
                version(champion)
        elif name.startswith("version."):
            kind, v = name.split(".", 1)[1], int(value)
            scores = {
                k: x
                for k, x in tags.items()
                if k not in _NOT_SCORES and isinstance(x, int | float)
            }
            by = "operator" if tags.get("reason") == "operator" else "policy"
            decisions.append(
                {"t": event.get("t_virtual"), "event": kind, "version": v}
                | {"round": tags.get("round"), "reason": tags.get("reason")}
                | {"by": by, "scores": scores}
            )
            if kind == "candidate":
                parent = tags.get("parent")
                version(v).update(
                    parent=parent, round=tags.get("round"), digest=tags.get("digest")
                )
                if parent is not None:
                    version(int(parent))
                continue
            target = version(v)
            if kind == "rejected":
                target["status"] = "rejected"
            elif kind == "promoted":
                before = tags.get("superseded")
                if before is not None:
                    version(int(before))["status"] = "superseded"
                target |= {"status": "active", "replaced": before}
                champion = v
            elif kind == "rollback":
                before = tags.get("previous")
                if before is not None:
                    version(int(before))["status"] = "rolled_back"
                target["status"] = "active"
                champion = v
            if by == "policy":  # what it scored as a challenger
                target |= {"decision": tags.get("reason"), "scores": scores}
        elif name.startswith("eval.") and tags.get("model") == "prequential":
            samples = float(tags.get("samples") or 0)
            if event.get("round") is not None and isinstance(value, int | float):
                metric = name[len("eval.") :]
                prequential.append((int(event["round"]), metric, value, samples))
        elif name == "data.arrived":
            edge = edge_of(event)
            edge["arrived"] += float(value or 0)
            edge["labelled"] += int(tags.get("labelled") or 0)
        elif name == "drift.detected" and event.get("role") == "edge":
            drift = edge_of(event)["drift"]
            kind = str(tags.get("kind"))
            drift[kind] = drift.get(kind, 0) + 1

    per_version: dict[int, dict[str, list[float]]] = {}
    for round, metric, score, samples in prequential:
        v = served.get(round - 1)
        if v is None or samples <= 0:
            continue
        acc = per_version.setdefault(v, {}).setdefault(metric, [0.0, 0.0])
        acc[0] += score * samples
        acc[1] += samples
    for v, item in versions.items():
        item["prequential"] = _weighted(per_version.get(v, {}))
        item["served"] = sorted(r for r, s in served.items() if s == v)
    return {
        "champion": champion,
        "versions": [versions[v] for v in sorted(versions)],
        "decisions": decisions,
        "rounds": [{"round": r, "version": v} for r, v in sorted(served.items())],
        "edges": [edges[n] for n in sorted(edges)],
    }
