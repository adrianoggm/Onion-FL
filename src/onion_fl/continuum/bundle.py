from __future__ import annotations

"""The model bundle: one version of a federation, whole, on disk (spec §6).

    bundle/
    ├── bundle.json          # round, root id and node ids
    ├── model.npz            # the global model
    ├── server.npz           # the server optimizer's state
    ├── nodes/<id>.npz       # each node's arrays (edge model, trainer memory, zone, ...)
    ├── nodes/<id>.json      # each node's metadata (random stream, memory layout, ...)
    ├── preprocessing.json   # per dataset: kept features, fill, mean, std
    ├── schema.json          # per dataset: task, classes, features
    ├── lineage.json         # version, run_id, parent
    └── config.yaml          # the config that produced it

Only ``.npz`` and text files: a bundle never holds a pickle.
"""

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from onion_fl.roles.snapshot import FederationSnapshot, NodeState

ROOT_PARTS = {"state": "model.npz", "server": "server.npz"}


@dataclass
class Bundle:
    snapshot: FederationSnapshot
    preprocessing: dict[str, Any]
    schema: dict[str, Any]
    lineage: dict[str, Any]
    config: dict[str, Any]


def _save_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    np.savez(path, **{key: np.asarray(value) for key, value in arrays.items()})


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def save_bundle(
    path: str | Path,
    snapshot: FederationSnapshot,
    *,
    preprocessing: Mapping[str, Any],
    schema: Mapping[str, Any],
    lineage: Mapping[str, Any],
    config: Mapping[str, Any],
) -> Path:
    path = Path(path)
    (path / "nodes").mkdir(parents=True, exist_ok=True)
    root = snapshot.root
    for node_id, node in snapshot.nodes.items():
        rest = dict(node.arrays)
        if node_id == root:
            for part, name in ROOT_PARTS.items():
                head = f"{part}/"
                _save_npz(
                    path / name,
                    {
                        k[len(head) :]: rest.pop(k)
                        for k in list(rest)
                        if k.startswith(head)
                    },
                )
        _save_npz(path / "nodes" / f"{node_id}.npz", rest)
        _write_json(path / "nodes" / f"{node_id}.json", node.meta)
    _write_json(
        path / "bundle.json",
        {"round": snapshot.round, "root": root, "nodes": sorted(snapshot.nodes)},
    )
    _write_json(path / "preprocessing.json", dict(preprocessing))
    _write_json(path / "schema.json", dict(schema))
    _write_json(path / "lineage.json", dict(lineage))
    (path / "config.yaml").write_text(
        yaml.safe_dump(
            json.loads(json.dumps(dict(config), default=str)), sort_keys=True
        ),
        encoding="utf-8",
    )
    return path


def load_bundle(path: str | Path) -> Bundle:
    path = Path(path)
    manifest = json.loads((path / "bundle.json").read_text(encoding="utf-8"))
    nodes: dict[str, NodeState] = {}
    for node_id in manifest["nodes"]:
        arrays = _load_npz(path / "nodes" / f"{node_id}.npz")
        meta = json.loads(
            (path / "nodes" / f"{node_id}.json").read_text(encoding="utf-8")
        )
        if node_id == manifest["root"]:
            for part, name in ROOT_PARTS.items():
                arrays |= {f"{part}/{k}": v for k, v in _load_npz(path / name).items()}
        nodes[node_id] = NodeState(arrays, meta)

    def read(name: str) -> Any:
        return json.loads((path / name).read_text(encoding="utf-8"))

    return Bundle(
        snapshot=FederationSnapshot(
            round=int(manifest["round"]), nodes=nodes, root=manifest["root"]
        ),
        preprocessing=read("preprocessing.json"),
        schema=read("schema.json"),
        lineage=read("lineage.json"),
        config=yaml.safe_load((path / "config.yaml").read_text(encoding="utf-8")) or {},
    )
