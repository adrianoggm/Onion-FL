from __future__ import annotations

"""Topology: the validated tree of nodes and links (spec §5.1, §10.1, §11, §12.1).

A topology lists the coordinator (root) and the aggregators of every level.
Edges are not listed: placement creates them from the data and hangs them under
the leaf aggregators, using the ``edge`` template for their link and settings.

Two YAML shapes compile to the same model:

- compact: ``levels``, ``root``, one ``<level>: {defaults, nodes}`` block per
  aggregation level and an ``<edge level>`` block;
- general: ``levels``, a flat ``nodes`` list and an ``edge`` block.
"""

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from onion_fl.core.codec import codecs
from onion_fl.core.registry import PluginError

NODE_ID = (
    r"^[A-Za-z0-9_.-]+$"  # node ids end up in MQTT topics: no '/', '+', '#', spaces
)


class TopologyError(ValueError):
    """A topology is malformed. The message says where."""


class LinkSpec(BaseModel):
    """The link from a node to its parent."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    transport: str | dict[str, Any] = Field(
        "mqtt", description="mqtt, memory, or {name: mqtt, broker: host:port, qos: 1}"
    )
    codec: str = "json"
    profile: str | dict[str, Any] = "lan"

    @field_validator("codec")
    @classmethod
    def _known_codec(cls, value: str) -> str:
        try:
            codecs.get(value)
        except PluginError as exc:
            raise ValueError(str(exc)) from exc
        return value


class NodeSpec(BaseModel):
    """A coordinator or aggregator, with the link to its parent."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(pattern=NODE_ID)
    level: str
    parent: str | None = None
    link_up: LinkSpec | None = None
    settings: dict[str, Any] = Field(default_factory=dict)


class EdgeSpec(BaseModel):
    """Template for the edges that placement hangs under the leaf aggregators."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    link_up: LinkSpec = Field(default_factory=LinkSpec)
    settings: dict[str, Any] = Field(default_factory=dict)


class Topology(BaseModel):
    """A validated tree: one root, aggregators on every level, leaves on the last one."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = ""
    levels: list[str] = Field(min_length=2)
    nodes: list[NodeSpec]
    edge: EdgeSpec = Field(default_factory=EdgeSpec)

    @model_validator(mode="before")
    @classmethod
    def _default_links(cls, data: Any) -> Any:
        if isinstance(data, Mapping) and isinstance(data.get("nodes"), list):
            data = dict(data)
            data["nodes"] = [
                {**node, "link_up": {}}
                if isinstance(node, Mapping)
                and node.get("parent")
                and node.get("link_up") is None
                else node
                for node in data["nodes"]
            ]
        return data

    @model_validator(mode="after")
    def _check_tree(self) -> Topology:
        levels = self.levels
        if len(set(levels)) != len(levels):
            raise ValueError(f"levels must be unique, got {levels}")
        depth = {level: i for i, level in enumerate(levels)}
        ids = [node.id for node in self.nodes]
        duplicates = sorted({i for i in ids if ids.count(i) > 1})
        if duplicates:
            raise ValueError(f"duplicate node ids {duplicates}")
        by_id = {node.id: node for node in self.nodes}

        for node in self.nodes:
            if node.level not in depth:
                raise ValueError(
                    f"node {node.id!r}: unknown level {node.level!r}; levels are {levels}"
                )
            if node.level == levels[-1]:
                raise ValueError(
                    f"node {node.id!r} is on the edge level {levels[-1]!r}; "
                    "edges are created by placement, not listed"
                )
        for level in levels[:-1]:
            if not any(node.level == level for node in self.nodes):
                raise ValueError(f"level {level!r} has no nodes")

        roots = [node for node in self.nodes if node.level == levels[0]]
        if len(roots) != 1:
            raise ValueError(
                f"exactly one root on level {levels[0]!r} is required, found {[n.id for n in roots]}"
            )
        if roots[0].parent is not None or roots[0].link_up is not None:
            raise ValueError(f"root {roots[0].id!r} cannot have a parent or a link_up")

        for node in self.nodes:
            if node is roots[0]:
                continue
            if node.parent not in by_id:
                raise ValueError(
                    f"node {node.id!r}: parent {node.parent!r} does not exist"
                )
            expected = levels[depth[node.level] - 1]
            if by_id[node.parent].level != expected:
                raise ValueError(
                    f"node {node.id!r}: parent {node.parent!r} is on level "
                    f"{by_id[node.parent].level!r}, expected {expected!r}"
                )

        parents = {node.parent for node in self.nodes}
        for node in self.nodes:
            if node.level != levels[-2] and node.id not in parents:
                raise ValueError(
                    f"aggregator {node.id!r} has no children; "
                    f"leaf aggregators must be on level {levels[-2]!r}"
                )
        return self

    # --- navigation ---------------------------------------------------------

    @property
    def root(self) -> NodeSpec:
        return next(node for node in self.nodes if node.level == self.levels[0])

    def node(self, node_id: str) -> NodeSpec:
        for node in self.nodes:
            if node.id == node_id:
                return node
        raise TopologyError(f"unknown node {node_id!r}")

    def parent(self, node_id: str) -> NodeSpec | None:
        parent = self.node(node_id).parent
        return None if parent is None else self.node(parent)

    def children(self, node_id: str) -> list[NodeSpec]:
        return [node for node in self.nodes if node.parent == node_id]

    def leaves(self) -> list[NodeSpec]:
        """Aggregators under which placement hangs the edges."""
        return [node for node in self.nodes if node.level == self.levels[-2]]

    def role(self, node_id: str) -> str:
        return (
            "coordinator"
            if self.node(node_id).level == self.levels[0]
            else "aggregator"
        )

    # --- identity and export ------------------------------------------------

    @property
    def topology_id(self) -> str:
        """SHA-256 of the canonical tree and links.

        Ignores the name, the declaration order and the node settings
        (aggregator, quorum, ...): those belong to the experiment's config_id.
        """
        canonical = {
            "levels": list(self.levels),
            "nodes": sorted(
                (
                    {
                        "id": node.id,
                        "level": node.level,
                        "role": self.role(node.id),
                        "parent": node.parent,
                        "link_up": None
                        if node.link_up is None
                        else node.link_up.model_dump(),
                    }
                    for node in self.nodes
                ),
                key=lambda entry: entry["id"],
            ),
            "edge": {
                "level": self.levels[-1],
                "link_up": self.edge.link_up.model_dump(),
            },
        }
        text = json.dumps(canonical, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def to_graph(self) -> dict[str, Any]:
        """JSON-ready graph for the topology editor, the report and the monitor."""
        return {
            "topology_id": self.topology_id,
            "name": self.name,
            "levels": list(self.levels),
            "nodes": [
                {
                    "id": node.id,
                    "level": node.level,
                    "role": self.role(node.id),
                    "parent": node.parent,
                    "settings": dict(node.settings),
                }
                for node in self.nodes
            ],
            "links": [
                {"src": node.id, "dst": node.parent, **node.link_up.model_dump()}
                for node in self.nodes
                if node.link_up is not None
            ],
            "edge": {
                "level": self.levels[-1],
                "link_up": self.edge.link_up.model_dump(),
                "settings": dict(self.edge.settings),
            },
        }


def parse_topology(raw: Mapping[str, Any]) -> Topology:
    """Build a topology from its compact or general mapping."""
    if not isinstance(raw, Mapping):
        raise TopologyError("a topology must be a mapping")
    general = dict(raw) if "nodes" in raw else _compile_compact(dict(raw))
    try:
        return Topology.model_validate(general)
    except ValidationError as exc:
        details = "; ".join(
            f"{'.'.join(str(part) for part in err['loc']) or 'topology'}: {err['msg']}"
            for err in exc.errors()
        )
        raise TopologyError(details) from exc


def load_topology(path: str | Path) -> Topology:
    """Read a topology YAML file (compact or general form)."""
    import yaml

    path = Path(path)
    if not path.exists():
        raise TopologyError(f"topology file not found: {path}")
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise TopologyError(f"{path.name} is not valid YAML: {exc}") from None
    return parse_topology(raw)


def _compile_compact(data: dict[str, Any]) -> dict[str, Any]:
    levels = data.get("levels")
    if not isinstance(levels, list) or len(levels) < 2:
        raise TopologyError(
            "levels: a root level and an edge level are required at least"
        )
    allowed = {"name", "levels", "root", *levels[1:]}
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise TopologyError(f"unknown keys {unknown}; expected {sorted(allowed)}")

    root = dict(data.get("root") or {})
    root_id = root.pop("id", None)
    if not root_id:
        raise TopologyError("root.id is required")
    nodes: list[dict[str, Any]] = [
        {
            "id": root_id,
            "level": levels[0],
            "parent": None,
            "link_up": None,
            "settings": root,
        }
    ]

    above_level, above_ids = levels[0], [root_id]
    for level in levels[1:-1]:
        block = dict(data.get(level) or {})
        extra = sorted(set(block) - {"defaults", "nodes"})
        if extra:
            raise TopologyError(
                f"{level}: unknown keys {extra}; expected ['defaults', 'nodes']"
            )
        defaults = dict(block.get("defaults") or {})
        default_link = dict(defaults.pop("link_up", None) or {})
        current = []
        for i, raw_node in enumerate(block.get("nodes") or []):
            where = f"{level}.nodes[{i}]"
            entry = dict(raw_node)
            node_id = entry.pop("id", None)
            if not node_id:
                raise TopologyError(f"{where}.id is required")
            parent = entry.pop("parent", None)
            if parent is None:
                if len(above_ids) != 1:
                    raise TopologyError(
                        f"{where}.parent: required because level {above_level!r} "
                        f"has {len(above_ids)} nodes"
                    )
                parent = above_ids[0]
            link = default_link | dict(entry.pop("link_up", None) or {})
            nodes.append(
                {
                    "id": node_id,
                    "level": level,
                    "parent": parent,
                    "link_up": link,
                    "settings": defaults | entry,
                }
            )
            current.append(node_id)
        above_level, above_ids = level, current

    edge = dict(data.get(levels[-1]) or {})
    edge_link = dict(edge.pop("link_up", None) or {})
    return {
        "name": data.get("name", ""),
        "levels": levels,
        "nodes": nodes,
        "edge": {"link_up": edge_link, "settings": edge},
    }
