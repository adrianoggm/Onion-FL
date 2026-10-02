from __future__ import annotations

"""Dry-run previews for the tutorial: pure functions, no data, nothing trained.

- ``sharing_preview``: which parameter groups cross each link and what each
  level keeps, for a sharing policy on a topology.
- ``placement_preview``: how many subjects of each dataset each leaf gets, for
  a placement plugin and hypothetical subject counts (bookkeeping only).
- ``link_preview``: latency, transmission time and loss of a network profile.
"""

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from onion_fl.core.context import node_rng
from onion_fl.core.topology import parse_topology
from onion_fl.data.placement import Item, placements, resolve_leaves
from onion_fl.learning.model import DataShape, models, param_groups
from onion_fl.learning.sharing import sharing, stored_at, traffic
from onion_fl.roles.policies import create
from onion_fl.runtime.network import resolve_profile


def sharing_preview(
    topology: Mapping[str, Any],
    sharing_ref: str | Mapping[str, Any],
    datasets: Sequence[str],
    model_ref: str | Mapping[str, Any] = "modular_mlp",
    task: str = "stress",
) -> dict[str, Any]:
    tree = parse_topology(topology)
    policy = create(sharing, sharing_ref, "fedavg")
    policy.validate_for(tree)
    params = (
        {}
        if isinstance(model_ref, str)
        else {k: v for k, v in model_ref.items() if k != "name"}
    )
    name = (
        model_ref
        if isinstance(model_ref, str)
        else model_ref.get("name", "modular_mlp")
    )
    family = models.create(
        name, params | policy.model_requirements
    )  # what the policy needs
    shapes = [
        DataShape(dataset=d, task=task, n_features=1, n_classes=2) for d in datasets
    ]
    groups = list(param_groups(family.build(shapes).state_dict()))
    return {
        "groups": groups,
        "links": traffic(tree, policy, groups),
        "held": {
            level: stored_at(tree, level, policy, groups) for level in tree.levels[:-1]
        },
        "model_requirements": policy.model_requirements,
    }


def placement_preview(
    topology: Mapping[str, Any],
    placement_ref: str | Mapping[str, Any],
    datasets: Mapping[str, int],
    seed: int = 0,
) -> dict[str, dict[str, Any]]:
    tree = parse_topology(topology)
    names = sorted(datasets)
    leaves = resolve_leaves(tree, names)
    items = [
        Item(
            f"{d}-{i}", d, 1, (1, 0) if i % 2 else (0, 1)
        )  # one subject, alternating class
        for d in names
        for i in range(1, int(datasets[d]) + 1)
    ]
    plugin = create(placements, placement_ref, "mixing")
    assigned = plugin.assign(items, leaves, node_rng(seed, "placement/clients"))
    dataset_of = {item.id: item.dataset for item in items}
    out = {}
    for leaf in leaves:
        counts: dict[str, int] = {}
        for item_id in assigned.get(leaf.id, []):
            counts[dataset_of[item_id]] = counts.get(dataset_of[item_id], 0) + 1
        total = sum(counts.values())
        entropy = (
            -sum(n / total * math.log2(n / total) for n in counts.values())
            if total
            else 0.0
        )
        out[leaf.id] = {
            "home": leaf.home,
            "subjects": total,
            "datasets": dict(sorted(counts.items())),
            "entropy": abs(entropy),
        }
    return out


def link_preview(
    profile_ref: str | Mapping[str, Any],
    size_bytes: int,
    samples: int = 2000,
    seed: int = 0,
) -> dict[str, Any]:
    profile = resolve_profile(profile_ref)
    rng = np.random.default_rng(seed)
    latencies = np.array([profile.sample_latency(rng) for _ in range(samples)])
    return {
        "profile": profile.model_dump(),
        "latency": {
            "mean": float(latencies.mean()),
            "p50": float(np.percentile(latencies, 50)),
            "p95": float(np.percentile(latencies, 95)),
        },
        "transmission_up_s": profile.transmission_s(size_bytes, "up"),
        "transmission_down_s": profile.transmission_s(size_bytes, "down"),
        "loss": profile.loss,
    }
