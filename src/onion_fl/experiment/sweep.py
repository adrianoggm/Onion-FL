from __future__ import annotations

"""Sweeps: every combination of the swept values, times every seed (spec §12.2).

A swept path points into the config (``data.placement.alpha``). A plugin given
by name becomes ``{name: ...}`` when a parameter of it is swept. Each scenario
has its ``config_id``, which leaves the seed out: runs that differ only in
the seed share it, which is what ``compare(over="seed")`` groups on.
"""

import copy
from dataclasses import dataclass
from itertools import product
from typing import Any

from onion_fl.core.ids import config_id
from onion_fl.experiment.config import ConfigError, ExperimentConfig, parse_experiment


@dataclass(frozen=True)
class Scenario:
    name: str
    seed: int
    config: ExperimentConfig
    config_id: str


def _set(tree: dict[str, Any], path: str, value: Any) -> None:
    parts = path.split(".")
    node = tree
    for i, part in enumerate(parts[:-1]):
        child = node.get(part)
        if isinstance(child, str) and parts[i + 1] != "name":
            child = {"name": child}  # a plugin given by name, whose param is swept
        if not isinstance(child, dict):
            raise ConfigError(
                f"sweep {path}: {'.'.join(parts[: i + 1])} is not a mapping"
            )
        node[part] = child
        node = child
    node[parts[-1]] = value


def identity(config: ExperimentConfig) -> dict[str, Any]:
    """What a scenario is: the validated config without its seeds and sweep."""
    out = config.dump()
    out.pop("sweep")
    out.pop("seeds")
    return out


def _label(value: Any) -> str:
    return value if isinstance(value, str) else repr(value)


def _apply(resolved: dict[str, Any], key: str, value: Any) -> str:
    """Set one sweep entry and return its label; ``a,b`` sets several paths at once."""
    paths = [path.strip() for path in key.split(",")]
    key = ",".join(paths)
    if len(paths) == 1:
        _set(resolved, key, value)
        return f"{key}={_label(value)}"
    if not isinstance(value, list) or len(value) != len(paths):
        raise ConfigError(
            f"sweep {key}: each value needs {len(paths)} entries, one per path; "
            f"got {value!r}"
        )
    for path, item in zip(paths, value, strict=True):
        _set(resolved, path, item)
    return f"{key}={','.join(_label(item) for item in value)}"


def scenarios(config: ExperimentConfig) -> list[Scenario]:
    base = config.dump()
    base.pop("sweep")
    seeds = base.pop("seeds")
    keys = sorted(config.sweep)
    out = []
    for values in product(*(config.sweep[k] for k in keys)):
        resolved = copy.deepcopy(base)
        labels = [
            _apply(resolved, key, value)
            for key, value in zip(keys, values, strict=True)
        ]
        name = ",".join(labels) or "base"
        try:
            scenario_config = parse_experiment(resolved | {"seeds": seeds})
        except ConfigError as exc:
            raise ConfigError(f"sweep scenario {name}:\n{exc}") from None
        cid = config_id(identity(scenario_config))
        out += [Scenario(name, seed, scenario_config, cid) for seed in seeds]
    return out
