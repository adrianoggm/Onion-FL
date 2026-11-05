from __future__ import annotations

"""Sweeps: every combination of the swept values, times every seed (spec §12.2).

A swept path points into the config (``data.placement.alpha``). A plugin given
by name becomes ``{name: ...}`` when a parameter of it is swept. Each scenario
has its ``config_id``, which leaves the seed out: runs that differ only in
the seed share it, which is what ``compare(over="seed")`` groups on.

A ``search`` then turns each swept scenario into ``budget`` trials, drawn
from the search's own seed (the same draws in every swept scenario). Each
trial keeps the search with its ``trial`` index, so the search seed is part
of its ``config_id``, and its ``configuration`` is the swept scenario's name.
"""

import copy
from dataclasses import dataclass
from itertools import product
from typing import Any

import numpy as np

from onion_fl.core.ids import config_id
from onion_fl.experiment.config import (
    ConfigError,
    ExperimentConfig,
    SearchConfig,
    parse_experiment,
)


@dataclass(frozen=True)
class Scenario:
    name: str
    seed: int
    config: ExperimentConfig
    config_id: str
    configuration: str | None = None  # the swept scenario of a search trial


def draw(search: SearchConfig) -> list[dict[str, Any]]:
    """The ``budget`` trials of a random search: a value per path, from its seed."""
    rng = np.random.default_rng(search.seed)
    trials = []
    for _ in range(search.budget):
        trial = {}
        for path in sorted(search.space):
            domain = search.space[path]
            if isinstance(domain, list):
                trial[path] = copy.deepcopy(domain[int(rng.integers(len(domain)))])
            elif domain.log:
                low, high = np.log(domain.low), np.log(domain.high)
                trial[path] = float(np.exp(rng.uniform(low, high)))
            else:
                trial[path] = float(rng.uniform(domain.low, domain.high))
        trials.append(trial)
    return trials


def _trials(resolved: dict[str, Any]) -> list[tuple[dict[str, Any], str | None]]:
    """The resolved config of each trial of its search, with its label."""
    raw = resolved.get("search")
    if raw is None or raw.get("trial") is not None:  # no search, or one trial
        return [(resolved, None)]
    try:
        search = SearchConfig.model_validate(raw)
    except ValueError as exc:
        raise ConfigError(f"search: {exc}") from None
    out = []
    for i, values in enumerate(draw(search)):
        trial = copy.deepcopy(resolved)
        for path, value in values.items():
            _set(trial, path, value)
        trial["search"]["trial"] = i
        out.append((trial, f"search.trial={i}"))
    return out


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
    """How a swept value appears in a scenario name; a whole plugin is name(k=v,...)."""
    if isinstance(value, str):
        return value
    if isinstance(value, dict) and isinstance(value.get("name"), str):
        params = ",".join(
            f"{k}={_label(v)}" for k, v in sorted(value.items()) if k != "name"
        )
        return f"{value['name']}({params})"
    return repr(value)


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
        swept = ",".join(labels) or "base"
        for trial, label in _trials(resolved):
            name = ",".join([*labels, label]) if label else swept
            try:
                scenario_config = parse_experiment(trial | {"seeds": seeds})
            except ConfigError as exc:
                raise ConfigError(f"sweep scenario {name}:\n{exc}") from None
            cid = config_id(identity(scenario_config))
            out += [
                Scenario(name, seed, scenario_config, cid, configuration=swept)
                for seed in seeds
            ]
    return out
