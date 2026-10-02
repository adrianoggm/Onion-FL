from __future__ import annotations

"""Experiment configuration (spec §12).

::

    name: mix_ab
    topology: four_fogs                 # topologies/four_fogs.yaml, a path, or inline
    data:
      datasets: {swell: {options: {label: binary}}, sweet: {}}
      roles: {test: 0.2, val: 0.1, local_val: 0.2, scaler: global}
      placement: {name: mixing, alpha: 0.5}
    learning:
      model: {name: modular_mlp, trunk_hidden: [64, 32]}
      sharing: fedavg
      trainer: {name: standard, local_epochs: 1, lr: 0.001}
      init: random
    rounds: 20
    evaluation: {metrics: [loss, accuracy, macro_f1], global: {every: 1}}
    runtime: {mode: sim, codec: json}
    seeds: [0, 1, 2]
    sweep: {data.placement.alpha: [0, 0.5, 1]}
    sinks: [prometheus]

Every error names its path (``learning.trainer: ...``). Plugin references are
``"name"`` or ``{name: ..., **params}`` and each plugin validates its params.
"""

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PositiveInt,
    ValidationError,
    field_validator,
    model_validator,
)

from onion_fl.baselines import baseline_models
from onion_fl.core.codec import codecs
from onion_fl.core.registry import PluginError, Registry
from onion_fl.data.ingest import readers, steps
from onion_fl.data.placement import placements
from onion_fl.data.roles import RolesConfig
from onion_fl.learning.aggregators import aggregators, server_optimizers
from onion_fl.learning.metrics import metrics
from onion_fl.learning.model import models
from onion_fl.learning.sharing import sharing
from onion_fl.learning.trainers import inits, trainers
from onion_fl.observability.diagnostics import diagnostics
from onion_fl.roles.policies import (
    create,
    participations,
    stale_weightings,
    stalenesses,
)
from onion_fl.runtime.devices import availability_models, compute_models
from onion_fl.runtime.network import link_profiles

PluginRef = str | dict[str, Any]
SINKS = ("prometheus", "otel")

REGISTRIES: dict[str, Registry] = {
    "codec": codecs,
    "link_profile": link_profiles,
    "compute_model": compute_models,
    "availability_model": availability_models,
    "model": models,
    "sharing": sharing,
    "aggregator": aggregators,
    "server_optimizer": server_optimizers,
    "trainer": trainers,
    "init": inits,
    "metric": metrics,
    "diagnostic": diagnostics,
    "participation": participations,
    "staleness": stalenesses,
    "stale_weighting": stale_weightings,
    "placement": placements,
    "reader": readers,
    "step": steps,
    "baseline": baseline_models,
}


class ConfigError(ValueError):
    """The experiment config is invalid; the message names the path of each error."""


def _plugin(registry: Registry, value: PluginRef) -> PluginRef:
    try:
        create(registry, value)
    except PluginError as exc:
        raise ValueError(str(exc)) from None
    return value


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class DatasetUse(Strict):
    descriptor: str | None = Field(
        None, description="Ruta del YAML; por defecto <datasets>/<nombre>.yaml"
    )
    options: dict[str, Any] = Field(default_factory=dict)


class DataConfig(Strict):
    datasets: dict[str, DatasetUse] = Field(min_length=1)
    roles: RolesConfig = Field(default_factory=RolesConfig)
    placement: PluginRef = Field(
        default_factory=lambda: {"name": "mixing", "alpha": 0.0}
    )

    _placement = field_validator("placement")(lambda v: _plugin(placements, v))


class LearningConfig(Strict):
    model: PluginRef = "modular_mlp"
    sharing: PluginRef = "fedavg"
    trainer: PluginRef = "standard"
    init: PluginRef = "random"

    _model = field_validator("model")(lambda v: _plugin(models, v))
    _sharing = field_validator("sharing")(lambda v: _plugin(sharing, v))
    _trainer = field_validator("trainer")(lambda v: _plugin(trainers, v))
    _init = field_validator("init")(lambda v: _plugin(inits, v))


class EdgeEval(Strict):
    every: PositiveInt | None = None
    models: list[Literal["received", "local"]] = Field(
        default_factory=lambda: ["received", "local"]
    )


class AggregatorEval(Strict):
    every: PositiveInt | None = None
    aggregate_children: bool = True
    holdout: bool = True


class GlobalEval(Strict):
    every: PositiveInt | None = 1


class EvaluationConfig(Strict):
    metrics: list[str] = Field(default_factory=lambda: ["loss", "accuracy", "macro_f1"])
    edge: EdgeEval = Field(default_factory=EdgeEval)
    aggregators: AggregatorEval = Field(default_factory=AggregatorEval)
    global_: GlobalEval = Field(default_factory=GlobalEval, alias="global")

    @field_validator("metrics")
    @classmethod
    def _known_metrics(cls, value: list[str]) -> list[str]:
        unknown = [m for m in value if m not in metrics.names()]
        if unknown:
            raise ValueError(f"unknown metrics {unknown}; available {metrics.names()}")
        return value


class RuntimeConfig(Strict):
    mode: Literal["sim", "real"] = "sim"
    timeout: float = Field(3600.0, gt=0, description="Límite de una ejecución real (s)")
    heartbeat: float | None = Field(
        10.0, gt=0, description="Cada cuánto emite node.heartbeat cada nodo real (s)"
    )
    codec: str | None = Field(
        None, description="Codec de todos los enlaces; cambia el topology_id"
    )

    @field_validator("codec")
    @classmethod
    def _known_codec(cls, value: str | None) -> str | None:
        if value is not None and value not in codecs.names():
            raise ValueError(f"unknown codec {value!r}; available {codecs.names()}")
        return value


class PathsConfig(Strict):
    topologies: str = "topologies"
    datasets: str = "datasets"
    runs: str = "runs"
    cache: str = "data/cache"


class ExperimentConfig(Strict):
    name: str = Field(pattern=r"^[A-Za-z0-9_.-]+$")
    description: str = ""
    topology: str | dict[str, Any]
    data: DataConfig
    learning: LearningConfig = Field(default_factory=LearningConfig)
    rounds: PositiveInt
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    seeds: list[int] = Field(default_factory=lambda: [0], min_length=1)
    sweep: dict[str, list[Any]] = Field(default_factory=dict)
    sinks: list[PluginRef] = Field(default_factory=list)
    paths: PathsConfig = Field(default_factory=PathsConfig)

    @field_validator("sinks")
    @classmethod
    def _known_sinks(cls, value: list[PluginRef]) -> list[PluginRef]:
        for sink in value:
            name = sink if isinstance(sink, str) else sink.get("name")
            if name not in SINKS:
                raise ValueError(
                    f"unknown sink {name!r}; available {list(SINKS)} (jsonl is always on)"
                )
        return value

    @model_validator(mode="after")
    def _sharing_fits_the_model(self) -> ExperimentConfig:
        policy = create(sharing, self.learning.sharing)
        family = create(models, self.learning.model)
        try:
            policy.check_model(family.config)
        except ValueError as exc:
            raise ValueError(f"learning.sharing: {exc}") from None
        return self

    def dump(self) -> dict[str, Any]:
        return self.model_dump(mode="json", by_alias=True)


def _format(exc: ValidationError) -> str:
    lines = []
    for error in exc.errors():
        where = ".".join(str(part) for part in error["loc"] if part != "function-after")
        message = error["msg"].removeprefix("Value error, ")
        lines.append(f"{where}: {message}" if where else message)
    return "\n".join(lines)


def parse_experiment(raw: Mapping[str, Any]) -> ExperimentConfig:
    try:
        return ExperimentConfig.model_validate(dict(raw))
    except ValidationError as exc:
        raise ConfigError(_format(exc)) from None


def load_experiment(path: str | Path) -> ExperimentConfig:
    return parse_experiment(
        yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    )


def experiment_schema() -> dict[str, Any]:
    """JSON Schema of the config plus the catalogue of every plugin, for the front."""
    schema = ExperimentConfig.model_json_schema(by_alias=True)
    schema["plugins"] = {
        kind: registry.describe() for kind, registry in REGISTRIES.items()
    }
    return schema
