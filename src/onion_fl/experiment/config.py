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
from onion_fl.continuum.memory import memories
from onion_fl.core.codec import codecs
from onion_fl.core.registry import PluginError, Registry
from onion_fl.data.ingest import readers, steps
from onion_fl.data.placement import placements
from onion_fl.data.roles import RolesConfig
from onion_fl.data.stream import LabelsConfig, Samples, Staggered, StreamConfig
from onion_fl.learning.aggregators import aggregators, server_optimizers
from onion_fl.learning.attacks import attacks
from onion_fl.learning.metrics import metrics
from onion_fl.learning.model import models
from onion_fl.learning.privacy import privacies
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
from onion_fl.transports import transports

PluginRef = str | dict[str, Any]
SINKS = ("prometheus", "otel")

REGISTRIES: dict[str, Registry] = {
    "codec": codecs,
    "transport": transports,
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
    "attack": attacks,
    "privacy": privacies,
    "memory": memories,
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


class ContinualConfig(Strict):
    memory: PluginRef = Field(
        "none", description="Memoria de replay de cada edge que entrena"
    )
    replay_ratio: float = Field(
        0.25,
        ge=0,
        lt=1,
        description="Fracción de cada entrenamiento que sale de la memoria",
    )

    _memory = field_validator("memory")(lambda v: _plugin(memories, v))


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
    server_optimizer: PluginRef | None = Field(
        None,
        description="Optimizador de servidor de la raíz; sin él, el de la topología",
        exclude_if=lambda v: v is None,  # unset, it keeps existing config_ids
    )
    aggregator: PluginRef | None = Field(
        None,
        description="Agregador de los nodos cuyos hijos son edges; sin él, el de la topología",
        exclude_if=lambda v: v is None,  # unset, it keeps existing config_ids
    )

    _model = field_validator("model")(lambda v: _plugin(models, v))
    _aggregator = field_validator("aggregator")(
        lambda v: v if v is None else _plugin(aggregators, v)
    )
    _server_optimizer = field_validator("server_optimizer")(
        lambda v: v if v is None else _plugin(server_optimizers, v)
    )
    _sharing = field_validator("sharing")(lambda v: _plugin(sharing, v))
    _trainer = field_validator("trainer")(lambda v: _plugin(trainers, v))
    _init = field_validator("init")(lambda v: _plugin(inits, v))


class EdgeEval(Strict):
    every: PositiveInt | None = None
    models: list[Literal["received", "local", "personal", "finetuned"]] = Field(
        default_factory=lambda: ["received", "local"]
    )
    finetune: PluginRef | None = Field(
        None,
        description="Entrenador del ajuste fino antes de puntuar 'finetuned'",
        exclude_if=lambda v: v is None,  # unset, it keeps existing config_ids
    )

    _finetune = field_validator("finetune")(
        lambda v: v if v is None else _plugin(trainers, v)
    )

    @model_validator(mode="after")
    def _finetune_and_finetuned_go_together(self) -> EdgeEval:
        if "finetuned" in self.models and self.finetune is None:
            raise ValueError("models: 'finetuned' needs evaluation.edge.finetune")
        if self.finetune is not None and "finetuned" not in self.models:
            raise ValueError(
                "finetune: it only runs for 'finetuned' scores; add it to models"
            )
        return self


class AggregatorEval(Strict):
    every: PositiveInt | None = None
    aggregate_children: bool = True
    holdout: bool = True


class GlobalEval(Strict):
    every: PositiveInt | None = 1
    subjects: Literal["test", "val"] | None = Field(
        None,
        description=(
            "Sujetos con los que se evalúa el modelo global: los de test (por "
            "defecto) o los de validación, para elegir hiperparámetros del servidor"
        ),
        exclude_if=lambda v: v is None,  # unset, it keeps existing config_ids
    )


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
    attack: PluginRef | None = Field(
        None,
        description="Ataque de una fracción de edges de cada dataset",
        exclude_if=lambda v: v is None,  # unset, it keeps existing config_ids
    )

    privacy: PluginRef | None = Field(
        None,
        description="Privacidad diferencial local en cada edge que entrena",
        exclude_if=lambda v: v is None,  # unset, it keeps existing config_ids
    )

    stream: StreamConfig | None = Field(
        None,
        description="Cada edge recibe sus filas en flujo, en su orden temporal",
        exclude_if=lambda v: v is None,  # unset, it keeps existing config_ids
    )
    labels: LabelsConfig | None = Field(
        None,
        description="Fracción etiquetada de un stream y retraso de sus etiquetas",
        exclude_if=lambda v: v is None,  # unset, it keeps existing config_ids
    )
    continual: ContinualConfig | None = Field(
        None,
        description="Memoria de replay de los edges de un stream",
        exclude_if=lambda v: v is None,  # unset, it keeps existing config_ids
    )

    _privacy = field_validator("privacy")(
        lambda v: v if v is None else _plugin(privacies, v)
    )
    _attack = field_validator("attack")(
        lambda v: v if v is None else _plugin(attacks, v)
    )

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

    @model_validator(mode="after")
    def _stream_fits(self) -> ExperimentConfig:
        """What a stream cannot do yet (continuum C3) is refused before any run."""
        if self.labels is not None and self.stream is None:
            raise ValueError("labels: they belong to a stream; set stream too")
        if self.continual is not None and self.stream is None:
            raise ValueError(
                "continual: a replay memory keeps rows of a stream; set stream too"
            )
        if self.stream is None:
            return self
        roles, init, ref = self.data.roles, self.learning.init, self.data.placement
        one_each = isinstance(self.stream.start, Staggered) or isinstance(
            self.stream.bootstrap, Samples
        )
        params = (
            {}
            if isinstance(ref, str)
            else {k: v for k, v in ref.items() if k != "name"}
        )
        name = ref if isinstance(ref, str) else ref["name"]
        merging = getattr(placements.create(name, params), "merge", False)
        refusals = [
            (self.runtime.mode == "real", "real runs do not stream yet (C9)"),
            (
                (init if isinstance(init, str) else init["name"]) == "run",
                "learning.init: run cannot continue a stream yet; the bundle holds "
                "no stream state",
            ),
            (
                roles.local_val > 0,
                "data.roles.local_val: a tail of rows conflicts with time order; "
                "a stream scores test-then-train instead",
            ),
            (
                roles.scaler == "local",
                "data.roles.scaler: local would refit on rows after the bootstrap",
            ),
            (
                one_each and roles.subjects_per_client > 1,
                "data.roles.subjects_per_client: a staggered start or a bootstrap "
                "in samples needs one subject per edge",
            ),
            (
                one_each and merging,
                f"data.placement: {name} merges edges, so a staggered start or a "
                "bootstrap in samples cannot be kept per subject",
            ),
        ]
        for refused, message in refusals:
            if refused:
                raise ValueError(f"stream: {message}")
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
