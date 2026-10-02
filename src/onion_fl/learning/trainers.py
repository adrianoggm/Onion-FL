from __future__ import annotations

"""Local trainers and model initialisation (spec §7.3).

A trainer updates the model in place and reports what it did; the edge role
turns ``samples`` into simulated compute time and sends ``examples`` up as the
FedAvg weight. Initialisations run once on the coordinator's global model.
"""

import fnmatch
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol

import numpy as np
import torch
import torch.nn.functional as F
from pydantic import BaseModel, ConfigDict, Field, PositiveInt
from torch import nn

from onion_fl.core.registry import Registry
from onion_fl.learning.model import group_of, load_arrays, state_arrays


class TrainError(ValueError):
    """A trainer or an initialisation cannot run with these inputs."""


class Samples(Protocol):
    X: np.ndarray  # float32[n, f]
    y: np.ndarray  # int64[n]


@dataclass(frozen=True)
class TrainResult:
    loss: float  # mean data loss over the processed samples
    samples: int  # samples processed (epochs × examples): what compute is charged for
    examples: int  # distinct local examples: the FedAvg weight
    batches: int


def batches_of(n: int, batch_size: int, rng: np.random.Generator) -> list[np.ndarray]:
    """Shuffled index batches covering ``range(n)`` once."""
    order = rng.permutation(n)
    return [order[i : i + batch_size] for i in range(0, n, batch_size)]


def trainable(model: nn.Module, frozen: Sequence[str]) -> list[str]:
    """Parameter names whose group matches none of the ``frozen`` names or patterns."""
    return [
        name
        for name, _ in model.named_parameters()
        if not any(fnmatch.fnmatchcase(group_of(name), p) for p in frozen)
    ]


def proximal_term(
    model: nn.Module,
    received: Mapping[str, np.ndarray],
    mu: float,
    names: Sequence[str] | None = None,
) -> torch.Tensor:
    """FedProx penalty ``mu/2 · ||w − w_received||²`` over the given (default: all) parameters."""
    params = dict(model.named_parameters())
    total = torch.zeros(())
    for name in params if names is None else names:
        if name in received:
            anchor = torch.as_tensor(
                np.asarray(received[name]), dtype=params[name].dtype
            )
            total = total + ((params[name] - anchor) ** 2).sum()
    return 0.5 * mu * total


trainers = Registry("trainer")

OPTIMIZERS = {"adam": torch.optim.Adam, "sgd": torch.optim.SGD}


class StandardParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    local_epochs: PositiveInt = 1
    batch_size: PositiveInt = 32
    lr: float = Field(1e-3, gt=0)
    optimizer: Literal["adam", "sgd"] = "adam"
    weight_decay: float = Field(0.0, ge=0)
    frozen: list[str] = Field(
        default_factory=list,
        description="Grupos que no se entrenan, por nombre o patrón (trunk, adapter.*)",
    )


@trainers.register(
    "standard",
    title="Estándar",
    description="Épocas locales de descenso por gradiente con entropía cruzada.",
    params=StandardParams,
    explain="Los grupos congelados no cambian: útil para afinar solo la cabeza sobre un tronco preentrenado.",
)
class Standard:
    Params: type[StandardParams] = StandardParams

    def __init__(self, **params: Any) -> None:
        self.params = self.Params(**params)

    def _check(self, received: Mapping[str, np.ndarray] | None) -> None:
        pass

    def _penalty(
        self, model: nn.Module, received: Any, names: Sequence[str]
    ) -> torch.Tensor | float:
        return 0.0

    def train(
        self,
        model: nn.Module,
        data: Samples,
        received: Mapping[str, np.ndarray] | None = None,
        ctx: Any = None,
    ) -> TrainResult:
        X, y = np.asarray(data.X, np.float32), np.asarray(data.y, np.int64)
        if len(X) == 0:
            raise TrainError("no samples to train on")
        if len(X) != len(y):
            raise TrainError(f"X has {len(X)} rows but y has {len(y)} labels")
        self._check(received)
        p = self.params
        names = trainable(model, p.frozen)
        if not names:
            raise TrainError(f"every parameter is frozen by {p.frozen}")

        params = dict(model.named_parameters())
        was = {name: t.requires_grad for name, t in params.items()}
        for name, tensor in params.items():
            tensor.requires_grad_(name in names)
        optimizer = OPTIMIZERS[p.optimizer](
            [params[n] for n in names], lr=p.lr, weight_decay=p.weight_decay
        )
        rng = ctx.rng if ctx is not None else np.random.default_rng(0)
        xs, ys = torch.from_numpy(X), torch.from_numpy(y)
        total, batches = 0.0, 0
        model.train()
        try:
            # Dropout draws from torch's global RNG: seed it from the node's rng
            # inside a fork so runs are reproducible and the caller's state is kept.
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(int(rng.integers(2**63)))
                for _ in range(p.local_epochs):
                    for idx in batches_of(len(X), p.batch_size, rng):
                        optimizer.zero_grad()
                        loss = F.cross_entropy(model(xs[idx]), ys[idx])
                        total += float(loss.item()) * len(idx)
                        (loss + self._penalty(model, received, names)).backward()
                        optimizer.step()
                        batches += 1
        finally:
            for name, tensor in params.items():
                tensor.requires_grad_(was[name])
        samples = p.local_epochs * len(X)
        return TrainResult(
            loss=total / samples, samples=samples, examples=len(X), batches=batches
        )


class FedProxParams(StandardParams):
    mu: float = Field(0.01, ge=0, description="Peso del término proximal")


@trainers.register(
    "fedprox",
    title="FedProx",
    description="Entrenamiento estándar más un término proximal hacia el modelo recibido.",
    params=FedProxParams,
    explain="Con datos heterogéneos limita cuánto se aleja cada edge del global (Li et al., 2020).",
)
class FedProx(Standard):
    Params = FedProxParams

    def _check(self, received: Mapping[str, np.ndarray] | None) -> None:
        if received is None:
            raise TrainError("fedprox needs the received global state")

    def _penalty(
        self, model: nn.Module, received: Any, names: Sequence[str]
    ) -> torch.Tensor:
        return proximal_term(model, received, self.params.mu, names)


class StubParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    shift: float = Field(1.0, description="Lo que se suma a cada peso")
    examples: PositiveInt = Field(10, description="Muestras que declara")


@trainers.register(
    "stub",
    title="Entrenador de prueba",
    description="No aprende: suma una constante a cada peso. Solo para tests de protocolo.",
    params=StubParams,
    explain="Permite comprobar rondas y agregación sin datos (docs/RULES.md).",
)
class Stub:
    def __init__(self, **params: Any) -> None:
        self.params = StubParams(**params)

    def train(
        self,
        model: nn.Module,
        data: Samples | None = None,
        received: Mapping[str, np.ndarray] | None = None,
        ctx: Any = None,
    ) -> TrainResult:
        with torch.no_grad():
            for tensor in model.parameters():
                tensor.add_(self.params.shift)
        n = self.params.examples
        return TrainResult(loss=0.0, samples=n, examples=n, batches=1)


# --- initialisation -----------------------------------------------------------------

inits = Registry("init")


class RandomParams(BaseModel):
    seed: int | None = Field(
        None, description="Semilla propia; sin ella vale la del experimento"
    )


@inits.register(
    "random",
    title="Aleatoria",
    description="Pesos aleatorios deterministas por parte del modelo.",
    params=RandomParams,
)
class RandomInit:
    def __init__(self, seed: int | None = None) -> None:
        self.seed = seed

    def init(self, model: nn.Module, ctx: Any = None) -> list[str]:
        if self.seed is not None:
            model.init_parameters(self.seed)
        return list(model.state_dict())


class CheckpointParams(BaseModel):
    path: str = Field(description="Fichero .npz con el estado (save_checkpoint)")
    groups: list[str] = Field(
        default_factory=lambda: ["*"], description="Grupos que se cargan"
    )


@inits.register(
    "checkpoint",
    title="Desde checkpoint",
    description="Carga grupos de un modelo guardado; el resto queda aleatorio.",
    params=CheckpointParams,
    explain="Transferencia: por ejemplo, partir del tronco de un baseline centralizado.",
)
class CheckpointInit:
    def __init__(self, path: str, groups: Sequence[str] = ("*",)) -> None:
        self.path, self.groups = Path(path), list(groups)

    def init(self, model: nn.Module, ctx: Any = None) -> list[str]:
        with np.load(self.path, allow_pickle=False) as archive:
            arrays = {key: archive[key] for key in archive.files}
        held = {group_of(key) for key in arrays}
        for pattern in self.groups:
            if not any(fnmatch.fnmatchcase(g, pattern) for g in held):
                raise TrainError(
                    f"{self.path} holds no group matching {pattern!r}: {sorted(held)}"
                )
        wanted = {
            key: value
            for key, value in arrays.items()
            if any(fnmatch.fnmatchcase(group_of(key), p) for p in self.groups)
        }
        loaded = load_arrays(model, wanted)
        if not loaded:
            raise TrainError(f"{self.path} has no key of this model in {self.groups}")
        return loaded


def save_checkpoint(model: nn.Module, path: str | Path) -> None:
    """Write the model state as an ``.npz`` that ``checkpoint`` can load."""
    np.savez(path, **state_arrays(model))
