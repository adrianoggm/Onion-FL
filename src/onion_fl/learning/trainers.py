from __future__ import annotations

"""Local trainers and model initialisation (spec §7.3).

A trainer updates the model in place and reports what it did; the edge role
turns ``samples`` into simulated compute time and sends ``examples`` up as the
FedAvg weight. Initialisations run once on the coordinator's global model.
"""

import copy
import fnmatch
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal, Protocol

import numpy as np
import torch
import torch.nn.functional as F
from pydantic import BaseModel, ConfigDict, Field, PositiveInt
from torch import nn
from torch.func import functional_call

from onion_fl.core.context import child_rng
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
    aux: Mapping[str, np.ndarray] = field(default_factory=dict)  # sent with the update


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


def _samples(data: Samples) -> tuple[torch.Tensor, torch.Tensor]:
    X, y = np.asarray(data.X, np.float32), np.asarray(data.y, np.int64)
    if len(X) == 0:
        raise TrainError("no samples to train on")
    if len(X) != len(y):
        raise TrainError(f"X has {len(X)} rows but y has {len(y)} labels")
    return torch.from_numpy(X), torch.from_numpy(y)


@contextmanager
def _training(
    models: Sequence[nn.Module], names: Sequence[str], rng: np.random.Generator
) -> Iterator[None]:
    """Train mode with gradients only on ``names``, dropout seeded from ``rng``.

    Dropout draws from torch's global RNG: it is seeded from the node's rng
    inside a fork, so runs are reproducible and the caller's state is kept.
    """
    wanted = set(names)
    saved = [{n: t.requires_grad for n, t in m.named_parameters()} for m in models]
    for module in models:
        module.train()
        for name, tensor in module.named_parameters():
            tensor.requires_grad_(name in wanted)
    try:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(rng.integers(2**63)))
            yield
    finally:
        for module, was in zip(models, saved, strict=True):
            for name, tensor in module.named_parameters():
                tensor.requires_grad_(was[name])


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
        xs, ys = _samples(data)
        self._check(received)
        p = self.params
        names = trainable(model, p.frozen)
        if not names:
            raise TrainError(f"every parameter is frozen by {p.frozen}")
        params = dict(model.named_parameters())
        optimizer = OPTIMIZERS[p.optimizer](
            [params[n] for n in names], lr=p.lr, weight_decay=p.weight_decay
        )
        rng = ctx.rng if ctx is not None else np.random.default_rng(0)
        total, batches = 0.0, 0
        with _training([model], names, rng):
            for _ in range(p.local_epochs):
                for idx in batches_of(len(xs), p.batch_size, rng):
                    optimizer.zero_grad()
                    loss = F.cross_entropy(model(xs[idx]), ys[idx])
                    total += float(loss.item()) * len(idx)
                    (loss + self._penalty(model, received, names)).backward()
                    optimizer.step()
                    batches += 1
        samples = p.local_epochs * len(xs)
        return TrainResult(
            loss=total / samples, samples=samples, examples=len(xs), batches=batches
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


class DittoParams(StandardParams):
    lam: float = Field(
        0.1, ge=0, description="λ: cuánto se ata el modelo personal al global"
    )
    personal_epochs: PositiveInt | None = Field(
        None, description="Épocas del modelo personal; por defecto local_epochs"
    )


@trainers.register(
    "ditto",
    title="Ditto",
    description="Entrena el global como standard y, aparte, un modelo personal atado al global.",
    params=DittoParams,
    explain=(
        "El modelo personal minimiza su pérdida más λ/2·‖v − w‖² hacia el global "
        "recibido; cada edge lo conserva entre rondas y se puntúa como 'personal' "
        "(Li et al., 2021)."
    ),
)
class Ditto(Standard):
    Params = DittoParams

    def __init__(self, **params: Any) -> None:
        super().__init__(**params)
        self._personal: nn.Module | None = None

    def train(
        self,
        model: nn.Module,
        data: Samples,
        received: Mapping[str, np.ndarray] | None = None,
        ctx: Any = None,
    ) -> TrainResult:
        if received is None:
            raise TrainError("ditto needs the received global state")
        if self._personal is None:
            self._personal = copy.deepcopy(model)  # the first global model
        result = super().train(model, data, received, ctx)
        p = self.params
        tied = FedProx(
            **p.model_dump(exclude={"lam", "personal_epochs", "local_epochs"}),
            local_epochs=p.personal_epochs or p.local_epochs,
            mu=p.lam,
        )
        # Its own stream: the global model trains exactly as with standard.
        rng = ctx.rng if ctx is not None else np.random.default_rng(0)
        own = tied.train(
            self._personal, data, received, SimpleNamespace(rng=child_rng(rng))
        )
        return TrainResult(
            loss=result.loss,
            samples=result.samples + own.samples,
            examples=result.examples,
            batches=result.batches + own.batches,
        )

    def personal(self) -> nn.Module | None:
        return self._personal


class APFLParams(StandardParams):
    alpha: float = Field(
        0.5, ge=0, le=1, description="Peso inicial del modelo personal en la mezcla"
    )
    adapt_alpha: bool = Field(True, description="Cada edge aprende su α")
    alpha_lr: float = Field(0.01, gt=0, description="Paso del descenso sobre α")


@trainers.register(
    "apfl",
    title="APFL",
    description="Mezcla un modelo personal con el global: α·v + (1−α)·w.",
    params=APFLParams,
    explain=(
        "En cada paso entrena el global w con sus datos y el personal v a través "
        "de la mezcla; con adapt_alpha cada edge ajusta su α (Deng et al., 2020)."
    ),
)
class APFL(Standard):
    Params = APFLParams

    def __init__(self, **params: Any) -> None:
        super().__init__(**params)
        self.alpha = self.params.alpha
        self._v: nn.Module | None = None
        self._w: dict[str, torch.Tensor] = {}

    def train(
        self,
        model: nn.Module,
        data: Samples,
        received: Mapping[str, np.ndarray] | None = None,
        ctx: Any = None,
    ) -> TrainResult:
        xs, ys = _samples(data)
        p = self.params
        names = trainable(model, p.frozen)
        if not names:
            raise TrainError(f"every parameter is frozen by {p.frozen}")
        if self._v is None:
            self._v = copy.deepcopy(model)
        w, v = dict(model.named_parameters()), dict(self._v.named_parameters())
        make = OPTIMIZERS[p.optimizer]
        opt_w = make([w[n] for n in names], lr=p.lr, weight_decay=p.weight_decay)
        opt_v = make([v[n] for n in names], lr=p.lr, weight_decay=p.weight_decay)
        alpha = torch.tensor(self.alpha, requires_grad=p.adapt_alpha)
        rng = ctx.rng if ctx is not None else np.random.default_rng(0)
        side = child_rng(rng)  # dropout of the personal pass, apart from w's
        total, batches = 0.0, 0
        with _training([model, self._v], names, rng):
            for _ in range(p.local_epochs):
                for idx in batches_of(len(xs), p.batch_size, rng):
                    opt_w.zero_grad()
                    loss = F.cross_entropy(model(xs[idx]), ys[idx])
                    loss.backward()
                    opt_w.step()
                    opt_v.zero_grad()
                    alpha.grad = None
                    # Uses w after this batch's step; Deng et al. take the previous
                    # iterate, as common implementations do not: the gap is one step.
                    mixed = {n: alpha * v[n] + (1 - alpha) * w[n].detach() for n in v}
                    with torch.random.fork_rng(devices=[]):
                        torch.manual_seed(int(side.integers(2**63)))
                        out = functional_call(self._v, mixed, (xs[idx],))
                    F.cross_entropy(out, ys[idx]).backward()
                    opt_v.step()
                    if p.adapt_alpha:
                        with torch.no_grad():
                            alpha -= p.alpha_lr * alpha.grad
                            alpha.clamp_(0.0, 1.0)
                    total += float(loss.item()) * len(idx)
                    batches += 1
        self.alpha = float(alpha)
        self._w = {n: t.detach().clone() for n, t in w.items()}
        seen = p.local_epochs * len(xs)
        return TrainResult(
            loss=total / seen,
            samples=2 * seen,  # the global and the personal model per batch
            examples=len(xs),
            batches=batches,
        )

    def personal(self) -> nn.Module | None:
        if self._v is None:
            return None
        out = copy.deepcopy(self._v)
        with torch.no_grad():
            for name, tensor in out.named_parameters():
                tensor.copy_(self.alpha * tensor + (1 - self.alpha) * self._w[name])
        return out


class FedRepParams(StandardParams):
    head_epochs: PositiveInt = Field(
        5, description="Épocas de la cabeza con el cuerpo congelado"
    )
    head: list[str] = Field(
        default_factory=lambda: ["head*"],
        description="Grupos que forman la cabeza, por nombre o patrón",
    )


@trainers.register(
    "fedrep",
    title="FedRep",
    description="Entrena primero la cabeza local y después el cuerpo compartido (local_epochs).",
    params=FedRepParams,
    explain=(
        "La cabeza se queda en el edge: exige una compartición que la mantenga "
        "local, como fedper; solo viaja la representación (Collins et al., 2021)."
    ),
)
class FedRep(Standard):
    Params = FedRepParams

    @property
    def local_groups(self) -> list[str]:
        return list(self.params.head)

    def train(
        self,
        model: nn.Module,
        data: Samples,
        received: Mapping[str, np.ndarray] | None = None,
        ctx: Any = None,
    ) -> TrainResult:
        p = self.params
        base = p.model_dump(exclude={"head_epochs", "head", "local_epochs", "frozen"})
        groups = {group_of(name) for name, _ in model.named_parameters()}
        body = sorted(
            g for g in groups if not any(fnmatch.fnmatchcase(g, h) for h in p.head)
        )
        head = Standard(
            **base, local_epochs=p.head_epochs, frozen=[*p.frozen, *body]
        ).train(model, data, received, ctx)
        rest = Standard(
            **base, local_epochs=p.local_epochs, frozen=[*p.frozen, *p.head]
        ).train(model, data, received, ctx)
        return TrainResult(
            loss=rest.loss,
            samples=head.samples + rest.samples,
            examples=rest.examples,
            batches=head.batches + rest.batches,
        )


class FedBABUParams(StandardParams):
    frozen: list[str] = Field(
        default_factory=lambda: ["head*"],
        description="Grupos congelados; por defecto la cabeza, que no se entrena",
    )


@trainers.register(
    "fedbabu",
    title="FedBABU",
    description="Solo aprende el cuerpo; la cabeza se queda como se inicializó.",
    params=FedBABUParams,
    explain=(
        "Se evalúa ajustando el modelo recibido con evaluation.edge.finetune y "
        "puntuándolo como 'finetuned' (Oh et al., 2022)."
    ),
)
class FedBABU(Standard):
    Params = FedBABUParams


class StubParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    shift: float = Field(1.0, description="Lo que se suma a cada peso")
    examples: PositiveInt = Field(10, description="Muestras que declara")
    noise: float = Field(
        0.0,
        ge=0,
        description="Desviación de un ruido por nodo (rng del nodo); 0 sin ruido",
    )


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
        rng = ctx.rng if ctx is not None else np.random.default_rng(0)
        with torch.no_grad():
            for tensor in model.parameters():
                tensor.add_(self.params.shift)
                if (
                    self.params.noise
                ):  # differs per node and round, still learns nothing
                    draw = rng.normal(0.0, self.params.noise, size=tuple(tensor.shape))
                    tensor.add_(torch.as_tensor(draw, dtype=tensor.dtype))
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
