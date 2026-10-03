from __future__ import annotations

"""Modular model with namespaced parameter keys (spec §7.1).

::

    adapter.<dataset>.* | adapter.*     per dataset, or one over harmonized features
    trunk.*      | trunk.<dataset>.*    shared body, or one per dataset
    head.<task>.* | head.<dataset>.*    one head per task, or one per dataset

An edge builds the model with its own ``DataShape`` and so instantiates only
its parts. The coordinator builds it with every shape of the experiment to get
the initial global state; aggregators never build a model, they average arrays.
"""

import hashlib
import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Literal

import numpy as np
import torch
from pydantic import BaseModel, ConfigDict, Field, PositiveInt
from torch import nn

from onion_fl.core.registry import Registry

NAME = r"^[A-Za-z][A-Za-z0-9_]*$"  # no dots or leading digits: keys stay unambiguous
NAMESPACES = ("adapter", "trunk", "head")


class DataShape(BaseModel):
    """What the model needs to know about one dataset."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    dataset: str = Field(pattern=NAME)
    task: str = Field(pattern=NAME)
    n_features: PositiveInt
    n_classes: int = Field(ge=2)


class ModularMLPConfig(BaseModel):
    """Configuration of ``modular_mlp``."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    adapter_width: PositiveInt = Field(
        64, description="Anchura común a la salida de cada adaptador"
    )
    adapters: Literal["per_dataset", "shared"] = Field(
        "per_dataset",
        description="Un adaptador por dataset, o uno común sobre features armonizadas",
    )
    trunk_hidden: list[PositiveInt] = Field(
        default_factory=lambda: [64, 32], description="Capas ocultas del tronco"
    )
    trunk: Literal["shared", "per_dataset"] = Field(
        "shared", description="Un tronco común o uno por dataset"
    )
    heads: Literal["per_task", "per_dataset"] = Field(
        "per_task", description="Una cabeza por tarea o una por dataset"
    )
    dropout: float = Field(0.2, ge=0, lt=1)


def _block(n_in: int, n_out: int, dropout: float) -> list[nn.Module]:
    return [nn.Linear(n_in, n_out), nn.ReLU(), nn.Dropout(dropout)]


def _seed_for(seed: int, name: str) -> int:
    digest = hashlib.sha256(f"{seed}/{name}".encode()).digest()
    return int.from_bytes(digest[:8], "little") & (2**63 - 1)


class ModularMLP(nn.Module):
    """Adapter per dataset, shared or per-dataset trunk, head per task or per dataset."""

    def __init__(
        self, config: ModularMLPConfig, shapes: Sequence[DataShape], seed: int = 0
    ) -> None:
        super().__init__()
        if not shapes:
            raise ValueError("ModularMLP needs at least one DataShape")
        names = [shape.dataset for shape in shapes]
        duplicates = sorted({n for n in names if names.count(n) > 1})
        if duplicates:
            raise ValueError(
                f"datasets must be unique, got {duplicates} more than once"
            )
        self.config = config
        self.shapes = {shape.dataset: shape for shape in shapes}

        classes: dict[str, int] = {}
        for shape in shapes:
            key = self._head_key(shape)
            if classes.setdefault(key, shape.n_classes) != shape.n_classes:
                raise ValueError(
                    f"head {key!r} is shared but its datasets have {classes[key]} and "
                    f"{shape.n_classes} classes; use heads='per_dataset' or rename the task"
                )

        width, dims = config.adapter_width, [config.adapter_width, *config.trunk_hidden]
        if config.adapters == "shared":
            features = {s.n_features for s in shapes}
            if len(features) != 1:
                raise ValueError(
                    "adapters='shared' needs the same n_features in every dataset "
                    f"(harmonized features), got {sorted(features)}"
                )
            self.adapter = nn.Sequential(*_block(features.pop(), width, config.dropout))
        else:
            self.adapter = nn.ModuleDict(
                {
                    s.dataset: nn.Sequential(
                        *_block(s.n_features, width, config.dropout)
                    )
                    for s in shapes
                }
            )

        def trunk() -> nn.Sequential:
            layers: list[nn.Module] = []
            for n_in, n_out in zip(dims, dims[1:], strict=False):
                layers += _block(n_in, n_out, config.dropout)
            return nn.Sequential(*layers)

        self.trunk = (
            trunk()
            if config.trunk == "shared"
            else nn.ModuleDict({n: trunk() for n in names})
        )
        self.head = nn.ModuleDict(
            {key: nn.Linear(dims[-1], n) for key, n in classes.items()}
        )
        self.init_parameters(seed)

    def _head_key(self, shape: DataShape) -> str:
        return shape.task if self.config.heads == "per_task" else shape.dataset

    def init_parameters(self, seed: int) -> None:
        # Every layer gets its own generator, seeded by (seed, layer path): a part
        # starts identical in an edge model and in the coordinator's global model.
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                gen = torch.Generator().manual_seed(_seed_for(seed, name))
                nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5), generator=gen)
                bound = 1 / math.sqrt(module.in_features)
                nn.init.uniform_(module.bias, -bound, bound, generator=gen)

    def _dataset(self, dataset: str | None) -> str:
        if dataset is None:
            if len(self.shapes) != 1:
                raise ValueError(
                    f"this model holds datasets {sorted(self.shapes)}; pass dataset=..."
                )
            dataset = next(iter(self.shapes))
        if dataset not in self.shapes:
            raise ValueError(
                f"unknown dataset {dataset!r}; this model holds {sorted(self.shapes)}"
            )
        return dataset

    def features(self, x: torch.Tensor, dataset: str | None = None) -> torch.Tensor:
        """The trunk output: the representation the head reads."""
        dataset = self._dataset(dataset)
        shared = self.config.adapters == "shared"
        hidden = self.adapter(x) if shared else self.adapter[dataset](x)
        return (
            self.trunk(hidden)
            if self.config.trunk == "shared"
            else self.trunk[dataset](hidden)
        )

    def forward(self, x: torch.Tensor, dataset: str | None = None) -> torch.Tensor:
        dataset = self._dataset(dataset)
        head = self.head[self._head_key(self.shapes[dataset])]
        return head(self.features(x, dataset))


AUX = "/"


def is_aux(key: str) -> bool:
    """An auxiliary array ``<algorithm>/<parameter key>`` (a control variate, …)."""
    return AUX in key


def group_of(key: str) -> str:
    """Parameter group of a key: ``adapter.<dataset>``, ``trunk``, ``trunk.<dataset>``, ``head.<task>``.

    An auxiliary key belongs to the group of the parameter it names.
    """
    parts = key.split(AUX, 1)[-1].split(".")
    if len(parts) < 2 or parts[0] not in NAMESPACES:
        raise ValueError(f"key {key!r} is outside the {'/'.join(NAMESPACES)} namespace")
    if parts[0] in ("adapter", "trunk") and parts[1][0].isdigit():
        return parts[0]  # shared adapter or trunk: no dataset in the key
    return f"{parts[0]}.{parts[1]}"


def param_groups(keys: Iterable[str]) -> dict[str, list[str]]:
    """Keys grouped by ``group_of``, in their original order."""
    grouped: dict[str, list[str]] = {}
    for key in keys:
        grouped.setdefault(group_of(key), []).append(key)
    return grouped


def state_arrays(model: nn.Module) -> dict[str, np.ndarray]:
    """The model state as NumPy copies, ready for a ``Payload``."""
    return {
        key: value.detach().cpu().numpy().copy()
        for key, value in model.state_dict().items()
    }


def load_arrays(model: nn.Module, arrays: Mapping[str, np.ndarray]) -> list[str]:
    """Load the keys the model has; ignore the rest. Returns the loaded keys."""
    state = model.state_dict()
    updates = {}
    for key, value in arrays.items():
        if key not in state:
            continue
        if tuple(np.shape(value)) != tuple(state[key].shape):
            raise ValueError(
                f"{key}: shape {tuple(np.shape(value))} does not match the model's "
                f"{tuple(state[key].shape)}"
            )
        updates[key] = torch.as_tensor(np.asarray(value), dtype=state[key].dtype)
    model.load_state_dict(updates, strict=False)
    return list(updates)


models = Registry("model")


@models.register(
    "modular_mlp",
    title="MLP modular",
    description="Adaptador por dataset, tronco común o por dataset y cabeza por tarea o por dataset.",
    params=ModularMLPConfig,
    explain=(
        "Cada edge solo instancia sus partes. La compartición decide qué grupos "
        "(adapter, trunk, head) viajan y hasta qué nivel se agregan."
    ),
)
class ModularMLPFamily:
    """Factory registered as ``modular_mlp``: holds the config, builds per node."""

    def __init__(self, **config: object) -> None:
        self.config = ModularMLPConfig(**config)

    def build(self, shapes: Sequence[DataShape], seed: int = 0) -> ModularMLP:
        return ModularMLP(self.config, shapes, seed=seed)
