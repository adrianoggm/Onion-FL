from __future__ import annotations

"""``torch_module``: any PyTorch module that keeps the key contract (issue #245).

The family takes a preset name or a ``package.module:Class`` reference and its
parameters, and builds the class as::

    Class(shapes, input_shape, **params)

``shapes`` are the ``DataShape`` of the datasets this node holds and
``input_shape`` is the shape of one example, ``(C, H, W)`` or ``(L,)``. The
module receives ``x`` of shape ``(batch, *input_shape)``; its ``forward`` may
take ``dataset=`` to route several datasets, as ``ModularModule`` does.

The framework adds what every family needs and a foreign class does not know:

- **Input.** The edge sees the columns preprocessing kept. Each ``x_<i>`` goes
  back to position ``i`` of the flattened example, and the columns dropped as
  constant are filled with 0 (their standardised value).
- **Keys.** Every ``state_dict`` key must fall in ``adapter.<ds>``, ``trunk``,
  ``trunk.<ds>`` or ``head.<task>``; a shared trunk is an ``nn.Sequential``, so
  its keys are ``trunk.<i>.*``. The build fails with the offending keys.
- **Initialisation.** ``init_parameters(seed)`` resets each submodule from
  (seed, module path), as the other families do, unless the class defines its
  own.

A trial forward pass on a zero batch checks that every dataset's input fits and
that the output has one logit per class.

**Trust.** Importing a module runs its code, and configs also arrive over HTTP
(the Studio validates them). So validating a config never imports anything: a
reference is checked as text, and its class is imported only when the model is
built. A preset always resolves; an external class only when its module is
under a prefix the operator lists in ``ONIONFL_TRUSTED_MODULES`` (comma
separated, empty by default), e.g. ``ONIONFL_TRUSTED_MODULES=mylab.models``.
"""

import importlib
import inspect
import math
import re
from collections.abc import Sequence
from typing import Any

import torch
from pydantic import BaseModel, ConfigDict, Field, PositiveInt, model_validator
from torch import nn

from onion_fl.core.registry import TRUSTED_ENV, is_trusted
from onion_fl.learning.model import DataShape, _seed_for, group_of, models

COLUMN = re.compile(r"^x_(\d+)$")
REFERENCE = re.compile(r"^[A-Za-z_]\w*(\.[A-Za-z_]\w*)*:[A-Za-z_]\w*$")

# Preset name -> (class reference, default input shape). They are plain
# ``pkg.mod:Class`` references: a preset goes through the same path as any
# external class.
PRESETS: dict[str, tuple[str, tuple[int, ...] | None]] = {
    "leaf_femnist_cnn": ("onion_fl.learning.presets:LeafFemnistCNN", (1, 28, 28)),
    "niid_bench_cnn": ("onion_fl.learning.presets:NiidBenchCNN", (3, 32, 32)),
    "leaf_shakespeare_lstm": ("onion_fl.learning.presets:LeafShakespeareLSTM", (80,)),
    "audio_cnn1d": ("onion_fl.learning.presets:AudioCNN1d", None),
}


class TorchModuleError(ValueError):
    """A module class, its parameters or its keys do not fit the contract."""


def check_reference(ref: str) -> None:
    """A preset, or a well-formed ``pkg.mod:Class`` under a trusted prefix; no import."""
    if ref in PRESETS:
        return
    if not REFERENCE.match(ref):
        raise TorchModuleError(
            f"unknown module {ref!r}: use a preset ({', '.join(sorted(PRESETS))}) "
            "or 'package.module:Class'"
        )
    if not is_trusted(ref.partition(":")[0]):
        raise TorchModuleError(
            f"external module {ref!r} is not trusted: the operator must list its "
            f"package in {TRUSTED_ENV} (comma separated) before it can be imported"
        )


def resolve_class(ref: str) -> tuple[type[nn.Module], tuple[int, ...] | None]:
    """The class of a preset or trusted ``pkg.mod:Class``, and its default input.

    This is the only place that imports, and it checks the trust first.
    """
    check_reference(ref)
    path, default = PRESETS.get(ref, (ref, None))
    module_name, _, attr = path.partition(":")
    try:
        cls = getattr(importlib.import_module(module_name), attr)
    except (ImportError, AttributeError) as exc:
        raise TorchModuleError(f"cannot import module {path!r}: {exc}") from exc
    if not (isinstance(cls, type) and issubclass(cls, nn.Module)):
        raise TorchModuleError(f"{path!r} is not a torch.nn.Module subclass")
    if default is None:
        given = getattr(cls, "input_shape", None)
        default = None if given is None else tuple(int(n) for n in given)
    return cls, default


class TorchModuleConfig(BaseModel):
    """Configuration of ``torch_module``."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    module: str = Field(
        description=(
            "Preset (leaf_femnist_cnn, niid_bench_cnn, leaf_shakespeare_lstm, "
            "audio_cnn1d) o clase externa 'paquete.modulo:Clase'"
        ),
    )
    params: dict[str, Any] = Field(
        default_factory=dict, description="Parámetros con los que se construye la clase"
    )
    input_shape: list[PositiveInt] | None = Field(
        None,
        description=(
            "Forma de un ejemplo, (C, H, W) o (L,), reconstruida desde las columnas "
            "x_i; sin ella, la del preset o el atributo input_shape de la clase"
        ),
    )

    @model_validator(mode="after")
    def _a_known_or_trusted_module(self) -> TorchModuleConfig:
        check_reference(self.module)  # text only: validating never imports
        if self.module in PRESETS:  # the framework's own code: checked in full
            self.resolved()
        return self

    def resolved(self) -> tuple[type[nn.Module], tuple[int, ...]]:
        """Import the class (trusted only), check its params and the input shape."""
        cls, default = resolve_class(self.module)
        shape = tuple(self.input_shape) if self.input_shape is not None else default
        if shape is None:
            raise TorchModuleError(
                f"{self.module}: no default input_shape; set input_shape, e.g. [1, 16000]"
            )
        try:
            inspect.signature(cls).bind(None, None, **self.params)
        except TypeError as exc:
            raise TorchModuleError(
                f"{self.module}: invalid params {self.params}: {exc}"
            ) from None
        return cls, shape


def columns_of(shape: DataShape, size: int) -> list[int] | None:
    """Where each column goes in the flattened example; None when it is the identity."""
    if shape.features is None:
        if shape.n_features != size:
            raise TorchModuleError(
                f"{shape.dataset}: {shape.n_features} features do not fill an input "
                f"of {size} values, and without column names they cannot be placed"
            )
        return None
    index = []
    for name in shape.features:
        match = COLUMN.match(name)
        if match is None:
            raise TorchModuleError(
                f"{shape.dataset}: torch_module rebuilds the input from x_<i> "
                f"columns, got {name!r}"
            )
        index.append(int(match.group(1)))
    outside = [i for i in index if i >= size]
    if outside:
        raise TorchModuleError(
            f"{shape.dataset}: columns x_{outside[0]}… fall outside an input of "
            f"{size} values"
        )
    if len(set(index)) != len(index):
        raise TorchModuleError(f"{shape.dataset}: repeated x_<i> columns")
    return None if index == list(range(size)) else index


def check_keys(module: nn.Module, shapes: Sequence[DataShape]) -> None:
    """Every key is in the contract, and the dataset or task it names exists."""
    datasets = {s.dataset for s in shapes}
    names = {"adapter": datasets, "trunk": datasets, "head": datasets}
    names["head"] = datasets | {s.task for s in shapes}
    wrong = []
    for key in module.state_dict():
        try:
            group = group_of(key)
        except ValueError:
            wrong.append(f"{key} (outside adapter/trunk/head)")
            continue
        space, _, name = group.partition(".")
        if name and name not in names[space]:
            wrong.append(
                f"{key} (group {group!r}: {name!r} is no dataset"
                + (" or task" if space == "head" else "")
                + "; a shared part is an nn.Sequential, keys <part>.<i>.*)"
            )
    if wrong:
        raise TorchModuleError(
            f"{type(module).__name__}: keys outside the contract: {wrong[:5]}"
            + (f" and {len(wrong) - 5} more" if len(wrong) > 5 else "")
        )


class TorchModule(nn.Module):
    """A foreign module behind the framework's input, keys and initialisation.

    The inner module's parts (``adapter``, ``trunk``, ``head``) are registered
    on this one under the same names, so the keys do not gain a prefix; the
    inner module itself is kept apart to call its ``forward``.
    """

    def __init__(
        self,
        inner: nn.Module,
        shapes: Sequence[DataShape],
        input_shape: tuple[int, ...],
    ) -> None:
        super().__init__()
        own = [*inner._parameters, *inner._buffers]
        if own:
            raise TorchModuleError(
                f"{type(inner).__name__}: tensors {own} are outside adapter/trunk/head"
            )
        object.__setattr__(self, "inner", inner)  # not a submodule: no key prefix
        for name, child in inner.named_children():
            self.add_module(name, child)
        self.shapes = {s.dataset: s for s in shapes}
        self.input_shape = tuple(input_shape)
        size = math.prod(self.input_shape)
        self.columns = {
            s.dataset: (None if (c := columns_of(s, size)) is None else torch.tensor(c))
            for s in shapes
        }
        self.routes = "dataset" in inspect.signature(inner.forward).parameters
        check_keys(self, shapes)

    def train(self, mode: bool = True) -> TorchModule:
        super().train(mode)
        self.inner.training = mode
        return self

    def _dataset(self, dataset: str | None) -> str:
        if dataset is None:
            if len(self.shapes) != 1:
                raise ValueError(
                    f"this model holds datasets {sorted(self.shapes)}; pass dataset=..."
                )
            return next(iter(self.shapes))
        if dataset not in self.shapes:
            raise ValueError(
                f"unknown dataset {dataset!r}; this model holds {sorted(self.shapes)}"
            )
        return dataset

    def example(self, x: torch.Tensor, dataset: str | None = None) -> torch.Tensor:
        """The columns as examples of ``input_shape``: ``(batch, *input_shape)``."""
        columns = self.columns[self._dataset(dataset)]
        if columns is not None:
            full = x.new_zeros((x.shape[0], math.prod(self.input_shape)))
            full[:, columns] = x
            x = full
        return x.reshape(x.shape[0], *self.input_shape)

    def _call(self, fn: Any, x: torch.Tensor, dataset: str | None) -> torch.Tensor:
        dataset = self._dataset(dataset)
        x = self.example(x, dataset)
        return fn(x, dataset=dataset) if self.routes else fn(x)

    def forward(self, x: torch.Tensor, dataset: str | None = None) -> torch.Tensor:
        return self._call(self.inner, x, dataset)

    def features(self, x: torch.Tensor, dataset: str | None = None) -> torch.Tensor:
        """The representation the head reads, when the module defines ``features``."""
        fn = getattr(self.inner, "features", None)
        if fn is None:
            raise TorchModuleError(
                f"{type(self.inner).__name__} defines no features(x); this trainer "
                "needs it"
            )
        return self._call(fn, x, dataset)

    def init_parameters(self, seed: int) -> None:
        own = getattr(self.inner, "init_parameters", None)
        if own is not None:
            own(seed)
            return
        # Each submodule draws from torch's generator seeded by (seed, path): a
        # part starts identical in an edge model and in the global model.
        for name, module in self.named_modules():
            reset = getattr(module, "reset_parameters", None)
            if name and callable(reset):
                with torch.random.fork_rng(devices=[]):
                    torch.manual_seed(_seed_for(seed, name))
                    reset()

    def check_output(self) -> None:
        """A zero batch through every dataset gives one logit per class."""
        was = self.training
        self.eval()
        try:
            with torch.no_grad():
                for name, shape in self.shapes.items():
                    x = torch.zeros((2, shape.n_features))
                    try:
                        out = self(x, dataset=name)
                    except Exception as exc:  # noqa: BLE001 - re-raised with context
                        raise TorchModuleError(
                            f"{type(self.inner).__name__} fails on {name}'s input "
                            f"{(2, *self.input_shape)}: {exc}"
                        ) from exc
                    if tuple(out.shape) != (2, shape.n_classes):
                        raise TorchModuleError(
                            f"{type(self.inner).__name__} gives {tuple(out.shape)} on "
                            f"{name}, expected (batch, {shape.n_classes})"
                        )
        finally:
            self.train(was)


@models.register(
    "torch_module",
    title="Módulo PyTorch",
    description=(
        "Cualquier nn.Module (preset o 'paquete.modulo:Clase') que respete las "
        "claves adapter.<ds>, trunk y head.<tarea>."
    ),
    params=TorchModuleConfig,
    explain=(
        "La forma de entrada, (C, H, W) o una secuencia, se reconstruye desde las "
        "columnas x_i; las que el preprocesado quitó por constantes valen 0. Presets: "
        "la CNN de LEAF para FEMNIST, la CNN simple de NIID-Bench para CIFAR-10, la "
        "LSTM de dos capas de LEAF para Shakespeare y una CNN 1D pequeña para audio. "
        "Una clase externa se construye como Clase(shapes, input_shape, **params) "
        "y solo se importa si su paquete está en ONIONFL_TRUSTED_MODULES."
    ),
)
class TorchModuleFamily:
    """Factory registered as ``torch_module``: holds the config, builds per node."""

    def __init__(self, **config: object) -> None:
        self.config = TorchModuleConfig(**config)
        self._resolved: tuple[type[nn.Module], tuple[int, ...]] | None = None

    def resolved(self) -> tuple[type[nn.Module], tuple[int, ...]]:
        """The class and input shape, imported on first use (a build, not a config)."""
        if self._resolved is None:
            self._resolved = self.config.resolved()
        return self._resolved

    @property
    def n_inputs(self) -> int:
        """Values in one example: the features a preview without data assumes."""
        return math.prod(self.resolved()[1])

    def build(self, shapes: Sequence[DataShape], seed: int = 0) -> TorchModule:
        cls, input_shape = self.resolved()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(_seed_for(seed, "torch_module"))
            inner = cls(list(shapes), input_shape, **self.config.params)
        model = TorchModule(inner, shapes, input_shape)
        model.init_parameters(seed)
        model.check_output()
        return model
