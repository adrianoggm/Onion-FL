from __future__ import annotations

"""Presets of ``torch_module``: models of the FL literature on the key contract.

Each one keeps the paper's layers and splits them into the three parts: the
first layer is the dataset's adapter (it fits the input's channels or
vocabulary), the rest of the body is the shared trunk and the last linear layer
is the task's head. They take ``(shapes, input_shape, **params)`` like any
external class and derive from ``ModularModule``, which routes each dataset.

- ``LeafFemnistCNN``: LEAF's FEMNIST CNN (Caldas et al., 2018), two 5x5
  convolutions of 32 and 64 filters with 2x2 max pooling, a dense layer of 2048.
- ``NiidBenchCNN``: NIID-Bench's simple CNN for CIFAR-10 (Li et al., 2022),
  convolutions of 6 and 16 5x5 filters with pooling, dense layers of 120 and 84.
- ``LeafShakespeareLSTM``: LEAF's next-character model, an 8-wide embedding of
  80 symbols and two LSTM layers of 256 units, read at the last step.
- ``AudioCNN1d``: a small 1D CNN for raw audio or feature sequences (C, L),
  three convolutions with pooling and a global average.
"""

from collections.abc import Sequence
from typing import Literal

import torch
from pydantic import BaseModel, ConfigDict
from torch import nn

from onion_fl.learning.model import DataShape, ModularModule


class Layout(BaseModel):
    """The presets' parts: an adapter per dataset, a shared trunk, a head per task."""

    model_config = ConfigDict(frozen=True)

    adapters: Literal["per_dataset"] = "per_dataset"
    trunk: Literal["shared"] = "shared"
    heads: Literal["per_task"] = "per_task"


class _Preset(ModularModule):
    def __init__(self, shapes: Sequence[DataShape]) -> None:
        super().__init__(Layout(), shapes)

    def _parts(self, adapter, trunk: nn.Module, width: int) -> None:
        self.adapter = nn.ModuleDict({d: adapter() for d in self.shapes})
        self.trunk = trunk
        self.head = nn.ModuleDict(
            {key: nn.Linear(width, n) for key, n in self.classes.items()}
        )


def _image(input_shape: Sequence[int], name: str) -> tuple[int, int, int]:
    if len(input_shape) != 3:
        raise ValueError(f"{name} needs input_shape (C, H, W), got {input_shape}")
    c, h, w = input_shape
    return c, h, w


class LeafFemnistCNN(_Preset):
    """LEAF's FEMNIST CNN: conv 5x5/32, pool, conv 5x5/64, pool, dense 2048."""

    input_shape = (1, 28, 28)

    def __init__(
        self,
        shapes: Sequence[DataShape],
        input_shape: Sequence[int],
        filters: Sequence[int] = (32, 64),
        hidden: int = 2048,
    ) -> None:
        super().__init__(shapes)
        c, h, w = _image(input_shape, "leaf_femnist_cnn")
        first, second = filters
        if h % 4 or w % 4:
            raise ValueError(
                f"leaf_femnist_cnn needs H and W divisible by 4, got {h}x{w}"
            )

        def adapter() -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(c, first, 5, padding=2), nn.ReLU(), nn.MaxPool2d(2)
            )

        trunk = nn.Sequential(
            nn.Conv2d(first, second, 5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(second * (h // 4) * (w // 4), hidden),
            nn.ReLU(),
        )
        self._parts(adapter, trunk, hidden)


class NiidBenchCNN(_Preset):
    """NIID-Bench's simple CNN: conv 5x5/6, pool, conv 5x5/16, pool, dense 120, 84."""

    input_shape = (3, 32, 32)

    def __init__(
        self,
        shapes: Sequence[DataShape],
        input_shape: Sequence[int],
        hidden: Sequence[int] = (120, 84),
    ) -> None:
        super().__init__(shapes)
        c, h, w = _image(input_shape, "niid_bench_cnn")
        side_h, side_w = ((h - 4) // 2 - 4) // 2, ((w - 4) // 2 - 4) // 2
        if side_h < 1 or side_w < 1:
            raise ValueError(f"niid_bench_cnn needs at least 16x16 inputs, got {h}x{w}")

        def adapter() -> nn.Sequential:
            return nn.Sequential(nn.Conv2d(c, 6, 5), nn.ReLU(), nn.MaxPool2d(2))

        layers: list[nn.Module] = [
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
        ]
        dims = [16 * side_h * side_w, *hidden]
        for n_in, n_out in zip(dims, dims[1:], strict=False):
            layers += [nn.Linear(n_in, n_out), nn.ReLU()]
        self._parts(adapter, nn.Sequential(*layers), dims[-1])


class Tokens(nn.Module):
    """Symbol indices, carried as float columns, to their embeddings."""

    def __init__(self, vocabulary: int, width: int) -> None:
        super().__init__()
        self.vocabulary = vocabulary
        self.embedding = nn.Embedding(vocabulary, width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embedding(x.long().clamp_(0, self.vocabulary - 1))


class LastStep(nn.Module):
    """A stacked LSTM read at the last step of the sequence."""

    def __init__(self, n_in: int, hidden: int, layers: int) -> None:
        super().__init__()
        self.lstm = nn.LSTM(n_in, hidden, num_layers=layers, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return out[:, -1]


class LeafShakespeareLSTM(_Preset):
    """LEAF's Shakespeare model: embedding 8 of 80 symbols, 2 LSTM layers of 256."""

    input_shape = (80,)

    def __init__(
        self,
        shapes: Sequence[DataShape],
        input_shape: Sequence[int],
        vocabulary: int = 80,
        embedding: int = 8,
        hidden: int = 256,
        layers: int = 2,
    ) -> None:
        super().__init__(shapes)
        if len(input_shape) != 1:
            raise ValueError(
                f"leaf_shakespeare_lstm needs input_shape (L,), got {input_shape}"
            )
        trunk = nn.Sequential(LastStep(embedding, hidden, layers))
        self._parts(lambda: nn.Sequential(Tokens(vocabulary, embedding)), trunk, hidden)


class AudioCNN1d(_Preset):
    """A small 1D CNN on (C, L): three conv/ReLU/pool blocks and a global average."""

    def __init__(
        self,
        shapes: Sequence[DataShape],
        input_shape: Sequence[int],
        channels: Sequence[int] = (16, 32, 64),
        kernel_size: int = 9,
        pool: int = 4,
    ) -> None:
        super().__init__(shapes)
        if len(input_shape) != 2:
            raise ValueError(f"audio_cnn1d needs input_shape (C, L), got {input_shape}")
        if len(channels) < 2:
            raise ValueError("audio_cnn1d needs at least two channel widths")
        c, pad = input_shape[0], kernel_size // 2

        def block(n_in: int, n_out: int) -> list[nn.Module]:
            return [nn.Conv1d(n_in, n_out, kernel_size, padding=pad), nn.ReLU()]

        def adapter() -> nn.Sequential:
            return nn.Sequential(*block(c, channels[0]), nn.MaxPool1d(pool))

        layers: list[nn.Module] = []
        for i, (n_in, n_out) in enumerate(zip(channels, channels[1:], strict=False)):
            layers += block(n_in, n_out)
            last = i == len(channels) - 2
            layers.append(nn.AdaptiveAvgPool1d(1) if last else nn.MaxPool1d(pool))
        layers.append(nn.Flatten())
        self._parts(adapter, nn.Sequential(*layers), channels[-1])
