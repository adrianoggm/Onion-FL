from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class SwellMLP(nn.Module):
    """Simple MLP for SWELL tabular features (binary stress classification).

    Args:
        input_dim: Number of input features
        hidden_dims: Hidden layer sizes
        num_classes: Output classes (2 for binary)
    """

    def __init__(
        self, input_dim: int, hidden_dims: list[int] | None = None, num_classes: int = 2
    ) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 64]

        dims = [input_dim] + hidden_dims + [num_classes]
        layers: list[nn.Module] = []
        for i in range(len(dims) - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def get_parameters(model: nn.Module) -> list[np.ndarray]:
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


def set_parameters(model: nn.Module, parameters) -> None:
    state_dict = model.state_dict()
    for (key, _), param in zip(state_dict.items(), parameters):
        if isinstance(param, torch.Tensor):
            state_dict[key] = param
        else:
            state_dict[key] = torch.tensor(param)
    model.load_state_dict(state_dict)
