from __future__ import annotations

"""Transport-agnostic messages exchanged between nodes (spec §5.1).

A ``Message`` is what a node hands to ``Context.send``; a codec turns it into
bytes for a link. It replaces the MQTT-specific envelopes of the old runtime
(``runtime_protocol.py``, removed in F9.1):

- ``GlobalModelEnvelope``      -> ``kind="global_model"``: ``payload.state`` is
  the model, ``meta["trace_context"]`` the trace.
- ``ClientUpdateEnvelope``     -> ``kind="update"`` from edge to fog: ``num_samples``
  becomes the weight of every key, ``loss``/``val_acc`` go to ``payload.metrics``,
  ``sent_at`` and ``trace_context`` to ``meta``.
- ``PartialAggregateEnvelope`` -> ``kind="update"`` from fog to its parent: the
  weights carry the summed samples per key, the staleness counters are metrics.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from numbers import Integral, Real
from typing import Any

import numpy as np

KINDS = (
    "hello",
    "global_model",
    "update",
    "eval_request",
    "eval_report",
    "control",
    "status",  # counts and drift statistics, never data (continuum C6)
)


class MessageError(ValueError):
    """A message, or its encoded form, is malformed."""


def _is_number(value: Any) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool)


@dataclass(frozen=True, eq=False)
class Payload:
    """Model state by key, samples per key and scalar metrics."""

    state: Mapping[str, np.ndarray] = field(default_factory=dict)
    weights: Mapping[str, float] = field(default_factory=dict)
    metrics: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for key, value in self.state.items():
            if not isinstance(value, np.ndarray) or value.dtype.kind not in "biuf":
                raise MessageError(f"state[{key!r}] must be a numeric numpy array")
        for key, value in self.weights.items():
            if key not in self.state:
                raise MessageError(f"weights[{key!r}] has no matching state key")
            if not _is_number(value) or value < 0:
                raise MessageError(f"weights[{key!r}] must be a non-negative number")
        for key, value in self.metrics.items():
            if not _is_number(value):
                raise MessageError(f"metrics[{key!r}] must be a number")

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Payload):
            return NotImplemented
        if self.state.keys() != other.state.keys():
            return False
        for key, array in self.state.items():
            twin = other.state[key]
            same = array.dtype == twin.dtype and array.shape == twin.shape
            if not (same and np.array_equal(array, twin)):
                return False
        return dict(self.weights) == dict(other.weights) and dict(self.metrics) == dict(
            other.metrics
        )

    __hash__ = None  # type: ignore[assignment]


@dataclass(frozen=True)
class Message:
    """One message between two nodes. Validated on construction."""

    kind: str
    src: str
    dst: str
    round: int | None = None
    payload: Payload = field(default_factory=Payload)
    meta: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise MessageError(f"kind must be one of {KINDS}, got {self.kind!r}")
        for name in ("src", "dst"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise MessageError(f"{name} must be a non-empty node id")
        if self.round is not None and (
            isinstance(self.round, bool)
            or not isinstance(self.round, Integral)
            or self.round < 0
        ):
            raise MessageError(
                f"round must be a non-negative int or None, got {self.round!r}"
            )
        if not isinstance(self.payload, Payload):
            raise MessageError("payload must be a Payload")
        if not isinstance(self.meta, Mapping):
            raise MessageError("meta must be a mapping")
