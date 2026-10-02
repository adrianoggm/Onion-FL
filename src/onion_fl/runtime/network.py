from __future__ import annotations

"""Simulated links: profiles, presets and the per-direction channel (spec §9.1).

In simulation the "in-memory transport" is this link layer: a message takes
``queueing + bytes * 8 / bandwidth + latency`` virtual seconds, or is lost.
"""

import math
from collections.abc import Mapping
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from onion_fl.core.registry import Registry

Direction = Literal["up", "down"]


class LinkProfile(BaseModel):
    """Network behaviour of a link. ``up`` is child -> parent, ``down`` parent -> child."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    latency_s: float = Field(0.0, ge=0, description="Mean one-way propagation latency")
    jitter_s: float = Field(0.0, ge=0, description="Standard deviation of the latency")
    distribution: Literal["fixed", "normal", "lognormal"] = "fixed"
    bandwidth_up_bps: float | None = Field(
        None, gt=0, description="None means unlimited"
    )
    bandwidth_down_bps: float | None = Field(
        None, gt=0, description="None means unlimited"
    )
    loss: float = Field(0.0, ge=0, le=1, description="Probability of losing a message")

    def sample_latency(self, rng: np.random.Generator) -> float:
        mean, std = self.latency_s, self.jitter_s
        if self.distribution == "fixed" or std == 0 or mean == 0:
            return mean
        if self.distribution == "normal":
            return max(0.0, float(rng.normal(mean, std)))
        # lognormal with the requested mean and standard deviation of the delay itself
        sigma2 = math.log1p((std / mean) ** 2)
        return float(rng.lognormal(math.log(mean) - sigma2 / 2, math.sqrt(sigma2)))

    def transmission_s(self, size_bytes: int, direction: Direction) -> float:
        bandwidth = (
            self.bandwidth_up_bps if direction == "up" else self.bandwidth_down_bps
        )
        return 0.0 if bandwidth is None else size_bytes * 8 / bandwidth


link_profiles = Registry("link_profile")

_PRESETS: dict[str, tuple[str, str, dict[str, Any]]] = {
    "lan": (
        "LAN cableada",
        "1 ms de latencia y 1 Gbps en ambos sentidos, sin pérdidas.",
        {
            "latency_s": 0.001,
            "jitter_s": 0.0002,
            "distribution": "normal",
            "bandwidth_up_bps": 1e9,
            "bandwidth_down_bps": 1e9,
        },
    ),
    "wifi": (
        "Wi-Fi",
        "5 ms ± 2 ms, 100 Mbps de bajada y 50 de subida, 0,5 % de pérdidas.",
        {
            "latency_s": 0.005,
            "jitter_s": 0.002,
            "distribution": "normal",
            "bandwidth_up_bps": 50e6,
            "bandwidth_down_bps": 100e6,
            "loss": 0.005,
        },
    ),
    "4g": (
        "4G móvil",
        "50 ms ± 15 ms (lognormal), 20 Mbps de bajada y 5 de subida, 1 % de pérdidas.",
        {
            "latency_s": 0.05,
            "jitter_s": 0.015,
            "distribution": "lognormal",
            "bandwidth_up_bps": 5e6,
            "bandwidth_down_bps": 20e6,
            "loss": 0.01,
        },
    ),
    "lora": (
        "LoRa",
        "1 s ± 0,3 s, 5 kbps en ambos sentidos, 5 % de pérdidas.",
        {
            "latency_s": 1.0,
            "jitter_s": 0.3,
            "distribution": "normal",
            "bandwidth_up_bps": 5e3,
            "bandwidth_down_bps": 5e3,
            "loss": 0.05,
        },
    ),
}

for _name, (_title, _description, _fields) in _PRESETS.items():
    link_profiles.register(
        _name,
        title=_title,
        description=_description,
        explain="Valores orientativos: ajústalos por enlace con {'preset': ..., <campo>: <valor>}.",
    )(lambda fields=_fields: LinkProfile(**fields))


def resolve_profile(spec: str | Mapping[str, Any] | LinkProfile) -> LinkProfile:
    """A preset name, a preset with overrides (``{"preset": "4g", "loss": 0.05}``) or a full profile."""
    if isinstance(spec, LinkProfile):
        return spec
    if isinstance(spec, str):
        return link_profiles.create(spec)
    fields = dict(spec)
    preset = fields.pop("preset", None)
    if preset is None:
        return LinkProfile(**fields)
    return LinkProfile(**(link_profiles.create(preset).model_dump() | fields))


class LinkChannel:
    """One direction of a link: FIFO, with transmission queueing and losses.

    A message starts transmitting when the channel is free, takes
    ``bytes * 8 / bandwidth`` seconds, then the propagation latency, and never
    overtakes the previous one. Lost messages still occupied the channel.
    """

    def __init__(
        self, profile: LinkProfile, direction: Direction, rng: np.random.Generator
    ) -> None:
        self.profile = profile
        self.direction = direction
        self._rng = rng
        self._free_at = 0.0
        self._last_arrival = 0.0

    def schedule(self, t_send: float, size_bytes: int) -> float | None:
        """Arrival time of a message sent at ``t_send``, or None if it is lost."""
        start = max(t_send, self._free_at)
        self._free_at = start + self.profile.transmission_s(size_bytes, self.direction)
        lost = (
            self._rng.random() < self.profile.loss
        )  # drawn every time: streams stay aligned
        arrival = max(
            self._free_at + self.profile.sample_latency(self._rng), self._last_arrival
        )
        if lost:
            return None
        self._last_arrival = arrival
        return arrival
