from __future__ import annotations

"""Device models for simulation: compute time and availability (spec §9.1)."""

from pydantic import BaseModel, Field, field_validator

from onion_fl.core.context import node_rng
from onion_fl.core.registry import Registry

compute_models = Registry("compute")
availability_models = Registry("availability")


# --- compute -----------------------------------------------------------------


class SamplesPerSecondParams(BaseModel):
    samples_per_second: float = Field(
        gt=0, description="Muestras procesadas por segundo"
    )
    overhead_s: float = Field(
        0.0, ge=0, description="Coste fijo de cada tarea con trabajo"
    )


@compute_models.register(
    "samples_per_second",
    title="Muestras por segundo",
    description="Duración = overhead + muestras declaradas / muestras_por_segundo.",
    params=SamplesPerSecondParams,
    explain="Determinista: permite simular dispositivos lentos y rezagados sin depender de esta máquina.",
)
class SamplesPerSecond:
    def __init__(self, samples_per_second: float, overhead_s: float = 0.0) -> None:
        self.samples_per_second = samples_per_second
        self.overhead_s = overhead_s

    def duration(self, samples: float, wall_s: float) -> float:
        if samples <= 0:
            return 0.0
        return self.overhead_s + samples / self.samples_per_second


class MeasuredParams(BaseModel):
    factor: float = Field(1.0, gt=0, description="Multiplicador del tiempo de pared")


@compute_models.register(
    "measured",
    title="Tiempo medido",
    description="Duración = tiempo de pared real del manejador × factor.",
    params=MeasuredParams,
    explain="Más realista, pero no determinista: dos ejecuciones con la misma semilla pueden diferir.",
)
class Measured:
    def __init__(self, factor: float = 1.0) -> None:
        self.factor = factor

    def duration(self, samples: float, wall_s: float) -> float:
        return wall_s * self.factor


# --- availability ------------------------------------------------------------


@availability_models.register(
    "always", title="Siempre disponible", description="El nodo nunca se cae."
)
class Always:
    def is_up(self, t: float, round: int | None, node_id: str, seed: int) -> bool:
        return True


class BernoulliParams(BaseModel):
    p: float = Field(
        ge=0, le=1, description="Probabilidad de estar desconectado en una ronda"
    )


@availability_models.register(
    "bernoulli",
    title="Bernoulli por ronda",
    description="En cada ronda el nodo está desconectado con probabilidad p.",
    params=BernoulliParams,
    explain="Solo afecta a mensajes de una ronda; hello y control siempre llegan.",
)
class Bernoulli:
    def __init__(self, p: float) -> None:
        self.p = p

    def is_up(self, t: float, round: int | None, node_id: str, seed: int) -> bool:
        if round is None:
            return True
        return node_rng(seed, f"{node_id}/availability/{round}").random() >= self.p


class ScheduleParams(BaseModel):
    offline: list[tuple[float, float]] = Field(
        description="Ventanas [inicio, fin) sin conexión"
    )

    @field_validator("offline")
    @classmethod
    def _ordered(cls, windows: list[tuple[float, float]]) -> list[tuple[float, float]]:
        for start, end in windows:
            if not 0 <= start < end:
                raise ValueError(
                    f"window [{start}, {end}) must satisfy 0 <= start < end"
                )
        return windows


@availability_models.register(
    "schedule",
    title="Calendario",
    description="Desconectado dentro de las ventanas [inicio, fin) indicadas.",
    params=ScheduleParams,
)
class Schedule:
    def __init__(self, offline: list[tuple[float, float]]) -> None:
        self.offline = offline

    def is_up(self, t: float, round: int | None, node_id: str, seed: int) -> bool:
        return not any(start <= t < end for start, end in self.offline)

    def up_again(self, t: float) -> float:
        """When a node offline at ``t`` is up again; windows may touch."""
        while not self.is_up(t, None, "", 0):
            t = max(end for start, end in self.offline if start <= t < end)
        return t


class CrashAtParams(BaseModel):
    t: float = Field(ge=0, description="Instante virtual de la caída")


@availability_models.register(
    "crash_at",
    title="Caída definitiva",
    description="Disponible hasta el instante t; después, caído para siempre.",
    params=CrashAtParams,
)
class CrashAt:
    def __init__(self, t: float) -> None:
        self.t = t

    def is_up(self, t: float, round: int | None, node_id: str, seed: int) -> bool:
        return t < self.t
