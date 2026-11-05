from __future__ import annotations

"""When a challenger becomes the champion (spec §5.2): promotion policies as plugins.

A policy gets the scores the validators reported for the champion and the
challenger on the recent window and on the reference, and answers with a
``Decision``. Every threshold is a parameter.
"""

from pydantic import BaseModel, Field, NonNegativeFloat, NonNegativeInt

from onion_fl.continuum.versions import MODELS, Decision, Scores
from onion_fl.core.registry import Registry
from onion_fl.data.stream import Positive

promotions = Registry("promotion")


class ChampionChallengerParams(BaseModel):
    metric: str = Field(
        "macro_f1", description="Métrica de evaluación con la que se compara"
    )
    min_delta: float = Field(
        -0.01,
        description="Lo más que el challenger puede quedar por debajo del champion "
        "en la ventana reciente",
    )
    max_forgetting: NonNegativeFloat = Field(
        0.03,
        description="Lo más que el challenger puede caer respecto al champion en la "
        "referencia",
    )
    min_samples: NonNegativeInt = Field(
        500, description="Etiquetas mínimas en la ventana reciente para decidir"
    )
    window: Positive | None = Field(
        None,
        description="Ventana reciente en tiempo de datos; sin ella, lo llegado desde "
        "que se promovió el champion",
    )


@promotions.register(
    "champion_challenger",
    title="Champion y challenger",
    description="Promueve el challenger si no empeora lo reciente más de min_delta "
    "ni olvida más de max_forgetting en la referencia.",
    params=ChampionChallengerParams,
    explain="La ventana reciente mide la plasticidad y la referencia, todo lo "
    "anterior, la retención; con menos de min_samples etiquetas recientes se "
    "rechaza, porque no hay con qué decidir.",
)
class ChampionChallenger:
    def __init__(
        self,
        metric: str = "macro_f1",
        min_delta: float = -0.01,
        max_forgetting: float = 0.03,
        min_samples: int = 500,
        window: float | None = None,
    ) -> None:
        self.metric, self.min_delta = metric, min_delta
        self.max_forgetting, self.min_samples = max_forgetting, min_samples
        self.window = window

    def _gain(self, champion: float, challenger: float) -> float:
        """How much better the challenger is; the loss is better lower."""
        sign = -1.0 if self.metric == "loss" else 1.0
        return sign * (challenger - champion)

    def decide(self, scores: Scores) -> Decision:
        """``scores[model][part]``: the metrics and ``samples`` of each model on
        each part, as the validators reported them together."""
        recent = {m: scores[m]["recent"] for m in MODELS}
        reference = {m: scores[m]["reference"] for m in MODELS}
        n_recent = float(recent["challenger"].get("samples", 0.0))
        n_reference = float(reference["challenger"].get("samples", 0.0))
        measures = {"recent_samples": n_recent, "reference_samples": n_reference}
        if n_recent < max(self.min_samples, 1):
            return Decision(False, "min_samples", measures)
        champion, challenger = (recent[m][self.metric] for m in MODELS)
        delta = self._gain(champion, challenger)
        measures |= {
            "recent_champion": champion,
            "recent_challenger": challenger,
            "delta": delta,
        }
        forgetting = 0.0  # nothing earlier yet: nothing to forget
        if n_reference:
            old, new = (reference[m][self.metric] for m in MODELS)
            forgetting = -self._gain(old, new)
            measures |= {"reference_champion": old, "reference_challenger": new}
        measures["forgetting"] = forgetting
        if delta < self.min_delta - 1e-12:
            return Decision(False, "min_delta", measures)
        if forgetting > self.max_forgetting + 1e-12:
            return Decision(False, "max_forgetting", measures)
        return Decision(True, "promoted", measures)
