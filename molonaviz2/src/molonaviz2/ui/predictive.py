"""Bande prédictive aux capteurs et sa couverture, pour l'onglet Résultats et la fenêtre d'un twin.

Tout le calcul est fait par `questions` ; l'interface ne fait que convertir l'unité et dessiner.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..errors import UserError
from ..i18n import t
from ..questions import coverage, observed_at, predictive_band, sensor_samples
from .units import level_label

PREDICTIVE_SEED = 0  # graine fixe : la bande ne change pas d'un affichage à l'autre


@dataclass(frozen=True)
class PredictiveBand:
    times: np.ndarray  # dates des champs (T′,)
    low: np.ndarray  # °C, (T′, 3)
    high: np.ndarray  # °C, (T′, 3)
    level: float
    coverage: float  # part des mesures dans la bande (NaN sans point)
    points: int  # mesures comparées

    def label(self) -> str:
        return t("Predictive {level} — {coverage} of measurements (n = {n})").format(
            level=level_label(self.level), coverage=f"{self.coverage * 100:.0f} %" if self.points else "–", n=self.points)


def predictive_at_sensors(result, times, observed, sigma: float, level: float) -> PredictiveBand:
    """Bande prédictive d'un résultat d'inversion ou des résultats d'un twin, et couverture des mesures
    `observed` (T, 3) aux dates `times`. Lève UserError (E_QUESTION_NO_FIELDS) sans champs stockés."""
    samples = sensor_samples(result).data
    band = predictive_band(samples["temperature"].value, sigma, level, seed=PREDICTIVE_SEED).data["temperature"]
    measured = observed_at(times, observed, samples["time"].value).data["temperature"].value
    answer = coverage(measured, band).data
    low, high = band.interval
    return PredictiveBand(samples["time"].value, low.T, high.T, level, answer["coverage"].value, answer["points"].value)


def try_predictive_at_sensors(result, times, observed, sigma: float, level: float) -> tuple[PredictiveBand | None, UserError | None]:
    try:
        return predictive_at_sensors(result, times, observed, sigma, level), None
    except UserError as error:
        return None, error
