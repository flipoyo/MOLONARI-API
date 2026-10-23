"""Qualité de l'ajustement aux trois capteurs (HN-40) : le modèle reproduit-il les mesures ?

Pour chaque capteur : RMSE, biais, RMSE/σT (1 si le modèle explique les mesures au bruit près), autocorrélation
d'ordre 1 des résidus (proche de 0 si les résidus sont du bruit) et, en mode mcmc avec champs, la part des
mesures couvertes par la bande prédictive.
"""

from __future__ import annotations

import numpy as np

from ..twin.results import TwinResults
from .answer import Answer, Quantity
from .sensors import PREDICTIVE_SEED, measurement_sigma, model_bands, sensor_series

CONSTANT_SERIES = 1e-18  # °C² : en dessous, la série est constante à l'arrondi près


def lag1_autocorrelation(residual: np.ndarray) -> np.ndarray:
    """Autocorrélation d'ordre 1 de chaque série (K, T) : Σ xᵢxᵢ₊₁ / Σ xᵢ², x centré. nan si la série est constante (à l'arrondi près)."""
    residual = np.asarray(residual, float)
    centred = residual - np.nanmean(residual, axis=1, keepdims=True)
    pairs = ~np.isnan(centred[:, :-1]) & ~np.isnan(centred[:, 1:])
    head, tail = np.where(pairs, centred[:, :-1], 0.0), np.where(pairs, centred[:, 1:], 0.0)
    variance = np.nansum(centred**2, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(variance > CONSTANT_SERIES, (head * tail).sum(axis=1) / variance, np.nan)


def fit_quality(results: TwinResults, level: float = 0.9, sigma: float | None = None, seed: int = PREDICTIVE_SEED) -> Answer:
    """RMSE, biais, RMSE/σT et autocorrélation des résidus par capteur (3 valeurs), et la couverture prédictive par
    capteur en mode mcmc avec champs stockés. σT : le paramètre, ou celui enregistré avec les résultats."""
    sigma = measurement_sigma(results, sigma)
    times, measured, simulated = sensor_series(results)
    residual = simulated - measured
    rmse = np.sqrt(np.nanmean(residual**2, axis=1))
    bands, warnings = model_bands(results, times, measured, sigma, level, seed)
    data = {
        "rmse": Quantity(rmse, "°C"),
        "bias": Quantity(np.nanmean(residual, axis=1), "°C"),
        "rmse_over_sigma": Quantity(rmse / sigma if sigma > 0 else np.full(rmse.shape, np.inf), "1"),
        "lag1_autocorrelation": Quantity(lag1_autocorrelation(residual), "1"),
        "sigma": Quantity(sigma, "°C"),
        "level": Quantity(float(level), "1"),
    }
    if "coverage_per_sensor" in bands:
        data["coverage_per_sensor"] = bands["coverage_per_sensor"]
    return Answer(data=data, provenance={"question": "fit_quality", "parameters": {"level": float(level), "sigma": sigma, "seed": int(seed)}},
                  warnings=list(results.warnings) + warnings)
