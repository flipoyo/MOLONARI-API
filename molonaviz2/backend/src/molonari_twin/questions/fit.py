"""Qualité de l'ajustement aux trois capteurs (HN-40) : le modèle reproduit-il les mesures ?

Pour chaque capteur : RMSE, biais, RMSE/σT (1 si le modèle explique les mesures au bruit près), autocorrélation
d'ordre 1 des résidus (proche de 0 si les résidus sont du bruit) et, en mode mcmc avec champs, la part des
mesures couvertes par la bande prédictive. La RMSE totale est la moyenne quadratique des trois RMSE (comme celle
de metadata.json).

Rapport public (P2c.1) : un twin porte un seul modèle. Un twin `direct` est le modèle forward, sans incertitude ; un
twin `mcmc` est le modèle d'inversion, représenté par son meilleur échantillon (celui des résidus enregistrés), et la
RMSE de la médiane a posteriori vient en complément. `model_difference` compare directement les deux modèles, forward
moins inversion, sans passer par les mesures.
"""

from __future__ import annotations

import numpy as np

from ..errors import Issue, UserError
from ..twin.results import TwinResults
from .answer import Answer, Quantity
from .flux import model_of
from .predictive import observed_at
from .sensors import (
    PREDICTIVE_SEED,
    SENSORS,
    measurement_sigma,
    model_bands,
    sensor_series,
)

CONSTANT_SERIES = 1e-18  # °C² : en dessous, la série est constante à l'arrondi près
MEDIAN_QUANTILE = 0.5
SENSOR_NAMES = tuple(f"t{i + 1}" for i in range(SENSORS))


def lag1_autocorrelation(residual: np.ndarray) -> np.ndarray:
    """Autocorrélation d'ordre 1 de chaque série (K, T) : Σ xᵢxᵢ₊₁ / Σ xᵢ², x centré. nan si la série est constante (à l'arrondi près)."""
    residual = np.asarray(residual, float)
    centred = residual - np.nanmean(residual, axis=1, keepdims=True)
    pairs = ~np.isnan(centred[:, :-1]) & ~np.isnan(centred[:, 1:])
    head, tail = np.where(pairs, centred[:, :-1], 0.0), np.where(pairs, centred[:, 1:], 0.0)
    variance = np.nansum(centred**2, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(variance > CONSTANT_SERIES, (head * tail).sum(axis=1) / variance, np.nan)


def total_rmse(rmse) -> float:
    """RMSE totale : moyenne quadratique des RMSE par capteur (celle de ForwardResult.rmse et de metadata.json)."""
    return float(np.sqrt(np.nanmean(np.asarray(rmse, float) ** 2)))


def posterior_median_rmse(results: TwinResults, times, measured) -> tuple[np.ndarray, float] | None:
    """RMSE par capteur et totale de la médiane a posteriori des températures simulées (quantiles.npz), contre les
    mesures aux dates des champs. None sans quantiles stockés ou sans le quantile 0,5."""
    stored = results.quantiles
    if stored is None or not results.sensor_depths:
        return None
    index = np.flatnonzero(np.isclose(stored["quantile"], MEDIAN_QUANTILE))
    if index.size == 0:
        return None
    depths = np.asarray(stored["depth_m"], float)
    rows = [int(np.argmin(np.abs(depths - d))) for d in results.sensor_depths]  # lignes les plus proches des capteurs
    median = np.asarray(stored["temperature_C"][index[0]], float)[rows]  # (3, T′)
    observed = observed_at(times, measured.T, stored["time"]).data["temperature"].value  # (3, T′)
    rmse = np.sqrt(np.nanmean((median - observed) ** 2, axis=1))
    return rmse, total_rmse(rmse)


def fit_quality(results: TwinResults, level: float = 0.9, sigma: float | None = None, seed: int = PREDICTIVE_SEED) -> Answer:
    """RMSE, biais, RMSE/σT et autocorrélation des résidus par capteur (3 valeurs), et la couverture prédictive par
    capteur en mode mcmc avec champs stockés. σT : le paramètre, ou celui enregistré avec les résultats.

    P2c.1 : `rmse_total`, les noms des capteurs, l'étiquette du modèle (`forward` ou `inversion`) et de son
    incertitude (`none` ou `posterior`) ; en mode mcmc, `median_rmse` et `median_rmse_total` de la médiane a
    posteriori (les autres valeurs sont celles du meilleur échantillon)."""
    sigma = measurement_sigma(results, sigma)
    times, measured, simulated = sensor_series(results)
    residual = simulated - measured
    rmse = np.sqrt(np.nanmean(residual**2, axis=1))
    bands, warnings = model_bands(results, times, measured, sigma, level, seed)
    mcmc = results.metadata.get("mode") == "mcmc"
    data = {
        "rmse": Quantity(rmse, "°C"),
        "bias": Quantity(np.nanmean(residual, axis=1), "°C"),
        "rmse_over_sigma": Quantity(rmse / sigma if sigma > 0 else np.full(rmse.shape, np.inf), "1"),
        "lag1_autocorrelation": Quantity(lag1_autocorrelation(residual), "1"),
        "sigma": Quantity(sigma, "°C"),
        "level": Quantity(float(level), "1"),
        "rmse_total": Quantity(total_rmse(rmse), "°C"),
        "sensor": Quantity(np.array(SENSOR_NAMES), "1"),
        "model": Quantity(model_of(results), "1"),
        "uncertainty": Quantity("posterior" if mcmc else "none", "1"),
    }
    if "coverage_per_sensor" in bands:
        data["coverage_per_sensor"] = bands["coverage_per_sensor"]
    if mcmc:
        median = posterior_median_rmse(results, times, measured)
        if median is None:
            warnings = warnings + [Issue(None, "W_FIT_NO_MEDIAN")]
        else:
            data["median_rmse"] = Quantity(median[0], "°C")
            data["median_rmse_total"] = Quantity(median[1], "°C")
    return Answer(data=data, provenance={"question": "fit_quality", "parameters": {"level": float(level), "sigma": sigma, "seed": int(seed)}},
                  warnings=list(results.warnings) + warnings)


def model_difference(forward: TwinResults, inversion: TwinResults) -> Answer:
    """Différence directe entre le modèle forward et le modèle d'inversion, forward − inversion, aux trois capteurs.

    Elle ne passe pas par les mesures : écart moyen (`mean_difference`) et RMSE de la différence (`rmse`) par capteur,
    puis la RMSE totale (`rmse_total`). Le modèle d'inversion est son meilleur échantillon. Les deux résultats
    doivent avoir les mêmes dates de simulation (E_QUESTION_MODELS_INCOMPARABLE) : le contrôle complet de
    comparabilité des deux instances (données, nettoyage, couches) est celui du générateur de rapport."""
    times, _, simulated_forward = sensor_series(forward)
    times_inversion, _, simulated_inversion = sensor_series(inversion)
    if times.shape != times_inversion.shape or not np.array_equal(times, times_inversion):
        raise UserError("E_QUESTION_MODELS_INCOMPARABLE", forward=len(times), inversion=len(times_inversion))
    difference = simulated_forward - simulated_inversion
    rmse = np.sqrt(np.nanmean(difference**2, axis=1))
    return Answer(
        data={
            "mean_difference": Quantity(np.nanmean(difference, axis=1), "°C"),
            "rmse": Quantity(rmse, "°C"),
            "rmse_total": Quantity(total_rmse(rmse), "°C"),
            "sensor": Quantity(np.array(SENSOR_NAMES), "1"),
            "points": Quantity(len(times), "1"),
        },
        provenance={"question": "model_difference", "parameters": {}},
        warnings=list(forward.warnings) + list(inversion.warnings),
    )
