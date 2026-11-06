"""Mesures et simulation aux trois capteurs d'un twin : intervalles (HN-33) et résidus (BN-31).

Tableaux rangés capteur d'abord : (3, T) aux dates des mesures, (3, T′) aux dates des champs échantillonnés.
- `measurement` : la mesure ± z·σ de l'instrument (modes D et M) ;
- `credible` : quantiles des températures simulées par les échantillons a posteriori (incertitude du modèle,
  mode M, champs stockés) ;
- `predictive` : les mêmes, bruités par N(0, σ²) (une nouvelle mesure), et la part des mesures qu'elle couvre.
La bande prédictive est tirée avec une graine fixe : c'est celle que dessine l'application (PREDICTIVE_SEED).
"""

from __future__ import annotations

import numpy as np

from ..errors import Issue, UserError
from ..twin.results import TwinResults
from .answer import Answer, Quantity
from .intervals import sensor_intervals
from .period import as_seconds
from .predictive import coverage, observed_at, predictive_band, sensor_samples

PREDICTIVE_SEED = 0
SENSORS = 3


def sensor_series(results: TwinResults) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Dates (T,), mesures (3, T) et simulation (3, T) aux capteurs, en °C."""
    frame = results.residuals
    measured = np.stack([frame[f"t{i + 1}_measured"].to_numpy(float) for i in range(SENSORS)])
    simulated = np.stack([frame[f"t{i + 1}_simulated"].to_numpy(float) for i in range(SENSORS)])
    return as_seconds(frame["date"].to_numpy()), measured, simulated


def measurement_sigma(results: TwinResults, sigma: float | None) -> float:
    """σ de l'instrument (°C) : le paramètre s'il est donné, sinon celui enregistré avec les résultats."""
    if sigma is not None:
        return float(sigma)
    recorded = results.metadata.get("instruments", {}).get("sigma_temperature_C")
    if recorded is None:
        raise UserError("E_QUESTION_SIGMA_REQUIRED")
    return float(recorded)


def model_bands(results: TwinResults, times, measured, sigma: float, level: float, seed: int) -> tuple[dict, list[Issue]]:
    """Bandes `credible` et `predictive` et couverture, pour un twin mcmc aux champs stockés. Sans champs, ou en
    mode direct, aucune bande : un avertissement dit pourquoi (mode mcmc sans champs seulement)."""
    if results.metadata.get("mode") != "mcmc" or not results.results_format.startswith("2."):
        return {}, []
    try:
        samples = sensor_samples(results).data
    except UserError as error:
        return {}, [Issue.from_error(None, error)]
    temperature = samples["temperature"].value
    credible = predictive_band(temperature, 0.0, level, seed).data["temperature"]
    predictive = predictive_band(temperature, sigma, level, seed).data["temperature"]
    at_fields = observed_at(times, measured.T, samples["time"].value).data["temperature"].value
    covered = coverage(at_fields, predictive)
    return {
        "field_time": samples["time"],
        "credible": credible,
        "predictive": predictive,
        "coverage": covered.data["coverage"],
        "coverage_per_sensor": covered.data["per_sensor"],
        "points": covered.data["points"],
    }, list(covered.warnings)


def sensor_report(results: TwinResults, level: float = 0.9, sigma: float | None = None, seed: int = PREDICTIVE_SEED) -> Answer:
    """Par capteur : mesure avec son intervalle d'instrument, médiane simulée et, en mode M, bandes crédible et
    prédictive et couverture des mesures par la bande prédictive."""
    sigma = measurement_sigma(results, sigma)
    times, measured, simulated = sensor_series(results)
    interval = sensor_intervals(measured, sigma, level).data
    bands, warnings = model_bands(results, times, measured, sigma, level, seed)
    return Answer(
        data={
            "time": Quantity(times, "datetime64[s]"),
            "measured": interval["temperature"],
            "simulated": Quantity(simulated, "°C"),
            "level": interval["level"],
            "z": interval["z"],
            "sigma": Quantity(sigma, "°C"),
            **bands,
        },
        provenance={"question": "sensor_intervals", "parameters": {"level": float(level), "sigma": sigma, "seed": int(seed)}},
        warnings=list(results.warnings) + warnings,
    )


def residuals(results: TwinResults, intervals: bool = False, level: float = 0.9, sigma: float | None = None,
              seed: int = PREDICTIVE_SEED) -> Answer:
    """Mesuré, simulé et résidu (simulé − mesuré) par capteur, avec le RMSE ; `intervals` ajoute les bandes
    crédible et prédictive (mode M)."""
    times, measured, simulated = sensor_series(results)
    residual = simulated - measured
    data = {
        "time": Quantity(times, "datetime64[s]"),
        "measured": Quantity(measured, "°C"),
        "simulated": Quantity(simulated, "°C"),
        "residual": Quantity(residual, "°C"),
        "rmse": Quantity(np.sqrt(np.nanmean(residual**2, axis=1)), "°C"),
    }
    warnings = list(results.warnings)
    parameters = {"intervals": bool(intervals)}
    if intervals:
        sigma = measurement_sigma(results, sigma)
        bands, more = model_bands(results, times, measured, sigma, level, seed)
        data.update(bands)
        warnings += more
        parameters.update(level=float(level), sigma=sigma, seed=int(seed))
    return Answer(data=data, provenance={"question": "residuals", "parameters": parameters}, warnings=warnings)
