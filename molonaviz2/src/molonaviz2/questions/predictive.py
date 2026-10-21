"""Bande prédictive aux capteurs et couverture des mesures.

La bande crédible (quantiles des champs simulés) ne contient que l'incertitude des paramètres ; la
bande prédictive y ajoute le bruit de mesure N(0, σ²), et c'est elle que les mesures doivent couvrir.
"""

from __future__ import annotations

import numpy as np

from ..engine.results import InversionResult
from ..errors import Issue, UserError
from ..twin.results import TwinResults
from .answer import Answer, Quantity


def _sensor_rows(depths, sensor_depths) -> list[int]:
    depths = np.asarray(depths, dtype=float)
    return [int(np.argmin(np.abs(depths - d))) for d in sensor_depths]


def sensor_samples(result: InversionResult | TwinResults) -> Answer:
    """Échantillons de température aux 3 capteurs (S, 3, T′), en °C, et leurs dates (T′).

    Champs stockés aux capteurs (`store_fields: sensors`) ou en entier (`full`, on prend alors les
    lignes les plus proches des capteurs), d'un résultat d'inversion ou des résultats relus d'un twin."""
    if isinstance(result, TwinResults):
        stored = result.fields.get("temperature")
        if stored is None or not result.sensor_depths:
            raise UserError("E_QUESTION_NO_FIELDS")
        values = stored["values"][:, _sensor_rows(stored["depth_m"], result.sensor_depths), :]
        times, depths = stored["time"], result.sensor_depths
    else:
        depths = result.best.sensor_depths
        if result.sensor_temperature is not None:
            values = result.sensor_temperature
        elif result.field_temperature is not None:
            values = result.field_temperature[:, _sensor_rows(result.field_depths, depths), :]
        else:
            raise UserError("E_QUESTION_NO_FIELDS")
        times = result.field_times
    return Answer(
        data={
            "temperature": Quantity(np.asarray(values, dtype=float), "°C"),
            "time": Quantity(np.asarray(times).astype("datetime64[s]"), "datetime64[s]"),
            "depth": Quantity(np.asarray(depths, dtype=float), "m"),
        },
        provenance={"question": "sensor_samples", "parameters": {}},
    )


def predictive_band(sensor_samples, sigma: float, level: float, seed: int) -> Answer:
    """Médiane et bande prédictive au niveau `level` : quantiles des échantillons (S, K, T′) bruités par N(0, σ²).

    Le bruit est tiré par numpy.random.default_rng(seed) : même graine, même bande."""
    if not 0.0 < level < 1.0:
        raise UserError("E_QUESTION_LEVEL", level=level)
    if not sigma >= 0.0:
        raise UserError("E_QUESTION_SIGMA", sigma=sigma)
    samples = np.asarray(sensor_samples, dtype=float)
    noisy = samples + np.random.default_rng(seed).normal(0.0, sigma, samples.shape)
    low, median, high = np.quantile(noisy, [(1.0 - level) / 2, 0.5, (1.0 + level) / 2], axis=0)
    return Answer(
        data={"temperature": Quantity(median, "°C", (low, high)), "level": Quantity(float(level), "1")},
        provenance={"question": "predictive_band",
                    "parameters": {"sigma": float(sigma), "level": float(level), "seed": int(seed), "samples": samples.shape[0]}},
    )


def coverage(observed, band: Quantity) -> Answer:
    """Part des mesures comprises dans l'intervalle de `band`, bornes incluses, au total et par ligne (capteur).

    Les points où la mesure ou une borne n'est pas finie (NaN) sont ignorés."""
    observed = np.asarray(observed, dtype=float)
    low, high = (np.asarray(bound, dtype=float) for bound in band.interval)
    if observed.shape != low.shape or observed.shape != high.shape:
        raise UserError("E_QUESTION_SHAPES", expected=str(low.shape), found=str(observed.shape))
    valid = np.isfinite(observed) & np.isfinite(low) & np.isfinite(high)
    inside = valid & (observed >= low) & (observed <= high)
    points = int(valid.sum())
    data = {
        "coverage": Quantity(float(inside.sum() / points) if points else float("nan"), "1"),
        "points": Quantity(points, "1"),
    }
    if observed.ndim > 1:  # une ligne par capteur, les dates sur le dernier axe
        per_sensor = valid.sum(axis=-1)
        with np.errstate(divide="ignore", invalid="ignore"):
            data["per_sensor"] = Quantity(np.where(per_sensor > 0, inside.sum(axis=-1) / per_sensor, np.nan), "1")
        data["points_per_sensor"] = Quantity(per_sensor, "1")
    warnings = [] if points else [Issue(None, "W_COVERAGE_NO_POINTS")]
    return Answer(data=data, provenance={"question": "coverage", "parameters": {}}, warnings=warnings)


def observed_at(times, observed, at_times) -> Answer:
    """Mesures (T, K) aux dates `times`, prises aux dates `at_times` (T′) et rangées en (K, T′).

    Une date de `at_times` absente de `times` donne NaN, que la couverture ignore."""
    times = np.asarray(times).astype("datetime64[s]")
    at_times = np.asarray(at_times).astype("datetime64[s]")
    observed = np.asarray(observed, dtype=float)
    if observed.ndim != 2 or observed.shape[0] != len(times):
        raise UserError("E_QUESTION_SHAPES", expected=f"({len(times)}, K)", found=str(observed.shape))
    index = np.clip(np.searchsorted(times, at_times), 0, len(times) - 1)
    found = times[index] == at_times
    values = np.where(found[:, None], observed[index], np.nan).T
    return Answer(
        data={"temperature": Quantity(values, "°C"), "time": Quantity(at_times, "datetime64[s]")},
        provenance={"question": "observed_at", "parameters": {}},
    )
