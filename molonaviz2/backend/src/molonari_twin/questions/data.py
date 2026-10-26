"""Mesures brutes d'un point (BN-03) et effet d'un nettoyage (BN-04), à partir des fichiers du logger.

Ces questions ne portent sur aucun twin : elles lisent le dossier d'un point (et, pour le nettoyage, celui du
laboratoire) avec le pipeline du brut (`twin.pipeline`). Rien n'est enregistré, rien n'est calculé par le modèle :
elles ne chargent pas JAX.

Tableaux rangés comme ceux des capteurs : thermomètres d'abord, `(4, T)`, du moins profond au plus profond.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..errors import UserError
from ..processing.cleanup import VARIABLES, CleanupSettings
from ..twin.pipeline import clean_point, load_point, read_raw_point
from .answer import Answer, Quantity
from .period import as_seconds, period_parameters

TEMPERATURES = ["t1", "t2", "t3", "t4"]


def _mask(times: np.ndarray, start, end) -> np.ndarray:
    mask = np.ones(times.shape, dtype=bool)
    if start is not None:
        mask &= times >= np.datetime64(start, "s")
    if end is not None:
        mask &= times <= np.datetime64(end, "s")
    return mask


def raw_series(point_folder: str | Path, start=None, end=None) -> Answer:
    """Mesures brutes d'un point entre `start` et `end` (bornes incluses) : températures de la tige (°C), tension
    (V) et température du capteur de pression (°C), avec le nombre de lignes écartées du fichier (événements du
    logger, valeurs vides ou illisibles, dates en double). Un fichier de mesures absent est un avertissement."""
    raw = read_raw_point(Path(point_folder), require_measures=False)
    data: dict[str, Quantity] = {}
    kept = 0
    if raw.temperature is not None:
        frame = raw.temperature.frame
        times = as_seconds(frame["date"].to_numpy())
        mask = _mask(times, start, end)
        kept += int(mask.sum())
        data["temperature_time"] = Quantity(times[mask], "datetime64[s]")
        data["temperature"] = Quantity(frame[TEMPERATURES].to_numpy(np.float64)[mask].T, "°C")
        data["temperature_dropped"] = Quantity(len(raw.temperature.dropped), "1")
    if raw.pressure is not None:
        frame = raw.pressure.frame
        times = as_seconds(frame["date"].to_numpy())
        mask = _mask(times, start, end)
        kept += int(mask.sum())
        data["pressure_time"] = Quantity(times[mask], "datetime64[s]")
        data["voltage"] = Quantity(frame["voltage"].to_numpy(np.float64)[mask], "V")
        data["pressure_temperature"] = Quantity(frame["temp_bed"].to_numpy(np.float64)[mask], "°C")
        data["pressure_dropped"] = Quantity(len(raw.pressure.dropped), "1")
    period = period_parameters(start, end)
    if not kept:
        raise UserError("E_QUESTION_PERIOD_EMPTY", start=period["start"] or "…", end=period["end"] or "…")
    parameters = {"point_folder": str(point_folder), **period}
    return Answer(data=data, provenance={"question": "raw_series", "parameters": parameters}, warnings=list(raw.warnings))


def processing_preview(point_folder: str | Path, lab_folder: str | Path, start=None, end=None, methods=None,
                       excluded=()) -> Answer:
    """Effet d'un nettoyage sur toutes les dates communes aux deux fichiers : mesures associées (températures,
    charge différentielle en m), raison d'exclusion de chaque date (`outside_window`, `manual`, `missing`,
    `iqr:<variable>`, `zscore:<variable>` ; vide si la date est gardée), décompte par raison et dates sans
    équivalent dans l'autre fichier. Ce sont les entrées du modèle, avec les σ des instruments."""
    point = load_point(Path(point_folder), Path(lab_folder))
    settings = CleanupSettings(
        start=start, end=end, methods={**CleanupSettings().methods, **(methods or {})}, excluded=list(excluded)
    )
    result = clean_point(point, settings)
    measures, reason = result.measures, result.reason.to_numpy(dtype=str)
    counts = result.counts()
    instruments = point.instruments
    data = {
        "time": Quantity(as_seconds(measures["date"].to_numpy()), "datetime64[s]"),
        "temperature": Quantity(measures[TEMPERATURES].to_numpy(np.float64).T, "°C"),
        "river_temperature": Quantity(measures["temp_bed"].to_numpy(np.float64), "°C"),
        "pressure": Quantity(measures["pressure"].to_numpy(np.float64), "m"),
        "kept": Quantity(result.kept.to_numpy(dtype=bool), "1"),
        "reason": Quantity(reason, "1"),
        "reasons": Quantity(np.array(list(counts), dtype=str), "1"),
        "reason_counts": Quantity(np.array(list(counts.values()), dtype=np.int64), "1"),
        "kept_count": Quantity(int(result.kept.sum()), "1"),
        "total_count": Quantity(len(measures), "1"),
        "unmatched_temperature": Quantity(result.unmatched_temperature, "1"),
        "unmatched_pressure": Quantity(result.unmatched_pressure, "1"),
        "sigma_pressure": Quantity(instruments.sigma_pressure, "m"),
        "sigma_temperature": Quantity(instruments.sigma_temperature, "°C"),
    }
    parameters = {
        "point_folder": str(point_folder), "lab_folder": str(lab_folder), **period_parameters(start, end),
        "methods": {v: settings.methods[v] for v in VARIABLES},
        "excluded": [str(np.datetime64(d, "s")) for d in settings.excluded],
    }
    return Answer(data=data, provenance={"question": "processing_preview", "parameters": parameters},
                  warnings=list(point.raw.warnings))
