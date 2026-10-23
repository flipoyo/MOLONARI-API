"""Température simulée à une profondeur et/ou à une date (HN-30)."""

from __future__ import annotations

import numpy as np

from ..errors import UserError
from ..twin.results import TwinResults
from .answer import Answer, Quantity
from .period import as_seconds


def _interpolate(axis: np.ndarray, values: np.ndarray, at: float, name: str, shown: str, low: str, high: str) -> np.ndarray:
    """Interpolation linéaire de `values` le long de son premier axe, au point `at` de `axis` (croissant)."""
    if not axis[0] <= at <= axis[-1]:
        raise UserError("E_QUESTION_OUT_OF_RANGE", name=name, value=shown, low=low, high=high)
    if len(axis) == 1:
        return values[0]
    i = int(np.clip(np.searchsorted(axis, at, side="right") - 1, 0, len(axis) - 2))
    weight = (at - axis[i]) / (axis[i + 1] - axis[i])
    return (1 - weight) * values[i] + weight * values[i + 1]


def temperature_at(results: TwinResults, z: float | None = None, t=None) -> Answer:
    """θ simulée en °C, interpolée linéairement : une valeur (z et t), un profil T(z) (t seul) ou une série T(t)
    (z seul). z en m, positif vers le bas, dans la colonne ; t, une date de la période calculée."""
    if z is None and t is None:
        raise UserError("E_QUESTION_Z_OR_T")
    depths = np.asarray(results.depths, dtype=float)
    times = as_seconds(results.times)
    values = np.asarray(results.temperature, dtype=float)  # (Z, T)
    if t is not None:
        at = np.datetime64(t, "s")
        seconds = (times - times[0]) / np.timedelta64(1, "s")
        values = _interpolate(seconds, values.T, float((at - times[0]) / np.timedelta64(1, "s")), "t", str(at),
                              str(times[0]), str(times[-1]))  # (Z,)
    if z is not None:
        values = _interpolate(depths, values, float(z), "z", f"{z:g} m", f"{depths[0]:g} m", f"{depths[-1]:g} m")
    return Answer(
        data={
            "temperature": Quantity(float(values) if np.ndim(values) == 0 else values, "°C"),
            "depth": Quantity(depths if z is None else float(z), "m"),
            "time": Quantity(times if t is None else np.datetime64(t, "s"), "datetime64[s]"),
        },
        provenance={"question": "temperature",
                    "parameters": {"z": None if z is None else float(z), "t": None if t is None else str(np.datetime64(t, "s"))}},
        warnings=list(results.warnings),
    )
