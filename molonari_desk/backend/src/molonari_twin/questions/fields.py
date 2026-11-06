"""Champs complets d'un twin (BN-30) : température, flux d'eau et flux de chaleur en fonction de la profondeur et du temps.

Ce sont les champs du meilleur ajustement (modes direct et mcmc). Chaque variable est un tableau (profondeurs, dates), avec
ses axes `depth` et `time`. Ils sont volumineux : demander le format binaire (Accept: application/x-npz).
"""

from __future__ import annotations

import numpy as np

from ..errors import UserError
from ..twin.results import TwinResults
from .answer import Answer, Quantity
from .period import as_seconds, period_parameters, within

# nom → (attribut des résultats, unité)
VARIABLES = {
    "temperature": ("temperature", "°C"),
    "water_flux": ("water_flux_field", "m/s"),
    "heat_flux_advective": ("heat_flux_advective", "W/m²"),
    "heat_flux_conductive": ("heat_flux_conductive", "W/m²"),
}


def result_fields(results: TwinResults, variables=None, start=None, end=None, depth_range=None, time_step: int = 1) -> Answer:
    """Les `variables` (toutes par défaut) sur la période, la plage de profondeurs `depth_range` = (bas, haut) en m
    (bornes incluses, None : sans borne) et une date sur `time_step`. Positif vers le bas pour les flux."""
    names = list(VARIABLES) if variables is None else list(dict.fromkeys(variables))
    for name in names:
        if name not in VARIABLES:
            raise UserError("E_QUESTION_VARIABLE", variable=name, known=", ".join(VARIABLES))
    depths = np.asarray(results.depths, float)
    times = as_seconds(results.times)
    columns = np.flatnonzero(within(times, start, end))[:: max(int(time_step), 1)]
    rows = np.ones(depths.shape, bool)
    low, high = (None, None) if depth_range is None else depth_range
    if low is not None:
        rows &= depths >= low
    if high is not None:
        rows &= depths <= high
    if not rows.any():
        raise UserError("E_QUESTION_OUT_OF_RANGE", name="depth_range", value=f"{low} – {high} m",
                        low=f"{depths[0]:g} m", high=f"{depths[-1]:g} m")
    data = {"depth": Quantity(depths[rows], "m"), "time": Quantity(times[columns], "datetime64[s]")}
    for name in names:
        attribute, unit = VARIABLES[name]
        values = getattr(results, attribute)
        if values is None:
            raise UserError("E_QUESTION_NO_FIELDS")
        data[name] = Quantity(np.ascontiguousarray(np.asarray(values, float)[np.ix_(np.flatnonzero(rows), columns)]), unit)
    parameters = {"variables": names, **period_parameters(start, end), "depth_range": None if depth_range is None else list(depth_range),
                  "time_step": int(time_step)}
    return Answer(data=data, provenance={"question": "fields", "parameters": parameters}, warnings=list(results.warnings))
