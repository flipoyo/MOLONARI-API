"""Flux d'eau à travers le lit, lus dans les résultats d'un twin (HN-10, HN-12).

q est le flux au fond de la rivière (z = 0), en m/s, positif vers le bas : infiltration (la rivière alimente
la nappe) ; négatif : exfiltration. Les conversions d'affichage (m/jour, L/m²) sont faites par le client.
"""

from __future__ import annotations

import numpy as np

from ..twin.results import TwinResults
from .answer import Answer, Quantity
from .period import as_seconds, period_parameters, within
from .volumes import exchanged_volume

RIVERBED = "q_riverbed_m_per_s"


def riverbed_flux(results: TwinResults, start=None, end=None) -> tuple[np.ndarray, np.ndarray]:
    """Dates (T,) et q au fond de la rivière (T,), sur la période."""
    times = as_seconds(results.flux["date"].to_numpy())
    q = results.flux[RIVERBED].to_numpy(float)
    mask = within(times, start, end)
    return times[mask], q[mask]


def water_flux(results: TwinResults, start=None, end=None) -> Answer:
    """Moyenne, médiane, P5 et P95 de q (m/s) sur la période, avec le nombre de dates et la période réelle."""
    times, q = riverbed_flux(results, start, end)
    p5, median, p95 = np.percentile(q, [5, 50, 95])
    return Answer(
        data={
            "mean": Quantity(float(q.mean()), "m/s"),
            "median": Quantity(float(median), "m/s"),
            "p5": Quantity(float(p5), "m/s"),
            "p95": Quantity(float(p95), "m/s"),
            "points": Quantity(int(q.size), "1"),
            "start": Quantity(times[0], "datetime64[s]"),
            "end": Quantity(times[-1], "datetime64[s]"),
        },
        provenance={"question": "water_flux", "parameters": period_parameters(start, end)},
        warnings=list(results.warnings),
    )


def twin_exchanged_volume(results: TwinResults, start=None, end=None, area: float | None = None) -> Answer:
    """Volumes infiltré, exfiltré et net à travers le lit sur la période (m³/m², ou m³ avec `area`)."""
    times, q = riverbed_flux(results, start, end)
    answer = exchanged_volume(times, q, area)
    parameters = {**period_parameters(start, end), "area": None if area is None else float(area)}
    return Answer(data=answer.data, provenance={"question": "exchanged_volume", "parameters": parameters},
                  warnings=list(answer.warnings) + list(results.warnings))
