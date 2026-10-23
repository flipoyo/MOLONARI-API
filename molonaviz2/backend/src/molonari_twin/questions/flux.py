"""Flux d'eau à travers le lit, lus dans les résultats d'un twin (HN-10, HN-12).

q est le flux au fond de la rivière (z = 0), en m/s, positif vers le bas : infiltration (la rivière alimente
la nappe) ; négatif : exfiltration (q = 0 compte comme l'infiltration, comme la colonne `direction` de
water_flux.csv). Les conversions d'affichage (m/jour, L/m²) sont faites par le client.
"""

from __future__ import annotations

import numpy as np

from ..errors import Issue, UserError
from ..twin.results import TwinResults
from .answer import Answer, Quantity
from .period import as_seconds, period_parameters, time_weights, within
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


def exchange_direction(results: TwinResults, start=None, end=None) -> Answer:
    """Sens de l'échange (HN-11) : parts du temps en infiltration et en exfiltration (pondérées par les intervalles
    de temps), régime dominant et nombre d'inversions de q au fond de la rivière, sur la période.

    En mode mcmc avec les champs de flux (`store_fields: full`), s'y ajoutent P(q ≥ 0) : la part des échantillons
    a posteriori en infiltration au fond, en série et en moyenne sur le temps."""
    times, q = riverbed_flux(results, start, end)
    infiltrating = q >= 0
    weights = time_weights(times)
    share = float(weights[infiltrating].sum() / weights.sum())
    data = {
        "infiltration_share": Quantity(share, "1"),
        "exfiltration_share": Quantity(1.0 - share, "1"),
        "dominant": Quantity("infiltration" if share >= 0.5 else "exfiltration", "1"),
        "reversals": Quantity(int(np.count_nonzero(np.diff(infiltrating.astype(np.int8)))), "1"),
        "points": Quantity(int(q.size), "1"),
        "start": Quantity(times[0], "datetime64[s]"),
        "end": Quantity(times[-1], "datetime64[s]"),
    }
    warnings = list(results.warnings)
    if results.metadata.get("mode") == "mcmc":
        probability, more = _infiltration_probability(results, start, end)
        data.update(probability)
        warnings += more
    return Answer(data=data, provenance={"question": "exchange_direction", "parameters": period_parameters(start, end)},
                  warnings=warnings)


def _infiltration_probability(results: TwinResults, start, end) -> tuple[dict, list[Issue]]:
    stored = results.fields.get("water_flux")
    if stored is None:
        return {}, [Issue.from_error(None, UserError("E_QUESTION_NO_FIELDS"))]
    times = as_seconds(stored["time"])
    try:
        inside = within(times, start, end)
    except UserError as error:  # aucune date de champ dans la période
        return {}, [Issue.from_error(None, error)]
    series = (stored["values"][:, 0, :][:, inside] >= 0).mean(axis=0)  # au fond de la rivière : première cellule
    weights = time_weights(times[inside])
    return {
        "p_infiltration": Quantity(float((series * weights).sum() / weights.sum()), "1"),
        "p_infiltration_series": Quantity(series, "1"),
        "field_time": Quantity(times[inside], "datetime64[s]"),
        "samples": Quantity(int(stored["values"].shape[0]), "1"),
    }, []
