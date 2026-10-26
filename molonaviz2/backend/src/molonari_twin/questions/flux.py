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


def model_of(results: TwinResults) -> str:
    """Étiquette publique du modèle : `forward` (twin direct, sans incertitude) ou `inversion` (twin mcmc)."""
    return "inversion" if results.metadata.get("mode") == "mcmc" else "forward"


def posterior_mean_flux(results: TwinResults, start=None, end=None, level: float = 0.9) -> tuple[dict, list[Issue]]:
    """Flux moyen a posteriori et son intervalle à `level`, tirés de q(0, t) de chaque champ conservé (P2c.1b).

    Chaque champ donne sa moyenne de q sur la période (aux dates des champs, bornes incluses) ; la valeur centrale
    est la moyenne de ces moyennes, `interval` leurs quantiles (1 − level)/2 et (1 + level)/2. C'est l'incertitude
    sur le flux moyen, et non la dispersion de q dans le temps. Sans séries stockées (résultat écrit avant P2c.1) ou
    sans date de champ dans la période, un avertissement dit pourquoi et le dictionnaire est vide."""
    stored = results.riverbed_flux
    if stored is None:
        return {}, [Issue(None, "W_FLUX_NO_SAMPLES")]
    try:
        inside = within(as_seconds(stored["time"]), start, end)
    except UserError as error:  # aucune date de champ dans la période
        return {}, [Issue.from_error(None, error)]
    means = np.asarray(stored["values"], float)[:, inside].mean(axis=1)  # une moyenne par échantillon
    low, high = np.quantile(means, [(1.0 - level) / 2, (1.0 + level) / 2])
    return {"mean": float(means.mean()), "interval": (float(low), float(high)), "samples": int(means.size)}, []


def water_flux(results: TwinResults, start=None, end=None, level: float = 0.9) -> Answer:
    """Flux au fond de la rivière sur la période (HN-10) : moyenne, médiane, P5 et P95 de q (m/s) dans le temps,
    avec le nombre de dates et la période réelle.

    En mode mcmc, `mean` est le flux moyen a posteriori et son `interval` (à `level`) celui de cette moyenne, tiré
    des séries q(0, t) des champs conservés ; `mean_best` est la moyenne du meilleur échantillon. En mode direct,
    `mean` est une valeur seule, sans `interval` : `uncertainty` vaut `none`. `median`, `p5` et `p95` sont toujours
    les percentiles de la série q(t) du modèle stocké (meilleur échantillon en mcmc), et non un intervalle a
    posteriori. `level` hors de ]0, 1[ : E_QUESTION_LEVEL."""
    if not 0.0 < level < 1.0:
        raise UserError("E_QUESTION_LEVEL", level=level)
    times, q = riverbed_flux(results, start, end)
    p5, median, p95 = np.percentile(q, [5, 50, 95])
    mean = Quantity(float(q.mean()), "m/s")
    extra: dict[str, Quantity] = {}
    warnings = list(results.warnings)
    if results.metadata.get("mode") == "mcmc":
        posterior, more = posterior_mean_flux(results, start, end, level)
        warnings += more
        if posterior:
            extra["mean_best"] = mean
            mean = Quantity(posterior["mean"], "m/s", posterior["interval"])
            extra["samples"] = Quantity(posterior["samples"], "1")
    data = {
        "mean": mean,
        "median": Quantity(float(median), "m/s"),
        "p5": Quantity(float(p5), "m/s"),
        "p95": Quantity(float(p95), "m/s"),
        "points": Quantity(int(q.size), "1"),
        "start": Quantity(times[0], "datetime64[s]"),
        "end": Quantity(times[-1], "datetime64[s]"),
        "level": Quantity(float(level), "1"),
        "model": Quantity(model_of(results), "1"),
        "uncertainty": Quantity("posterior" if mean.interval is not None else "none", "1"),
        **extra,
    }
    return Answer(data=data, provenance={"question": "water_flux", "parameters": {**period_parameters(start, end), "level": float(level)}},
                  warnings=warnings)


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
