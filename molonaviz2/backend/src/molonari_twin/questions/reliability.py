"""Puis-je faire confiance à ce résultat ? (HN-41) Un verdict vert, orange ou rouge, et ses raisons.

Critères, chacun avec son niveau (seuils : questions/thresholds.py, **provisoires et non validés**, choix C9) :
- convergence (mode mcmc) : R̂ maximal et ESS minimal des paramètres ;
- fit : RMSE/σT maximal sur les capteurs ;
- identifiability (mcmc) : rapport maximal des écarts-types a posteriori et a priori ;
- coverage (mcmc avec champs stockés) : écart entre la couverture prédictive et son niveau ;
- data : part des mesures attendues qui manquent sur la période ;
- provenance : complète (vert), sinon orange.

Le verdict est le pire des niveaux. Chaque raison est un avertissement `W_RELIABILITY_*` avec ses paramètres (traduit
par le client) ; seuls les critères non verts en font un, `reasons` donne le texte anglais de tous.
"""

from __future__ import annotations

import numpy as np

from ..domain.compute import PARAMETER_LABELS
from ..errors import Issue
from ..twin.results import TwinResults
from .answer import Answer, Quantity
from .convergence import convergence_extremes
from .sensors import (
    PREDICTIVE_SEED,
    SENSORS,
    measurement_sigma,
    model_bands,
    sensor_series,
)
from .soil import soil_properties
from .thresholds import (
    LEVELS,
    THRESHOLDS_VALIDATED,
    ReliabilityThresholds,
    grade_above,
    grade_below,
    resolve_thresholds,
)


def _worst(*levels: str) -> str:
    return max(levels, key=LEVELS.index)


def data_gaps(results: TwinResults) -> float:
    """Part des mesures attendues (dates régulières × 3 capteurs) qui manquent : dates absentes et valeurs non finies."""
    times, measured, _ = sensor_series(results)
    seconds = times.astype(np.int64).astype(float)
    step = np.median(np.diff(seconds)) if seconds.size > 1 else 0.0
    expected = round((seconds[-1] - seconds[0]) / step) + 1 if step > 0 else seconds.size
    present = int(np.isfinite(measured).sum())
    return float(max(0.0, 1.0 - present / (max(expected, seconds.size) * SENSORS)))


def _convergence(results: TwinResults, limits: ReliabilityThresholds):
    rhat, ess = convergence_extremes(results)
    level = _worst(grade_below(rhat, limits.rhat_green, limits.rhat_orange), grade_above(ess, limits.ess_green, limits.ess_orange))
    return level, Issue(None, "W_RELIABILITY_CONVERGENCE", rhat=rhat, ess=ess, rhat_green=limits.rhat_green,
                        rhat_orange=limits.rhat_orange, ess_green=limits.ess_green, ess_orange=limits.ess_orange)


def _identifiability(results: TwinResults, limits: ReliabilityThresholds):
    answer = soil_properties(results)
    worst, where = -1.0, ("", "")
    names = [str(n) for n in answer.data["layers"].value]
    for parameter, label in (("neg_log10_k", "k"), ("porosity", "n"), ("lambda_s", "lambda_s"), ("rhos_cs", "rhos_cs")):
        ratios = answer.data[f"{label}_identifiability"].value
        i = int(np.nanargmax(ratios)) if not np.isnan(ratios).all() else 0
        if not np.isnan(ratios[i]) and ratios[i] > worst:
            worst, where = float(ratios[i]), (PARAMETER_LABELS[parameter], names[i])
    ratio = worst if worst >= 0 else float("nan")
    return (grade_below(ratio, limits.identifiability_green, limits.identifiability_orange),
            Issue(None, "W_RELIABILITY_IDENTIFIABILITY", ratio=ratio, parameter=where[0], layer=where[1],
                  green=limits.identifiability_green, orange=limits.identifiability_orange))


def reliability(results: TwinResults, level: float = 0.9, sigma: float | None = None, seed: int = PREDICTIVE_SEED,
                thresholds: dict | None = None, environment: dict | None = None) -> Answer:
    """Verdict de fiabilité d'un résultat, avec le niveau et la raison de chaque critère. `thresholds` surcharge les
    seuils (voir thresholds.resolve_thresholds, qui lève E_THRESHOLDS_INVALID)."""
    limits = resolve_thresholds(thresholds, environment)
    sigma = measurement_sigma(results, sigma)
    mcmc = results.metadata.get("mode") == "mcmc" and results.posterior is not None
    graded: list[tuple[str, str, Issue]] = []
    warnings = list(results.warnings)

    if mcmc:
        graded.append(("convergence", *_convergence(results, limits)))

    times, measured, simulated = sensor_series(results)
    ratio = float(np.nanmax(np.sqrt(np.nanmean((simulated - measured) ** 2, axis=1)) / sigma)) if sigma > 0 else float("inf")
    graded.append(("fit", grade_below(ratio, limits.fit_green, limits.fit_orange),
                   Issue(None, "W_RELIABILITY_FIT", ratio=ratio, green=limits.fit_green, orange=limits.fit_orange)))

    if mcmc:
        graded.append(("identifiability", *_identifiability(results, limits)))
        bands, more = model_bands(results, times, measured, sigma, level, seed)
        warnings += more
        if "coverage" in bands:
            covered = float(bands["coverage"].value)
            gap = abs(covered - level)
            graded.append(("coverage", grade_below(gap, limits.coverage_green, limits.coverage_orange),
                           Issue(None, "W_RELIABILITY_COVERAGE", coverage=covered, level=float(level),
                                 green=limits.coverage_green, orange=limits.coverage_orange)))

    gaps = data_gaps(results)
    graded.append(("data", grade_below(gaps, limits.gaps_green, limits.gaps_orange),
                   Issue(None, "W_RELIABILITY_DATA", gaps=gaps, green=limits.gaps_green, orange=limits.gaps_orange)))
    problems = sum(1 for w in results.warnings if w.code.startswith("W_PROVENANCE_"))
    graded.append(("provenance", "green" if problems == 0 else "orange", Issue(None, "W_RELIABILITY_PROVENANCE", count=problems)))

    levels = [level_ for _, level_, _ in graded]
    names = np.array(list(limits.as_dict()))
    return Answer(
        data={
            "verdict": Quantity(_worst(*levels), "1"),
            "criteria": Quantity(np.array([name for name, _, _ in graded]), "1"),
            "levels": Quantity(np.array(levels), "1"),
            "reasons": Quantity(np.array([str(issue) for _, _, issue in graded]), "1"),
            "threshold_names": Quantity(names, "1"),
            "thresholds": Quantity(np.array(list(limits.as_dict().values())), "1"),
            "thresholds_validated": Quantity(THRESHOLDS_VALIDATED, "1"),
            "level": Quantity(float(level), "1"),
        },
        provenance={"question": "reliability", "parameters": {"level": float(level), "sigma": sigma, "seed": int(seed),
                                                                "thresholds": limits.as_dict()}},
        warnings=warnings + [issue for _, level_, issue in graded if level_ != "green"],
    )
