"""Diagnostics de convergence des chaînes (BN-40), en numpy seul (engine.stats) : ni JAX ni ArviZ.

Par paramètre et par couche : R̂ (rank-normalized), ESS bulk et tail, MCSE de la moyenne. Par chaîne : acceptation,
énergie moyenne, ESS bulk de chaque paramètre et R̂ maximal recalculé sans cette chaîne (une chaîne qui le fait
beaucoup baisser n'est pas d'accord avec les autres). Les statistiques portent sur la sélection (chaînes, itérations
sautées, thin) ; le R̂ maximal de toutes les chaînes et itérations est donné à côté, pour voir l'effet de la sélection.
"""

from __future__ import annotations

import numpy as np

from ..domain.compute import INVERTED_PARAMETERS
from ..engine import stats
from ..twin.results import TwinResults
from .answer import Answer, Quantity
from .chains import _posterior, select_samples
from .thresholds import THRESHOLDS_VALIDATED, ReliabilityThresholds

MIN_KEPT_FOR_STATISTICS = 4  # itérations gardées par chaîne en dessous desquelles R̂ et ESS ne sont pas définis


def _max(values: np.ndarray) -> float:
    return float(np.nan) if np.isnan(values).all() else float(np.nanmax(values))


def convergence_extremes(results: TwinResults) -> tuple[float, float]:
    """R̂ maximal (rank-normalized) et ESS minimal (bulk et tail) de tous les paramètres, toutes chaînes et itérations
    gardées. nan s'il y a moins de 4 itérations gardées par chaîne. C'est le calcul de HN-41 et de HN-45."""
    samples = np.moveaxis(np.asarray(results.posterior["samples"], float), 1, 0)  # (N, I, L, 4)
    if samples.shape[1] < MIN_KEPT_FOR_STATISTICS:
        return float("nan"), float("nan")
    rhat = float(np.nanmax(stats.rank_normalized_rhat(samples)))
    ess = float(min(np.nanmin(stats.ess_bulk(samples)), np.nanmin(stats.ess_tail(samples))))
    return rhat, ess


def public_convergence(results: TwinResults) -> Answer:
    """Convergence de l'algorithme pour le rapport public (HN-45), sans verdict : R̂ maximal, ESS minimal, taux
    d'acceptation moyen, le burn-in a-t-il convergé, avec les seuils de référence.

    Les seuils sont ceux par défaut de thresholds.py (ni la variable MOLONARI_THRESHOLDS ni un paramètre : le rapport
    doit se reproduire), et `thresholds_validated` dit s'ils ont été validés par un hydrogéologue (choix C9). Ni
    verdict, ni couleur : ils mêlent des critères que cette réponse ne porte pas. Le seuil de R̂ du développeur
    (`gr_threshold`) est privé et n'y figure pas. Mode mcmc seulement (E_QUESTION_NO_POSTERIOR sans échantillons)."""
    posterior = _posterior(results)
    rhat, ess = convergence_extremes(results)
    mcmc = results.metadata.get("mcmc") or {}
    limits = ReliabilityThresholds()
    samples = np.asarray(posterior["samples"])
    data = {
        "rhat_max": Quantity(rhat, "1"),
        "ess_min": Quantity(ess, "1"),
        "acceptance_mean": Quantity(float(np.mean(posterior["acceptance"])), "1"),
        "burn_in_converged": Quantity(bool(mcmc.get("burn_in_converged", False)), "1"),
        "burn_in_iterations": Quantity(int(mcmc.get("burn_in_iterations", 0)), "1"),
        "chains": Quantity(int(samples.shape[1]), "1"),
        "samples_per_chain": Quantity(int(samples.shape[0]), "1"),
        "rhat_green": Quantity(limits.rhat_green, "1"),
        "rhat_orange": Quantity(limits.rhat_orange, "1"),
        "ess_green": Quantity(limits.ess_green, "1"),
        "ess_orange": Quantity(limits.ess_orange, "1"),
        "thresholds_validated": Quantity(THRESHOLDS_VALIDATED, "1"),
    }
    return Answer(data=data, provenance={"question": "public_convergence", "parameters": {}}, warnings=list(results.warnings))


def convergence(results: TwinResults, chains=None, burn: int = 0, thin: int = 1) -> Answer:
    """Statistiques de convergence des chaînes choisies. Arrays : (L, 4) par couche et paramètre, (N′,) par chaîne,
    (N′, L, 4) pour l'ESS de chaque chaîne."""
    posterior = _posterior(results)
    rows, _, selected = select_samples(posterior, chains, thin, None, burn)
    samples = np.asarray(posterior["samples"], float)  # (I, N, L, 4)
    kept = np.moveaxis(samples[np.ix_(rows, selected)], 1, 0)  # (N′, I′, L, 4)
    everything = np.moveaxis(samples, 1, 0)

    energy = np.asarray(posterior["energy"], float)[np.ix_(rows, selected)]
    without = np.full(len(selected), np.nan)
    if len(selected) > 2:  # R̂ demande au moins 2 chaînes
        for i in range(len(selected)):
            without[i] = _max(stats.rank_normalized_rhat(np.delete(kept, i, axis=0)))
    per_chain = np.stack([stats.ess_bulk(kept[i : i + 1]) for i in range(len(selected))]) if len(rows) else np.empty((0,))

    mcmc = results.metadata.get("mcmc") or {}
    layers = posterior.get("layer")
    data = {
        "rhat": Quantity(stats.rank_normalized_rhat(kept), "1"),
        "ess_bulk": Quantity(stats.ess_bulk(kept), "1"),
        "ess_tail": Quantity(stats.ess_tail(kept), "1"),
        "mcse_mean": Quantity(stats.mcse_mean(kept), "1"),
        "rhat_max": Quantity(_max(stats.rank_normalized_rhat(kept)), "1"),
        "rhat_max_all": Quantity(_max(stats.rank_normalized_rhat(everything)), "1"),
        "chain": Quantity(np.array(selected, dtype=np.int64), "1"),
        "acceptance": Quantity(np.asarray(posterior["acceptance"], float)[selected], "1"),
        "mean_energy": Quantity(energy.mean(axis=0) if len(rows) else np.full(len(selected), np.nan), "1"),
        "chain_ess_bulk": Quantity(per_chain, "1"),
        "rhat_without_chain": Quantity(without, "1"),
        "layer": Quantity(np.array([str(n) for n in layers]) if layers is not None else np.arange(samples.shape[2]).astype(str), "1"),
        "parameter": Quantity(np.array(INVERTED_PARAMETERS), "1"),
        "samples_per_chain": Quantity(len(rows), "1"),
        "burn_in_converged": Quantity(bool(mcmc.get("burn_in_converged", False)), "1"),
        "burn_in_iterations": Quantity(int(mcmc.get("burn_in_iterations", 0)), "1"),
    }
    return Answer(data=data, provenance={"question": "convergence", "parameters": {
        "chains": None if chains is None else selected, "burn": int(burn), "thin": int(thin)}},
        warnings=list(results.warnings))
