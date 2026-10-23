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


def _max(values: np.ndarray) -> float:
    return float(np.nan) if np.isnan(values).all() else float(np.nanmax(values))


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
