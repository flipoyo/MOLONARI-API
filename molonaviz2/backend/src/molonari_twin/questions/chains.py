"""Données par chaîne : traces des paramètres et des énergies, acceptation et réinitialisations."""

from __future__ import annotations

import numpy as np

from ..domain.compute import INVERTED_PARAMETERS, PARAMETER_UNITS
from ..engine.results import InversionResult
from ..errors import UserError
from ..twin.results import TwinResults
from .answer import Answer, Quantity


def _posterior(result: InversionResult | TwinResults) -> dict[str, np.ndarray]:
    """Tableaux a posteriori, avec le numéro d'itération calculée (après le burn-in) de chaque itération gardée."""
    if isinstance(result, TwinResults):
        if result.posterior is None:
            raise UserError("E_QUESTION_NO_POSTERIOR")
        return result.posterior
    kept, nb_chains = result.energies.shape
    return {
        "samples": result.samples,
        "energy": result.energies,
        "acceptance": result.acceptance,
        "resets": result.resets if result.resets is not None else np.zeros(nb_chains, np.int64),
        "iteration": (np.arange(kept) + 1) * result.thin,  # comme posterior.npz
    }


def chain_data(result: InversionResult | TwinResults, chains=None, thin: int = 1, iterations=None) -> Answer:
    """Paramètres (un tableau (I′, N′, L) par paramètre inversé), énergies (I′, N′), acceptation et
    réinitialisations (N′,) des chaînes choisies.

    `thin` garde une itération gardée sur `thin` ([thin − 1 :: thin]), comme le `thin` du calcul (A2) :
    l'itération i (à partir de 0) est l'itération calculée (i + 1)·thin_calcul·thin. `iterations`
    = (début, fin), bornes incluses (None : sans borne), restreint ces numéros. Avec toutes les chaînes
    et toute la plage, count = nb_chains × ⌊nb_iter / (thin_calcul·thin)⌋ (BN-36)."""
    if isinstance(thin, bool) or not isinstance(thin, (int, np.integer)) or thin < 1:
        raise UserError("E_QUESTION_THIN", thin=thin)
    posterior = _posterior(result)
    samples = np.asarray(posterior["samples"])
    nb_chains = samples.shape[1]
    selected = list(range(nb_chains)) if chains is None else [int(c) for c in chains]
    unknown = [c for c in selected if not 0 <= c < nb_chains]
    if unknown:
        raise UserError("E_QUESTION_CHAINS", chain=unknown[0], n=nb_chains)

    rows = np.arange(samples.shape[0])[thin - 1 :: thin]
    numbers = np.asarray(posterior["iteration"])[rows]
    if iterations is not None:
        start, stop = iterations
        inside = np.ones(len(rows), bool)
        if start is not None:
            inside &= numbers >= start
        if stop is not None:
            inside &= numbers <= stop
        rows, numbers = rows[inside], numbers[inside]

    kept = samples[np.ix_(rows, selected)]  # (I′, N′, L, 4)
    data = {"iteration": Quantity(numbers, "1"), "chain": Quantity(np.array(selected, dtype=np.int64), "1")}
    for k, name in enumerate(INVERTED_PARAMETERS):
        data[f"samples.{name}"] = Quantity(kept[..., k], PARAMETER_UNITS[name])
    data.update({
        "energy": Quantity(np.asarray(posterior["energy"])[np.ix_(rows, selected)], "1"),
        "acceptance": Quantity(np.asarray(posterior["acceptance"])[selected], "1"),
        "resets": Quantity(np.asarray(posterior["resets"])[selected], "1"),
        "count": Quantity(len(rows) * len(selected), "1"),
    })
    return Answer(
        data=data,
        provenance={"question": "chain_data", "parameters": {
            "chains": None if chains is None else selected, "thin": int(thin),
            "iterations": None if iterations is None else list(iterations)}},
    )
