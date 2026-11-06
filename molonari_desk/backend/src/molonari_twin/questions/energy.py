"""Décomposition de l'énergie d'un calcul (BN-43) : −log vraisemblance gaussienne.

    énergie = (N/2)·ln σ² + Σ r² / (2σ²)

N est le nombre de mesures comparées (3 capteurs × dates), r le résidu (simulé − mesuré) et σ l'incertitude des
thermomètres. Si σ est juste, le terme des résidus vaut en moyenne N/2 (repère `reference`) : `ratio` au-dessus de 1
dit que le modèle est moins bon que ne le dit σ. Même formule que engine.jax.forward.energy.
"""

from __future__ import annotations

import numpy as np

from ..errors import UserError
from ..twin.results import TwinResults
from .answer import Answer, Quantity
from .sensors import measurement_sigma, sensor_series


def energy_breakdown(results: TwinResults, sigma: float | None = None) -> Answer:
    """Termes de l'énergie, calculés avec les résidus enregistrés (le meilleur ajustement en mode mcmc)."""
    sigma = measurement_sigma(results, sigma)
    if not sigma > 0:
        raise UserError("E_QUESTION_SIGMA", sigma=sigma)
    _, measured, simulated = sensor_series(results)
    residual = (simulated - measured).ravel()
    residual = residual[~np.isnan(residual)]
    n = int(residual.size)
    squares = float(np.sum(residual**2))
    log_term = n / 2 * float(np.log(sigma**2))
    residual_term = squares / (2 * sigma**2)
    return Answer(
        data={
            "n_points": Quantity(n, "1"),
            "sigma": Quantity(sigma, "°C"),
            "log_term": Quantity(log_term, "1"),
            "residual_term": Quantity(residual_term, "1"),
            "energy": Quantity(log_term + residual_term, "1"),
            "implicit_rmse": Quantity(float(np.sqrt(squares / n)) if n else float("nan"), "°C"),
            "reference": Quantity(n / 2, "1"),
            "ratio": Quantity(residual_term / (n / 2) if n else float("nan"), "1"),
        },
        provenance={"question": "energy_breakdown", "parameters": {"sigma": sigma}},
        warnings=list(results.warnings),
    )
