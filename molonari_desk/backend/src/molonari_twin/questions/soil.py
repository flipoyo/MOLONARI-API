"""Propriétés du sol estimées par couche (HN-42) : k, K, n, λs et ρs·cs.

En mode mcmc : meilleure valeur, médiane, intervalle HDI et identifiabilité, calculés sur les échantillons a
posteriori. Le HDI et l'identifiabilité se calculent dans l'espace des lois a priori (−log₁₀ k, n, λs, ρs·cs), où
elles sont uniformes ; les bornes de k et de K s'en déduisent par la même transformation (k = 10^−x est monotone,
l'ordre des bornes s'inverse). En mode direct : les valeurs de la configuration, sans intervalle.

Identifiabilité = écart-type a posteriori / écart-type d'une loi uniforme sur le prior, ((haut − bas)/√12), dans la
même échelle. Proche de 1 : les données n'apprennent rien ; proche de 0 : le paramètre est bien déterminé.
"""

from __future__ import annotations

import math

import numpy as np

from ..domain.compute import INVERTED_PARAMETERS
from ..engine import stats
from ..errors import UserError
from ..twin.results import TwinResults
from .answer import Answer, Quantity

# Conductivité hydraulique K = k·ρw·g/μ. L'eau est prise à 20 °C : approximation, la viscosité dépend de la température.
WATER_DENSITY = 1000.0  # ρw, kg/m³
GRAVITY = 9.81  # g, m/s²
WATER_VISCOSITY = 1.0e-3  # μ, Pa·s, à 20 °C
HYDRAULIC_FACTOR = WATER_DENSITY * GRAVITY / WATER_VISCOSITY  # K (m/s) = k (m²) · HYDRAULIC_FACTOR

# (nom de la sortie, paramètre inversé, unité)
PROPERTIES = (
    ("k", "neg_log10_k", "m²"),
    ("K", "neg_log10_k", "m/s"),
    ("n", "porosity", "1"),
    ("lambda_s", "lambda_s", "W/(m·K)"),
    ("rhos_cs", "rhos_cs", "J/(m³·K)"),
)


def hydraulic_conductivity(permeability):
    """K (m/s) à partir de k (m²)."""
    return np.asarray(permeability, float) * HYDRAULIC_FACTOR


def _physical(name: str, x: np.ndarray) -> np.ndarray:
    """Valeur de la propriété `name` pour la valeur x du paramètre inversé (espace des priors)."""
    if name == "k":
        return 10.0 ** (-np.asarray(x, float))
    if name == "K":
        return hydraulic_conductivity(10.0 ** (-np.asarray(x, float)))
    return np.asarray(x, float)


def _best(results: TwinResults, layer_names: list[str]) -> np.ndarray:
    """Meilleures valeurs (L, 4) dans l'espace des priors : celles du calcul, ou celles de la configuration (mode direct)."""
    mcmc = results.metadata.get("mcmc") or {}
    best = mcmc.get("best_parameters")
    if best:
        return np.array([[best[layer][name] for name in INVERTED_PARAMETERS] for layer in layer_names], float)
    layers = results.metadata.get("config", {}).get("layers", [])
    return np.array([[layer[name] for name in INVERTED_PARAMETERS] for layer in layers], float)


def _prior_widths(results: TwinResults, count: int) -> np.ndarray:
    """Écart-type d'une loi uniforme sur le prior de chaque paramètre et couche (L, 4) ; nan sans priors enregistrés."""
    priors = results.metadata.get("config", {}).get("mcmc", {}).get("priors")
    if not priors or len(priors) != count:
        return np.full((count, len(INVERTED_PARAMETERS)), np.nan)
    return np.array([[(p[name][1] - p[name][0]) / math.sqrt(12) for name in INVERTED_PARAMETERS] for p in priors], float)


def soil_properties(results: TwinResults, level: float = 0.9, layer_names: list[str] | None = None) -> Answer:
    """Pour chaque propriété (k, K, n, λs, ρs·cs) et chaque couche : `best` (value) et HDI au niveau `level`
    (interval) en mode mcmc ; `<nom>_median` et `<nom>_identifiability` en mode mcmc seulement."""
    if not 0.0 < level < 1.0:
        raise UserError("E_QUESTION_LEVEL", level=level)
    posterior = results.posterior
    if posterior is not None:
        names = [str(name) for name in posterior["layer"]]
    else:
        count = len(results.metadata.get("config", {}).get("layers", []))
        names = list(layer_names) if layer_names and len(layer_names) == count else [f"layer_{i + 1}" for i in range(count)]
    best = _best(results, names)
    data: dict[str, Quantity] = {"layers": Quantity(np.array(names), "1")}
    if posterior is not None:
        samples = np.asarray(posterior["samples"], float)  # (I, N, L, 4)
        flat = samples.reshape(-1, len(names), len(INVERTED_PARAMETERS))
        low, high = stats.hdi(flat, level, axis=0)
        median = np.median(flat, axis=0)
        identifiability = flat.std(axis=0, ddof=1) / _prior_widths(results, len(names))
        data["level"] = Quantity(float(level), "1")
    for name, parameter, unit in PROPERTIES:
        k = INVERTED_PARAMETERS.index(parameter)
        if posterior is None:
            data[name] = Quantity(_physical(name, best[:, k]), unit)
            continue
        bounds = sorted_pair(_physical(name, low[:, k]), _physical(name, high[:, k]))
        data[name] = Quantity(_physical(name, best[:, k]), unit, bounds)
        data[f"{name}_median"] = Quantity(_physical(name, median[:, k]), unit)
        data[f"{name}_identifiability"] = Quantity(identifiability[:, k], "1")
    return Answer(data=data, provenance={"question": "soil_properties", "parameters": {"level": float(level)}},
                  warnings=list(results.warnings))


def sorted_pair(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(bas, haut) élément par élément : la transformation de k renverse l'ordre des bornes."""
    return np.minimum(a, b), np.maximum(a, b)
