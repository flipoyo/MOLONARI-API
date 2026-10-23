"""Registre des paramètres réglables d'un twin (BN-10) : réglages bornés, priors et bornes physiques, clés du twin.yaml.

Ne dépend d'aucun twin : tout vient du cœur (domain.compute et twin.config), donc ce que l'API annonce est ce que la
validation d'une configuration accepte.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from ..domain.compute import (
    DEFAULT_PRIORS,
    INVERTED_PARAMETERS,
    PARAMETER_LABELS,
    PHYSICAL_BOUNDS,
    SETTING_LIMITS,
    STORE_FIELDS,
    InversionConfig,
)
from ..twin.config import LAYER_KEYS, MCMC_KEYS
from .answer import Answer, Quantity

# réglage → (unité, description). Les clés sont celles de SETTING_LIMITS (un test vérifie qu'aucune ne manque).
SETTINGS = {
    "nb_cells": ("1", "Number of cells of the column (spatial resolution of the model)."),
    "sigma_t": ("°C", "Measurement uncertainty of the thermometers; sets the likelihood (twin.yaml: sensors.sigma_temperature)."),
    "nb_chains": ("1", "Number of parallel chains (DREAM needs at least 2·delta + 1)."),
    "nb_iter": ("1", "Iterations computed after the burn-in, per chain."),
    "burn_in_max": ("1", "Burn-in iterations at most, per chain; it stops earlier when the Gelman-Rubin statistic falls under gr_threshold."),
    "seed": ("1", "Seed of the random numbers (same seed, same result)."),
    "delta": ("1", "Number of chain pairs used by the proposal."),
    "n_cr": ("1", "Number of crossover values."),
    "c": ("1", "Width of the uniform noise added to the proposal."),
    "c_star": ("1", "Width of the small Gaussian noise added to the proposal."),
    "gr_threshold": ("1", "Gelman-Rubin threshold under which the burn-in is over."),
    "n_field_samples": ("1", "Total number of simulated fields (temperature and flux) kept for the quantiles; replaced by fields_per_chain when that is given."),
    "thin": ("1", "Keep the iterations that are multiples of thin (parameters, energies, acceptance, fields)."),
    "fields_per_chain": ("1", "Fields kept per chain, from 1 to the iterations kept per chain; replaces n_field_samples."),
    "space_step": ("1", "Spatial sub-sampling step of the kept fields (1: every cell)."),
    "time_step": ("1", "Temporal sub-sampling step of the kept fields (1: every date)."),
}

STORE_FIELDS_DESCRIPTION = (
    "What the inversion keeps of the sampled fields besides the quantiles: none; sensors (temperature at the three "
    "sensors); full (temperature and water flux over the whole column, needed for the probability of infiltration)."
)

LAYER_KEY_DESCRIPTIONS = {
    "name": "Name of the layer.",
    "z_low": "Depth of the bottom of the layer (m, positive downwards); the last one is the depth of the column.",
    "neg_log10_k": "Direct mode: −log₁₀ of the permeability k (k in m²).",
    "porosity": "Direct mode: porosity n.",
    "lambda_s": "Direct mode: thermal conductivity of the solid, λs (W/(m·K)).",
    "rhos_cs": "Direct mode: volumetric heat capacity of the solid, ρs·cs (J/(m³·K)).",
}

MCMC_KEY_DESCRIPTIONS = {
    "priors": "Per layer, the uniform prior range [min, max] of each inverted parameter (see the priors of this catalogue).",
    "quantiles": "Quantiles (between 0 and 1) computed on the simulated fields.",
    "store_fields": STORE_FIELDS_DESCRIPTION,
    **{name: SETTINGS[name][1] for name in SETTINGS if name not in ("nb_cells", "sigma_t", "seed")},
}


def _number(value) -> float:
    return float("nan") if value is None else float(value)


def parameters_catalog() -> Answer:
    """Réglages (bornes, défaut, unité), `store_fields`, priors (défaut et bornes physiques) et clés du twin.yaml. Les
    tableaux sont des listes de textes ou de nombres de même longueur ; nan : sans borne ou sans défaut."""
    defaults = {field.name: field.default for field in dataclasses.fields(InversionConfig)}
    names = list(SETTING_LIMITS)
    layer_keys, mcmc_keys = sorted(LAYER_KEYS), sorted(MCMC_KEYS)
    data = {
        "settings.name": Quantity(np.array(names), "1"),
        "settings.minimum": Quantity(np.array([_number(SETTING_LIMITS[n][0]) for n in names]), "1"),
        "settings.maximum": Quantity(np.array([_number(SETTING_LIMITS[n][1]) for n in names]), "1"),
        "settings.default": Quantity(np.array([_number(defaults.get(n)) for n in names]), "1"),
        "settings.unit": Quantity(np.array([SETTINGS[n][0] for n in names]), "1"),
        "settings.description": Quantity(np.array([SETTINGS[n][1] for n in names]), "1"),
        "store_fields.values": Quantity(np.array(STORE_FIELDS), "1"),
        "store_fields.default": Quantity(str(defaults["store_fields"]), "1"),
        "store_fields.description": Quantity(STORE_FIELDS_DESCRIPTION, "1"),
        "priors.parameter": Quantity(np.array(INVERTED_PARAMETERS), "1"),
        "priors.label": Quantity(np.array([PARAMETER_LABELS[n] for n in INVERTED_PARAMETERS]), "1"),
        "priors.low": Quantity(np.array([DEFAULT_PRIORS[n].low for n in INVERTED_PARAMETERS]), "1"),
        "priors.high": Quantity(np.array([DEFAULT_PRIORS[n].high for n in INVERTED_PARAMETERS]), "1"),
        "priors.physical_low": Quantity(np.array([PHYSICAL_BOUNDS[n][0] for n in INVERTED_PARAMETERS]), "1"),
        "priors.physical_high": Quantity(np.array([PHYSICAL_BOUNDS[n][1] for n in INVERTED_PARAMETERS]), "1"),
        "layer_keys.name": Quantity(np.array(layer_keys), "1"),
        "layer_keys.description": Quantity(np.array([LAYER_KEY_DESCRIPTIONS[k] for k in layer_keys]), "1"),
        "mcmc_keys.name": Quantity(np.array(mcmc_keys), "1"),
        "mcmc_keys.description": Quantity(np.array([MCMC_KEY_DESCRIPTIONS[k] for k in mcmc_keys]), "1"),
    }
    return Answer(data=data, provenance={"question": "parameters_catalog", "parameters": {}})
