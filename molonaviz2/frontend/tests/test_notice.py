"""Notice des paramètres : complète, juste (valeurs lues dans le code), consultable depuis l'interface."""

import json
from dataclasses import fields
from pathlib import Path

import pytest
from molonari_twin.domain.compute import (
    DEFAULT_PRIORS,
    INVERTED_PARAMETERS,
    PHYSICAL_BOUNDS,
    SETTING_LIMITS,
    ForwardConfig,
    InversionConfig,
    LayerSpec,
)
from molonari_twin.twin.config import LAYER_KEYS, MCMC_KEYS, TwinConfig
from molonaviz2.notice import BY_KEY, ENTRIES, SECTIONS, limits_text, number

CATALOG = json.loads((Path(__file__).resolve().parents[2] / "backend" / "src" / "molonari_twin" / "i18n" / "fr.json").read_text(encoding="utf-8"))


def test_entries_are_unique_and_grouped_in_known_sections():
    assert len(BY_KEY) == len(ENTRIES)
    assert {entry.section for entry in ENTRIES} == set(SECTIONS)
    for entry in ENTRIES:
        assert entry.summary, entry.key


def test_every_text_has_a_french_translation():
    for entry in ENTRIES:
        for text in (entry.section, entry.summary, entry.details):
            assert not text or text in CATALOG, (entry.key, text)


@pytest.mark.parametrize("name", list(SETTING_LIMITS))
def test_limits_and_defaults_come_from_the_domain(name):
    key = "numerics.nb_cells" if name == "nb_cells" else f"inversion.{name}"
    entry = BY_KEY[key]
    assert entry.limits() == limits_text(*SETTING_LIMITS[name])
    if name == "fields_per_chain":  # pas de valeur par défaut : n_field_samples sert
        assert InversionConfig(layers=[]).fields_per_chain is None and entry.default()
    elif name != "sigma_t":  # σT par défaut : celui du thermomètre
        default = ForwardConfig(layers=[]).nb_cells if name == "nb_cells" else getattr(InversionConfig(layers=[]), name)
        assert entry.default() == number(default)


@pytest.mark.parametrize("name", INVERTED_PARAMETERS)
def test_layer_parameters_show_default_prior_and_physical_range(name):
    entry = BY_KEY[f"layers.{name}"]
    prior = DEFAULT_PRIORS[name]
    assert entry.default() == f"{number(getattr(LayerSpec('', 1.0), name))} (direct model) · prior {number(prior.low)} – {number(prior.high)} (inversion)"
    assert entry.limits() == f"physical range {limits_text(*PHYSICAL_BOUNDS[name])}"


def test_twin_defaults_come_from_twin_config():
    defaults = {f.name: f.default for f in fields(TwinConfig)}
    assert BY_KEY["twin.offset"].default() == str(defaults["offset"])
    assert BY_KEY["twin.sigma_pressure"].default() == str(defaults["sigma_pressure"])


def test_every_twin_key_of_layers_and_mcmc_is_documented():
    documented = " ".join(entry.twin_key for entry in ENTRIES)
    for key in LAYER_KEYS:
        assert f"layers[].{key}" in documented, key
    for key in MCMC_KEYS:
        assert f"mcmc.{key}" in documented, key


def test_shared_limits_match_the_validation_of_the_configurations():
    low, high = SETTING_LIMITS["nb_cells"]
    layers = [LayerSpec("a", 0.4)]
    assert ForwardConfig(layers, low).validate(0.4) == [] and ForwardConfig(layers, high).validate(0.4) == []
    assert ForwardConfig(layers, low - 1).validate(0.4) and ForwardConfig(layers, high + 1).validate(0.4)


def test_cleanup_constants_appear_in_the_outlier_entry():
    from molonari_twin.processing.cleanup import IQR_FACTOR, ZSCORE_LIMIT

    details = BY_KEY["cleaning.outliers"].text("details")
    assert f"Q1 − {number(IQR_FACTOR)}·IQR" in details and f"more than {number(ZSCORE_LIMIT)} standard deviations" in details
