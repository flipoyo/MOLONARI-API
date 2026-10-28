# ruff: noqa: F811  (la fixture data_repo, importée de test_twin_schema, est reprise en argument des tests)
"""Clés de publication du schéma molonari-twin/2.0.0 (P2.3) : `visibility`, `monitoring`, `site_label` et `report`.

Les clés inconnues restent refusées, les nouvelles sont validées, un twin sans `report` a le rapport par défaut (que
HN-02 affiche), et aucune de ces clés n'entre dans le `twin_id`. La graine n'est jamais un élément de visibilité."""


import pytest
from conftest import EXAMPLE_TWINS
from molonari_twin.questions import describe
from molonari_twin.twin.config import (
    DEFAULT_REPORT_ENTRIES,
    VISIBILITY_DEFAULTS,
    VISIBILITY_ELEMENTS,
    TwinConfigError,
    load_twin,
)
from test_twin_schema import (  # noqa: F401
    RAW_DIRECT,
    RAW_MCMC,
    codes,
    data_repo,
    write_orx,
    write_twin,
)

PUBLIC = """site_label: Rivière du Bas-Rhin, site amont
monitoring: {status: actif}
visibility:
  inversion_settings: false
  forward_parameters: true
  posterior: true
  states: true
  raw_data: false
  processed_data: true
report:
  models: {forward: twin_01, inversion: twin_05}
  entries:
    - {question: water_flux, parameters: {level: 0.9}}
    - {question: fit_quality}
    - {question: provenance}
"""


def refused(tmp_path, text):
    with pytest.raises(TwinConfigError) as error:
        load_twin(write_twin(tmp_path, text))
    return codes(error)


# Défauts --------------------------------------------------------------------------------------------------


def test_a_twin_without_the_new_keys_has_the_defaults(tmp_path, data_repo):
    config = load_twin(write_twin(tmp_path, RAW_MCMC))
    assert config.site_label == "" and config.monitoring_status is None
    assert config.visibility == VISIBILITY_DEFAULTS
    assert config.visibility == {"inversion_settings": False, "inversion_settings_public": True, "forward_parameters": False,
                                 "posterior": True, "states": False, "raw_data": False, "processed_data": False}
    assert not config.report.declared
    assert [(e.question, e.parameters) for e in config.report.entries] == [("water_flux", {"level": 0.9}), ("fit_quality", {})]
    assert config.report.forward is None and config.report.inversion is None


def test_the_default_report_is_the_water_flux_at_90_percent_and_the_fit_quality():
    assert DEFAULT_REPORT_ENTRIES == (("water_flux", {"level": 0.9}), ("fit_quality", {}))


def test_the_keys_are_read(tmp_path, data_repo):
    config = load_twin(write_twin(tmp_path, RAW_MCMC + PUBLIC))
    assert config.site_label == "Rivière du Bas-Rhin, site amont" and config.monitoring_status == "actif"
    assert config.visibility["forward_parameters"] and config.visibility["states"] and config.visibility["processed_data"]
    assert not config.visibility["inversion_settings"] and not config.visibility["raw_data"]
    assert config.visibility["posterior"] and config.visibility["inversion_settings_public"]  # défauts conservés
    assert config.report.declared and (config.report.forward, config.report.inversion) == ("twin_01", "twin_05")
    assert [e.question for e in config.report.entries] == ["water_flux", "fit_quality", "provenance"]


def test_a_direct_twin_reads_them_too(tmp_path, data_repo):
    config = load_twin(write_twin(tmp_path, RAW_DIRECT + "monitoring: {status: inactif}\nvisibility: {states: true}\n"))
    assert config.monitoring_status == "inactif" and config.visibility["states"]


def test_the_new_keys_do_not_change_the_twin_id(tmp_path, data_repo):
    reference = load_twin(write_orx(write_twin(tmp_path / "a", RAW_MCMC))).twin_id("v1")
    assert load_twin(write_orx(write_twin(tmp_path / "b", RAW_MCMC + PUBLIC))).twin_id("v1") == reference


# Compatibilité -------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("key", ["visibility: {}", "monitoring: {status: actif}", "site_label: x", "report: {entries: []}"])
def test_the_keys_are_unknown_before_2_0_0(tmp_path, key):
    folder = tmp_path / "twin_01"
    import shutil

    shutil.copytree(EXAMPLE_TWINS / "twin_01", folder)
    with (folder / "twin.yaml").open("a", encoding="utf-8") as handle:
        handle.write(f"\n{key}\n")
    with pytest.raises(TwinConfigError) as error:
        load_twin(folder)
    assert (key.split(":")[0], "E_TWIN_UNKNOWN_KEY") in codes(error)


def test_a_1_x_twin_has_the_defaults():
    config = load_twin(EXAMPLE_TWINS / "twin_01")
    assert config.visibility == VISIBILITY_DEFAULTS and not config.report.declared and config.site_label == ""


def test_an_unknown_key_of_the_new_sections_is_refused(tmp_path, data_repo):
    found = refused(tmp_path, RAW_MCMC + "monitoring: {status: actif, since: 2026}\nreport: {entries: [{question: water_flux, x: 1}], weekly: true}\n")
    assert ("monitoring.since", "E_TWIN_UNKNOWN_KEY") in found
    assert ("report.weekly", "E_TWIN_UNKNOWN_KEY") in found
    assert ("report.entries[0].x", "E_TWIN_UNKNOWN_KEY") in found


# visibility (P2.3a) ---------------------------------------------------------------------------------------------


def test_the_seed_is_not_a_visibility_element(tmp_path, data_repo):
    assert "seed" not in VISIBILITY_ELEMENTS
    assert ("visibility.seed", "E_TWIN_VISIBILITY_SEED") in refused(tmp_path, RAW_MCMC + "visibility: {seed: true}\n")


def test_an_unknown_visibility_element_is_refused(tmp_path, data_repo):
    assert ("visibility.metadata", "E_TWIN_UNKNOWN_KEY") in refused(tmp_path, RAW_MCMC + "visibility: {metadata: true}\n")


def test_a_visibility_value_is_true_or_false(tmp_path, data_repo):
    found = refused(tmp_path, RAW_MCMC + "visibility: {states: public, raw_data: 1}\n")
    assert ("visibility.states", "E_TWIN_EXPECTED_BOOLEAN") in found
    assert ("visibility.raw_data", "E_TWIN_EXPECTED_BOOLEAN") in found


def test_the_visibility_is_a_mapping(tmp_path, data_repo):
    assert ("visibility", "E_TWIN_EXPECTED_MAPPING") in refused(tmp_path, RAW_MCMC + "visibility: [states]\n")


def test_public_samples_reveal_the_public_settings(tmp_path, data_repo):
    """La PDF publique révèle nb_chains, nb_iter et thin : ces réglages ne peuvent pas rester privés (T11)."""
    assert ("visibility.inversion_settings_public", "E_TWIN_VISIBILITY_REVEALED") in refused(
        tmp_path, RAW_MCMC + "visibility: {inversion_settings_public: false}\n")
    config = load_twin(write_twin(tmp_path / "private", RAW_MCMC + "visibility: {posterior: false, inversion_settings_public: false}\n"))
    assert not config.visibility["posterior"] and not config.visibility["inversion_settings_public"]


# monitoring (P2.3b) ---------------------------------------------------------------------------------------------


def test_the_monitoring_status_is_actif_or_inactif(tmp_path, data_repo):
    assert ("monitoring.status", "E_TWIN_ONE_OF") in refused(tmp_path / "a", RAW_MCMC + "monitoring: {status: paused}\n")
    assert ("monitoring.status", "E_TWIN_REQUIRED") in refused(tmp_path / "b", RAW_MCMC + "monitoring: {}\n")
    assert ("monitoring", "E_TWIN_EXPECTED_MAPPING") in refused(tmp_path / "c", RAW_MCMC + "monitoring: actif\n")


# site_label (P2.3d) ---------------------------------------------------------------------------------------------


def test_the_site_label_does_not_give_the_point_or_the_twin_name(tmp_path, data_repo):
    assert ("site_label", "E_TWIN_SITE_LABEL_DISTINCT") in refused(tmp_path / "a", RAW_MCMC + "site_label: Station Point034\n")
    assert ("site_label", "E_TWIN_SITE_LABEL_DISTINCT") in refused(tmp_path / "b", RAW_MCMC + "site_label: twin_01\n")
    assert load_twin(write_twin(tmp_path / "c", RAW_MCMC + "site_label: Point0345 amont\n")).site_label == "Point0345 amont"


def test_the_site_label_is_one_short_line(tmp_path, data_repo):
    assert ("site_label", "E_TWIN_EMPTY") in refused(tmp_path / "a", RAW_MCMC + 'site_label: "  "\n')
    assert ("site_label", "E_TWIN_SITE_LABEL_FORMAT") in refused(tmp_path / "b", RAW_MCMC + 'site_label: "a\\nb"\n')
    assert ("site_label", "E_TWIN_SITE_LABEL_FORMAT") in refused(tmp_path / "c", RAW_MCMC + f"site_label: {'x' * 121}\n")
    assert ("site_label", "E_TWIN_EXPECTED_TEXT") in refused(tmp_path / "d", RAW_MCMC + "site_label: [a]\n")


# report (P2.3c, P2.3e, P2.3f) ---------------------------------------------------------------------------------


def test_a_report_question_must_exist_and_answer_about_a_twin(tmp_path, data_repo):
    found = refused(tmp_path, RAW_MCMC + "report:\n  entries:\n    - {question: flux}\n    - {question: parameters_catalog}\n    - {question: water_flux}\n    - {question: fit_quality}\n")
    assert ("report.entries[0].question", "E_TWIN_REPORT_QUESTION") in found
    assert ("report.entries[1].question", "E_TWIN_REPORT_QUESTION_KIND") in found


def test_report_parameters_are_checked_like_the_questions_do(tmp_path, data_repo):
    found = refused(tmp_path, RAW_MCMC + "report:\n  entries:\n    - {question: water_flux, parameters: {level: 1.5}}\n    - {question: fit_quality, parameters: {depth: 1}}\n    - {question: provenance, parameters: [1]}\n")
    fields = [field for field, code in found if code == "E_TWIN_REPORT_PARAMETERS"]
    assert "report.entries[0].parameters.level" in fields and "report.entries[1].parameters.depth" in fields
    assert ("report.entries[2].parameters", "E_TWIN_EXPECTED_MAPPING") in found


def test_a_report_entry_needs_a_question(tmp_path, data_repo):
    assert ("report.entries[0].question", "E_TWIN_REQUIRED") in refused(tmp_path, RAW_MCMC + "report: {entries: [{parameters: {}}]}\n")


def test_the_same_question_with_the_same_parameters_twice_is_refused(tmp_path, data_repo):
    text = RAW_MCMC + "report:\n  entries:\n    - {question: water_flux}\n    - {question: water_flux, parameters: {level: 0.9}}\n    - {question: water_flux, parameters: {level: 0.9}}\n    - {question: fit_quality}\n"
    assert refused(tmp_path, text) == [("report.entries[2]", "E_TWIN_REPORT_DUPLICATE")]


def test_a_published_twin_without_report_is_refused(tmp_path, data_repo):
    """P2.3f : l'absence de clé donne le défaut ; un rapport déclaré vide, ou sans le minimum, est refusé."""
    assert ("report.entries", "E_TWIN_REPORT_EMPTY") in refused(tmp_path / "a", RAW_MCMC + "report: {entries: []}\n")
    assert ("report.entries", "E_TWIN_REQUIRED") in refused(tmp_path / "b", RAW_MCMC + "report: {models: {}}\n")
    assert not load_twin(write_twin(tmp_path / "c", RAW_MCMC + "report:\n")).report.declared  # `report:` vide : le défaut
    found = refused(tmp_path / "d", RAW_MCMC + "report: {entries: [{question: provenance}]}\n")
    assert [code for _, code in found].count("E_TWIN_REPORT_MINIMUM") == 2  # water_flux et fit_quality
    assert load_twin(write_twin(tmp_path / "e", RAW_MCMC)).report.entries  # sans la clé : le défaut


def test_an_mcmc_twin_reports_the_flux_at_90_percent(tmp_path, data_repo):
    text = "report: {entries: [{question: water_flux, parameters: {level: 0.95}}, {question: fit_quality}]}\n"
    assert ("report.entries", "E_TWIN_REPORT_LEVEL") in refused(tmp_path / "mcmc", RAW_MCMC + text)
    # un twin direct n'a pas d'intervalle : le flux seul suffit
    direct = "report: {entries: [{question: water_flux, parameters: {level: 0.95}}, {question: fit_quality}]}\n"
    assert load_twin(write_twin(tmp_path / "direct", RAW_DIRECT + direct)).report.declared


def test_a_90_percent_report_needs_the_quantiles_05_and_95(tmp_path, data_repo):
    """P2.3e : mcmc.quantiles contient 0,05 et 0,95 si un rapport à 90 % est demandé."""
    cases = {"[0.05, 0.5]": "0.95", "[0.5, 0.95]": "0.05", "[0.5]": "0.05, 0.95"}
    for i, (quantiles, missing) in enumerate(cases.items()):
        text = RAW_MCMC.replace("  n_field_samples: 7\n", f"  n_field_samples: 7\n  quantiles: {quantiles}\n")
        error = [e for e in _issues(tmp_path / str(i), text) if e.code == "E_TWIN_REPORT_QUANTILES"]
        assert len(error) == 1 and error[0].field == "mcmc.quantiles" and error[0].params["quantiles"] == missing
    ok = RAW_MCMC.replace("  n_field_samples: 7\n", "  n_field_samples: 7\n  quantiles: [0.05, 0.25, 0.95]\n")
    assert load_twin(write_twin(tmp_path / "ok", ok)).mcmc.quantiles == (0.05, 0.25, 0.95)


def _issues(tmp_path, text):
    with pytest.raises(TwinConfigError) as error:
        load_twin(write_twin(tmp_path, text))
    return error.value.issues


def test_the_default_report_also_asks_for_90_percent(tmp_path, data_repo):
    """Sans clé `report`, le rapport par défaut demande le flux à 90 % : mcmc.quantiles doit contenir 0,05 et 0,95.
    Un twin direct n'a pas de quantiles : rien à vérifier."""
    text = RAW_MCMC.replace("  n_field_samples: 7\n", "  n_field_samples: 7\n  quantiles: [0.5]\n")
    assert refused(tmp_path / "mcmc", text) == [("mcmc.quantiles", "E_TWIN_REPORT_QUANTILES")]
    assert not load_twin(write_twin(tmp_path / "direct", RAW_DIRECT)).report.declared


def test_the_two_models_are_two_twin_folders(tmp_path, data_repo):
    found = refused(tmp_path / "a", RAW_MCMC + "report:\n  models: {forward: ../twin_01, inversion: 3}\n  entries: [{question: water_flux}, {question: fit_quality}]\n")
    assert ("report.models.forward", "E_TWIN_REPORT_MODEL") in found and ("report.models.inversion", "E_TWIN_REPORT_MODEL") in found
    same = RAW_MCMC + "report:\n  models: {forward: twin_01, inversion: twin_01}\n  entries: [{question: water_flux}, {question: fit_quality}]\n"
    assert ("report.models", "E_TWIN_REPORT_MODEL_SAME") in refused(tmp_path / "b", same)


# HN-02 (describe) ------------------------------------------------------------------------------------------------


def values(answer):
    return {name: quantity.value for name, quantity in answer.data.items()}


def test_describe_shows_the_default_report_when_the_key_is_absent(tmp_path, data_repo):
    data = values(describe(load_twin(write_twin(tmp_path, RAW_MCMC))))
    assert data["report_declared"] is False
    assert list(data["report_questions"]) == ["water_flux", "fit_quality"]
    assert list(data["report_parameters"]) == ['{"level": 0.9}', "{}"]
    assert data["report_forward"] == "" and data["report_inversion"] == ""
    assert data["site_label"] == "" and data["monitoring_status"] == ""
    assert {name: data[f"visibility.{name}"] for name in VISIBILITY_ELEMENTS} == VISIBILITY_DEFAULTS


def test_describe_shows_the_declared_keys(tmp_path, data_repo):
    data = values(describe(load_twin(write_twin(tmp_path, RAW_MCMC + PUBLIC))))
    assert data["report_declared"] is True and list(data["report_questions"]) == ["water_flux", "fit_quality", "provenance"]
    assert data["report_forward"] == "twin_01" and data["report_inversion"] == "twin_05"
    assert data["site_label"] == "Rivière du Bas-Rhin, site amont" and data["monitoring_status"] == "actif"
    assert data["visibility.states"] is True and data["visibility.inversion_settings"] is False


def test_describe_of_a_1_x_twin_shows_the_defaults():
    data = values(describe(load_twin(EXAMPLE_TWINS / "twin_01")))
    assert data["report_declared"] is False and data["visibility.posterior"] is True


def test_no_key_and_no_answer_gives_the_seed(tmp_path, data_repo):
    answer = describe(load_twin(write_twin(tmp_path, RAW_MCMC + PUBLIC)))
    assert not [name for name in answer.data if "seed" in name]
    assert not [element for element in VISIBILITY_ELEMENTS if "seed" in element]


def test_the_visibility_of_the_questions_is_built_from_the_key(tmp_path, data_repo):
    """`questions/visibility.py` (P2c.2) prend ses noms et sa déclaration dans la configuration : rien n'est dupliqué."""
    from molonari_twin.questions import ELEMENTS, Visibility

    assert tuple(ELEMENTS) == VISIBILITY_ELEMENTS and Visibility() == Visibility(**VISIBILITY_DEFAULTS)
    config = load_twin(write_twin(tmp_path, RAW_MCMC + PUBLIC))
    visibility = Visibility.from_config(config)
    assert visibility.to_dict() == config.visibility
    assert visibility.is_public("states") and visibility.is_public("posterior") and not visibility.is_public("raw_data")
    assert Visibility.from_config(load_twin(write_twin(tmp_path / "default", RAW_DIRECT))) == Visibility()


def test_the_public_description_gives_the_site_label_and_the_monitoring_but_not_the_declaration(tmp_path, data_repo):
    from molonari_twin.questions import public_answer

    public = public_answer("describe", describe(load_twin(write_twin(tmp_path, RAW_MCMC + PUBLIC))))
    assert public.data["site_label"].value == "Rivière du Bas-Rhin, site amont" and public.data["monitoring_status"].value == "actif"
    assert not [name for name in public.data if name.startswith(("visibility.", "report_")) or name in ("name", "point", "description")]
