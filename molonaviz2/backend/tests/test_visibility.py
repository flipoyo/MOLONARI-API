"""Visibilité des questions et projections publiques (P2c.2) : chaque question déclare ce qu'elle expose, le contrôle
refuse ce qui est privé, les projections publiques de HN-02 et HN-43 suivent leur liste blanche, et la graine
n'entre dans aucune projection ni dans HN-02 (patch « accès » du 26/10)."""

import copy
import json
from pathlib import Path

import pytest
from molonari_twin.api import (
    CATALOG,
    DEV,
    TwinAPI,
    check_publishable,
    exposed_elements,
    publishable_answer,
)
from molonari_twin.errors import UserError
from molonari_twin.questions import ELEMENTS, PUBLIC_DATA, Visibility, public_answer
from molonari_twin.questions.public import PUBLIC_ENVIRONMENT
from molonari_twin.questions.visibility import ELEMENT_MODES

GATE_FILE = Path(__file__).resolve().parents[1] / "docs" / "API_V1.md"

# Colonne « Accès » de CATALOGUE_QUESTIONS.md (26/10) : rapport, rapport si public, dév.
ACCESS = {
    "HN-02": "report", "HN-10": "report", "HN-11": "report", "HN-12": "report", "HN-41": "report", "HN-43": "report",
    "HN-45": "report", "HN-20": "report_if_public", "HN-30": "report_if_public", "HN-33": "report_if_public",
    "HN-40": "report_if_public", "HN-42": "report_if_public", "BN-30": "report_if_public", "BN-32": "report_if_public",
    "BN-34": "report_if_public", "BN-31": "dev", "BN-36": "dev", "BN-40": "dev", "BN-43": "dev", "BN-10": "dev",
    "BN-11": "dev", "BN-03": "dev", "BN-04": "dev",
}


@pytest.fixture
def twins(computed_twins):
    return computed_twins


def by_id(question_id):
    return next(q for q in CATALOG if q.id == question_id)


# Déclaration d'exposition ---------------------------------------------------------------------------


def test_every_question_declares_what_it_exposes():
    for question in CATALOG:
        assert question.exposes == DEV or isinstance(question.exposes, tuple), question.id
        if question.exposes != DEV:
            assert set(question.exposes) <= set(ELEMENTS), question.id


def test_exposure_is_required_by_the_catalog_entry():
    from molonari_twin.api.catalog import Question

    assert "exposes" in Question.__dataclass_fields__
    with pytest.raises(TypeError):
        Question("x", "HN-99", "discovery", "label", "description.", ("D",), None, "none", lambda t, a: None)


def test_access_column_is_derived_from_the_declaration():
    assert {q.ref for q in CATALOG} == set(ACCESS)
    for question in CATALOG:
        assert question.access == ACCESS[question.ref], question.ref


def test_api_reference_states_the_access_of_each_question():
    text = GATE_FILE.read_text(encoding="utf-8")
    for question in CATALOG:
        block = text.split(f"### `{question.id}` ({question.ref})", 1)[1].split("###", 1)[0]
        assert "- Accès : " in block, question.id


def test_seed_is_not_an_element_of_visibility():
    assert "seed" not in ELEMENTS
    assert "seed" not in Visibility.__dataclass_fields__
    with pytest.raises(KeyError):
        Visibility().is_public("seed")


# Visibilité par défaut et contrôle --------------------------------------------------------------------


def test_default_visibility_is_private_except_the_pdf():
    default = Visibility()
    assert default.posterior is True and default.inversion_settings_public is True
    for element in ("inversion_settings", "forward_parameters", "states", "raw_data", "processed_data"):
        assert getattr(default, element) is False, element
    assert set(default.to_dict()) == set(ELEMENTS)


def test_exposure_depends_on_the_mode():
    assert exposed_elements(by_id("soil_properties"), "direct") == ("forward_parameters",)
    assert exposed_elements(by_id("soil_properties"), "mcmc") == ("posterior",)
    assert ELEMENT_MODES["posterior"] == "M"


def test_developer_only_questions_are_never_published():
    with pytest.raises(UserError) as error:
        check_publishable(by_id("residuals"), Visibility(posterior=True, states=True), "M")
    assert error.value.code == "E_REPORT_DEV_ONLY"


def test_private_element_is_refused_and_named():
    with pytest.raises(UserError) as error:
        check_publishable(by_id("temperature"), Visibility(), "direct")
    assert error.value.code == "E_REPORT_PRIVATE" and "states" in error.value.params["elements"]


def test_public_element_lets_the_answer_in():
    check_publishable(by_id("temperature"), Visibility(states=True), "direct")  # ne lève pas


def test_private_element_of_the_other_mode_does_not_block():
    # la PDF d'un twin direct n'existe pas : posterior privée n'empêche pas soil_properties en mode direct
    check_publishable(by_id("soil_properties"), Visibility(posterior=False, forward_parameters=True), "direct")


def test_a_question_without_projection_is_refused_even_when_public(twins):
    api = TwinAPI(twins)
    answer = api.ask("twin_01", "temperature", {"z": 0.2})
    with pytest.raises(UserError) as error:
        publishable_answer(by_id("temperature"), answer, Visibility(states=True), "direct")
    assert error.value.code == "E_REPORT_NO_PROJECTION"


# Projections publiques ---------------------------------------------------------------------------------


def test_public_answer_keeps_only_the_whitelist(twins):
    api = TwinAPI(twins)
    for ref, mode in (("twin_01", "direct"), ("twin_03", "mcmc")):
        for question in api.questions(ref):
            if question.id not in PUBLIC_DATA:
                continue
            answer = api.ask(ref, question.id)
            public = publishable_answer(question, answer, Visibility(), mode, instance=f"{ref}@test")
            allowed = set(PUBLIC_DATA[question.id]) | (set(PUBLIC_ENVIRONMENT) if question.id == "provenance" else set())
            assert set(public.data) <= allowed, (ref, question.id)
            assert set(public.provenance) <= {"question", "ref", "parameters", "instance"}, (ref, question.id)


def test_projection_does_not_change_the_original_answer(twins):
    api = TwinAPI(twins)
    answer = api.ask("twin_01", "describe")
    before = copy.deepcopy(answer.provenance), set(answer.data)
    public_answer("describe", answer)
    assert (answer.provenance, set(answer.data)) == before


def test_hn02_public_projection_has_no_seed_name_point_or_settings(twins):
    api = TwinAPI(twins)
    for ref in ("twin_01", "twin_03"):
        answer = api.ask(ref, "describe")
        public = publishable_answer(by_id("describe"), answer, Visibility(), "mcmc" if ref == "twin_03" else "direct")
        assert not {"seed", "name", "point", "description", "sigma_temperature", "nb_cells", "store_fields",
                    "burn_in_max"} & set(public.data), ref
        text = json.dumps({k: str(v.value) for k, v in public.data.items()})
        assert "seed" not in text and str(twins) not in text


def test_hn02_has_no_seed_for_the_developer_either(twins):
    api = TwinAPI(twins)
    for ref in ("twin_01", "twin_03"):
        assert "seed" not in api.ask(ref, "describe").data, ref


def test_hn43_public_projection_is_the_whitelist(twins):
    api = TwinAPI(twins)
    answer = api.ask("twin_01", "provenance")
    public = publishable_answer(by_id("provenance"), answer, Visibility(), "direct", instance="twin_01@abc")
    assert set(public.data) <= set(PUBLIC_DATA["provenance"]) | set(PUBLIC_ENVIRONMENT)
    assert "code_version" in public.data
    assert {"jax", "numpy"} <= set(public.data)  # versions de librairies
    assert "platform" in public.data  # plateforme
    text = json.dumps({"data": {k: str(v.value) for k, v in public.data.items()}, "provenance": public.provenance}, default=str)
    for forbidden in ("remote", "data_file", "data_sha256", "twin_id", "reproduce", "seed", str(twins), "origin", "commit"):
        assert forbidden not in text, forbidden
    assert public.provenance["instance"] == "twin_01@abc"


def test_hn43_keeps_everything_for_the_developer(twins):
    data = TwinAPI(twins).ask("twin_01", "provenance").data
    assert {"twin_id", "data_file", "data_sha256", "reproduce"} <= set(data)


def test_the_gate_list_is_not_empty_and_matches_the_catalog():
    assert set(PUBLIC_DATA) <= {q.id for q in CATALOG if q.access != "dev"}
    assert {q.id for q in CATALOG if q.access == "report"} - set(PUBLIC_DATA) <= {"reliability"}
    assert {"describe", "provenance"} <= set(PUBLIC_DATA)


def test_no_public_projection_for_the_seed_anywhere():
    for name, fields in PUBLIC_DATA.items():
        assert "seed" not in fields, name



# Questions « rapport si public » livrées (P2c.2e) ------------------------------------------------------------

# Éléments exposés par chaque question de la colonne « rapport si public », par mode du twin ('D' direct, 'M' mcmc).
IF_PUBLIC = {
    "heat_budget": {"D": {"states"}, "M": {"states"}},  # HN-20
    "temperature": {"D": {"states"}, "M": {"states"}},  # HN-30
    "sensor_intervals": {"D": {"raw_data", "processed_data", "states"}, "M": {"raw_data", "processed_data", "states"}},  # HN-33
    "fit_quality": {"D": {"raw_data", "processed_data"}, "M": {"raw_data", "processed_data"}},  # HN-40
    "soil_properties": {"D": {"forward_parameters"}, "M": {"posterior"}},  # HN-42
    "fields": {"D": {"states"}, "M": {"states"}},  # BN-30
    "posterior_samples": {"D": set(), "M": {"posterior", "inversion_settings_public"}},  # BN-32 (mode M seulement)
    "chain_data": {"D": set(), "M": {"posterior", "inversion_settings_public"}},  # BN-34 (mode M seulement)
}
MODE_NAMES = {"D": "direct", "M": "mcmc"}


def test_if_public_table_is_the_catalog_column():
    assert {q.id for q in CATALOG if q.access == "report_if_public"} == set(IF_PUBLIC)
    assert {q.ref for q in CATALOG if q.id in IF_PUBLIC} == {
        "HN-20", "HN-30", "HN-33", "HN-40", "HN-42", "BN-30", "BN-32", "BN-34"}


@pytest.mark.parametrize("question_id", sorted(IF_PUBLIC))
def test_if_public_question_declares_its_exposure(question_id):
    question = by_id(question_id)
    assert question.exposes != DEV
    assert set(question.exposes) <= set(ELEMENTS)
    for mode, expected in IF_PUBLIC[question_id].items():
        assert set(exposed_elements(question, MODE_NAMES[mode])) == expected, (question_id, mode)


@pytest.mark.parametrize("question_id", sorted(IF_PUBLIC))
def test_if_public_question_is_refused_while_an_exposed_element_is_private(question_id):
    question = by_id(question_id)
    for mode, expected in IF_PUBLIC[question_id].items():
        if mode not in question.modes:
            continue
        for element in expected:
            only_this_private = Visibility(**{**{e: True for e in ELEMENTS}, element: False})
            with pytest.raises(UserError) as error:
                check_publishable(question, only_this_private, MODE_NAMES[mode])
            assert error.value.code == "E_REPORT_PRIVATE", (question_id, mode, element)
            assert element in error.value.params["elements"]


@pytest.mark.parametrize("question_id", sorted(IF_PUBLIC))
def test_if_public_question_passes_when_its_elements_are_public(question_id):
    question = by_id(question_id)
    for mode in IF_PUBLIC[question_id]:
        if mode not in question.modes:
            continue
        everything_but_that = Visibility(**{e: e in IF_PUBLIC[question_id][mode] for e in ELEMENTS})
        check_publishable(question, everything_but_that, MODE_NAMES[mode])  # ne lève pas


@pytest.mark.parametrize("question_id", sorted(IF_PUBLIC))
def test_if_public_question_needs_its_whitelist_to_enter_a_report(question_id, twins):
    # aucune de ces questions n'a encore de projection (P2c.3) : même avec tout public, la réponse n'entre pas telle quelle
    question = by_id(question_id)
    assert question_id not in PUBLIC_DATA
    ref, mode = ("twin_03", "mcmc") if "D" not in question.modes else ("twin_01", "direct")
    answer = TwinAPI(twins).ask(ref, question_id, {"z": 0.2} if question_id == "temperature" else None)
    with pytest.raises(UserError) as error:
        publishable_answer(question, answer, Visibility(**{e: True for e in ELEMENTS}), mode)
    assert error.value.code == "E_REPORT_NO_PROJECTION"


def test_default_visibility_lets_only_the_pdf_questions_in():
    default = Visibility()
    allowed = set()
    for question in CATALOG:
        if question.id not in IF_PUBLIC:
            continue
        for mode in question.modes:
            try:
                check_publishable(question, default, MODE_NAMES[mode])
            except UserError:
                continue
            allowed.add((question.id, mode))
    assert allowed == {("posterior_samples", "M"), ("chain_data", "M"), ("soil_properties", "M")}


def test_no_question_exposes_the_seed():
    for question in CATALOG:
        if question.exposes != DEV:
            assert "seed" not in question.exposes, question.id
    assert "seed" not in by_id("describe").outputs
