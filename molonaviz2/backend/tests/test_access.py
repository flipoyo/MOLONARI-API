"""Niveaux d'accès aux questions (T11) : le niveau `user` n'interroge que l'eau, la chaleur et la température, par
liste blanche ; l'admin garde tout. Sans authentification : un guide d'usage, pas une sécurité."""

import json

import pytest
from molonari_twin.api import CATALOG, USER_QUESTIONS, LeveledAPI, TwinAPI
from molonari_twin.api.access import USER_DATA, WATER_QUESTIONS
from molonari_twin.contract.codec import answer_to_json, answer_to_npz
from molonari_twin.errors import UserError
from molonari_twin.questions import PUBLIC_DATA, Visibility
from molonari_twin.questions.public import PUBLIC_PROVENANCE
from molonari_twin.twin.config import load_twin
from molonari_twin.twin.run import twin_status

FIVE = ["water_flux", "exchange_direction", "exchanged_volume", "heat_budget", "temperature"]
PARAMS = {"temperature": {"z": 0.2}}
# Clés qui désigneraient la graine, un chemin, le twin, le point ou les échantillons.
FORBIDDEN_KEYS = {"seed", "samples", "twin", "twin_id", "results", "point", "name", "path", "folder", "repositories"}


@pytest.fixture(scope="module")
def user(computed_twins):
    return LeveledAPI(TwinAPI(computed_twins), "user")


@pytest.fixture(scope="module")
def admin(computed_twins):
    return LeveledAPI(TwinAPI(computed_twins), "admin")


def refused(call, name):
    with pytest.raises(UserError) as error:
        call()
    assert error.value.code == "E_ACCESS_ADMIN_ONLY" and error.value.params == {"name": name}


def test_the_user_level_lists_exactly_five_questions(user, admin):
    assert USER_QUESTIONS == tuple(FIVE) and [q.id for q in user.questions()] == FIVE
    assert [q.id for q in admin.questions()] == [q.id for q in CATALOG]
    for ref in ("twin_01", "twin_03"):  # direct et mcmc : les cinq s'appliquent aux deux
        assert [q.id for q in user.questions(ref)] == FIVE
    assert "sensor_intervals" not in FIVE  # elle révèle les mesures


def test_an_unknown_level_is_refused(computed_twins):
    with pytest.raises(ValueError):
        LeveledAPI(TwinAPI(computed_twins), "root")


@pytest.mark.parametrize("ref", ["twin_01", "twin_03"])
@pytest.mark.parametrize("question", [q.id for q in CATALOG])
def test_only_the_five_questions_answer_at_the_user_level(user, admin, computed_twins, ref, question):
    """Tout le catalogue, sur un twin direct et sur un twin mcmc."""
    if question not in FIVE:
        refused(lambda: user.ask(ref, question, PARAMS.get(question)), question)
        refused(lambda: user.ask_global(question), question)
        return
    answer = user.ask(ref, question, PARAMS.get(question))
    full = admin.ask(ref, question, PARAMS.get(question))
    assert set(answer.data) <= set(PUBLIC_DATA[question] if question in WATER_QUESTIONS else USER_DATA[question])
    assert answer.data and set(answer.data) <= set(full.data)
    assert set(answer.provenance) <= set(PUBLIC_PROVENANCE) and answer.provenance["question"] == question
    config = load_twin(computed_twins / ref)
    twin_id = twin_status(computed_twins / ref).twin_id
    for text in (json.dumps(answer_to_json(answer)), answer_to_npz(answer).decode("latin-1")):
        assert str(computed_twins) not in text and twin_id not in text and config.point not in text
    assert not FORBIDDEN_KEYS & set(answer.data) and not FORBIDDEN_KEYS & set(answer.provenance)
    assert not FORBIDDEN_KEYS & set(answer.provenance.get("parameters", {}))
    # les valeurs gardées sont celles de la réponse complète
    for name, quantity in answer.data.items():
        assert quantity.unit == full.data[name].unit


def test_the_water_answers_are_the_public_projection(user, admin):
    from molonari_twin.questions import public_answer

    for question in WATER_QUESTIONS:
        assert answer_to_json(user.ask("twin_03", question)) == answer_to_json(public_answer(question, admin.ask("twin_03", question)))
    assert "samples" not in user.ask("twin_03", "water_flux").data  # présents, par échantillon, pour l'admin
    assert "samples" in admin.ask("twin_03", "water_flux").data


def test_heat_and_temperature_have_their_own_lists_and_stay_out_of_the_public_report(user, admin):
    assert set(USER_DATA) == {"heat_budget", "temperature"}
    assert not set(USER_DATA) & set(PUBLIC_DATA)  # le rapport public ne change pas : ces deux questions y restent refusées
    heat = user.ask("twin_01", "heat_budget").data
    assert set(heat) == set(USER_DATA["heat_budget"])
    temperature = user.ask("twin_01", "temperature", {"z": 0.2}).data
    assert set(temperature) == {"temperature", "depth", "time"}
    from molonari_twin.api import publishable_answer

    for question in USER_DATA:
        with pytest.raises(UserError) as error:
            publishable_answer(admin.question(question), admin.ask("twin_01", question, PARAMS.get(question)), Visibility(), "D")
        assert error.value.code in ("E_REPORT_PRIVATE", "E_REPORT_NO_PROJECTION")


def test_temperature_still_needs_a_depth_or_a_date(user):
    with pytest.raises(UserError) as error:
        user.ask("twin_01", "temperature")
    assert error.value.code == "E_QUESTION_PARAMS"
    assert user.ask("twin_03", "temperature", {"z": 0.2, "t": "2016-07-01T12:00"}).data["temperature"].unit == "°C"


def test_the_admin_level_changes_nothing(admin, computed_twins):
    api = TwinAPI(computed_twins)
    for ref, question in (("twin_01", "heat_budget"), ("twin_03", "residuals")):
        assert answer_to_json(admin.ask(ref, question)) == answer_to_json(api.ask(ref, question))
    assert admin.questions("twin_03") == api.questions("twin_03")
    admin.require_admin("twin.run")


def test_the_refusal_comes_before_the_other_checks(user):
    refused(lambda: user.ask("twin_99", "residuals", {"bogus": 1}), "residuals")  # ni twin, ni paramètres vérifiés
    refused(lambda: user.ask("twin_01", "HN-33"), "sensor_intervals")  # la référence du plan vaut l'identifiant
    refused(lambda: user.require_admin("twin.run"), "twin.run")
    with pytest.raises(UserError) as error:
        user.ask("twin_01", "no_such_question")
    assert error.value.code == "E_QUESTION_UNKNOWN"
    with pytest.raises(UserError) as error:
        user.ask("twin_99", "water_flux")
    assert error.value.code == "E_TWIN_NOT_FOUND"
