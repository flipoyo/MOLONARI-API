"""Verdict de fiabilité (HN-41) : chaque critère passe du vert à l'orange puis au rouge selon les seuils, le verdict est
le pire des critères, les seuils se surchargent (paramètre, MOLONARI_THRESHOLDS) et ne sont pas validés."""

import dataclasses

import numpy as np
import pytest
from conftest import synthetic_inputs
from molonari_twin.domain.compute import ForwardConfig, LayerSpec
from molonari_twin.engine.api import run_forward
from molonari_twin.errors import UserError
from molonari_twin.questions import Answer, reliability
from molonari_twin.questions.thresholds import ReliabilityThresholds, resolve_thresholds
from test_questions import (
    PRIOR,
    results_of,
    uniform_samples,
    with_posterior,
    with_residuals,
)

NO_ENVIRONMENT: dict = {}


@pytest.fixture(scope="module")
def direct(tmp_path_factory):
    forward = run_forward(synthetic_inputs(nb_times=80), ForwardConfig([LayerSpec("a", 0.4)], nb_cells=20))
    return dataclasses.replace(results_of(tmp_path_factory.mktemp("reliable"), forward), warnings=[])  # provenance complète


def ask(results, **kwargs) -> dict:
    """Niveau de chaque critère."""
    answer = reliability(results, environment=NO_ENVIRONMENT, **kwargs)
    return dict(zip(answer.data["criteria"].value, answer.data["levels"].value, strict=True))


def code_of(call):
    with pytest.raises(UserError) as error:
        call()
    return error.value.code


def provenance_problem():
    from molonari_twin.errors import Issue

    return Issue(None, "W_PROVENANCE_NO_REMOTE", target="twin")


# Un critère à la fois -----------------------------------------------------------------------------


def test_a_direct_result_with_a_good_fit_is_green(direct):
    answer = reliability(with_residuals(direct, 0.0), environment=NO_ENVIRONMENT)
    assert answer.data["verdict"].value == "green"
    assert list(answer.data["criteria"].value) == ["fit", "data", "provenance"]  # pas de convergence en mode direct
    assert not [w for w in answer.warnings if w.code.startswith("W_RELIABILITY_")]


@pytest.mark.parametrize("offset, expected", [(0.1, "green"), (0.2, "orange"), (0.4, "red")])  # σ = 0,1 : 1,5σ vert, 3σ orange
def test_fit_follows_the_rmse_over_sigma(direct, offset, expected):
    levels = ask(with_residuals(direct, offset * 0.99 if expected == "green" else offset * 1.001))
    assert levels["fit"] == expected


def test_fit_boundaries_belong_to_the_better_level(direct):
    assert ask(with_residuals(direct, 0.15 - 1e-12))["fit"] == "green"
    assert ask(with_residuals(direct, 0.3 - 1e-12))["fit"] == "orange"


def drop_rows(results, fraction):
    """Résultats dont une part de la fin des mesures manque (valeurs non finies)."""
    frame = results.residuals.copy()
    n = int(len(frame) * fraction)
    for i in (1, 2, 3):
        frame.loc[frame.index[:n], f"t{i}_measured"] = np.nan
    return dataclasses.replace(results, residuals=frame)


@pytest.mark.parametrize("fraction, expected", [(0.0, "green"), (0.04, "green"), (0.1, "orange"), (0.3, "red")])
def test_data_gaps(direct, fraction, expected):
    assert ask(drop_rows(with_residuals(direct, 0.0), fraction))["data"] == expected


def test_missing_dates_are_gaps_too(direct):
    frame = with_residuals(direct, 0.0).residuals
    kept = frame.drop(frame.index[10:40])  # 30 dates sur 80 absentes
    assert ask(dataclasses.replace(direct, residuals=kept))["data"] == "red"


def test_provenance_is_green_when_complete_and_orange_otherwise(direct):
    assert ask(direct)["provenance"] == "green"
    assert ask(dataclasses.replace(direct, warnings=[provenance_problem()]))["provenance"] == "orange"
    assert ask(dataclasses.replace(direct, warnings=[provenance_problem()] * 3))["provenance"] == "orange"  # jamais rouge


# Mode mcmc ----------------------------------------------------------------------------------------


def mcmc(direct, samples=None, **kwargs):
    return with_posterior(with_residuals(direct, 0.0), uniform_samples(kept=400, chains=5) if samples is None else samples, **kwargs)


def test_converged_chains_are_green_but_not_identifiable_when_equal_to_the_prior(direct):
    levels = ask(mcmc(direct))
    assert levels["convergence"] == "green"
    assert levels["identifiability"] == "red"  # échantillons égaux au prior : rapport 1
    assert reliability(mcmc(direct), environment=NO_ENVIRONMENT).data["verdict"].value == "red"


def test_identifiability_goes_from_green_to_red_with_the_posterior_width(direct):
    def narrow(scale):
        samples = uniform_samples(kept=400, chains=5)
        for k, (low, high) in enumerate(PRIOR.values()):
            samples[..., k] = (low + high) / 2 + (samples[..., k] - (low + high) / 2) * scale
        return samples

    assert ask(mcmc(direct, narrow(0.3)))["identifiability"] == "green"  # rapport 0,3
    assert ask(mcmc(direct, narrow(0.65)))["identifiability"] == "orange"
    assert ask(mcmc(direct, narrow(0.9)))["identifiability"] == "red"


def test_convergence_follows_r_hat_and_ess(direct):
    samples = uniform_samples(kept=400, chains=5)
    shifted = samples.copy()
    shifted[:, 0] += 0.15 * np.array([high - low for low, high in PRIOR.values()])  # une chaîne légèrement décalée
    assert ask(mcmc(direct, samples))["convergence"] == "green"
    assert ask(mcmc(direct, shifted))["convergence"] in ("orange", "red")
    drifting = uniform_samples(kept=400, chains=5) + np.linspace(0, 1, 400)[:, None, None, None] * np.array([high - low for low, high in PRIOR.values()])
    assert ask(mcmc(direct, drifting))["convergence"] == "red"
    few = uniform_samples(kept=30, chains=5)  # ESS d'environ 150 : orange
    assert ask(mcmc(direct, few))["convergence"] == "orange"


def test_too_few_iterations_cannot_be_judged_and_are_red(direct):
    assert ask(mcmc(direct, uniform_samples(kept=3, chains=5)))["convergence"] == "red"


# Verdict, seuils ----------------------------------------------------------------------------------


def test_the_verdict_is_the_worst_criterion(direct):
    for offset, verdict in ((0.0, "green"), (0.2, "orange"), (0.4, "red")):
        assert reliability(with_residuals(direct, offset), environment=NO_ENVIRONMENT).data["verdict"].value == verdict
    both = dataclasses.replace(with_residuals(direct, 0.2), warnings=[provenance_problem()])  # orange + orange
    assert reliability(both, environment=NO_ENVIRONMENT).data["verdict"].value == "orange"
    mixed = drop_rows(with_residuals(direct, 0.2), 0.5)  # orange + rouge
    assert reliability(mixed, environment=NO_ENVIRONMENT).data["verdict"].value == "red"


def test_each_reason_is_a_warning_with_its_parameters(direct):
    answer = reliability(with_residuals(direct, 0.4), environment=NO_ENVIRONMENT)
    reasons = {w.code: w for w in answer.warnings if w.code.startswith("W_RELIABILITY_")}
    assert set(reasons) == {"W_RELIABILITY_FIT"}  # seuls les critères non verts
    assert reasons["W_RELIABILITY_FIT"].params == {"ratio": pytest.approx(4.0), "green": 1.5, "orange": 3.0}
    texts = answer.data["reasons"].value
    assert len(texts) == 3 and texts[0].startswith("Fit: largest RMSE/σT 4")  # une raison par critère, en anglais


def test_thresholds_are_reported_and_not_validated(direct):
    answer = reliability(direct, environment=NO_ENVIRONMENT)
    assert answer.data["thresholds_validated"].value is False
    used = dict(zip(answer.data["threshold_names"].value, answer.data["thresholds"].value, strict=True))
    assert used == ReliabilityThresholds().as_dict() and used["rhat_green"] == 1.05
    assert answer.provenance["parameters"]["thresholds"] == used


def test_thresholds_can_be_overridden_by_parameter(direct):
    results = with_residuals(direct, 0.2)  # 2σ : orange par défaut
    assert ask(results)["fit"] == "orange"
    assert ask(results, thresholds={"fit_green": 2.5, "fit_orange": 4.0})["fit"] == "green"
    answer = reliability(results, thresholds={"fit_orange": 1.8}, environment=NO_ENVIRONMENT)
    assert dict(zip(answer.data["levels"].value, answer.data["criteria"].value, strict=True))["red"] == "fit"
    assert answer.data["thresholds"].value[list(answer.data["threshold_names"].value).index("fit_orange")] == 1.8


def test_thresholds_can_be_overridden_by_the_environment_and_the_parameter_wins(direct, tmp_path):
    path = tmp_path / "thresholds.yaml"
    path.write_text("fit_green: 2.5\nfit_orange: 4.0\n", encoding="utf-8")
    results = with_residuals(direct, 0.2)
    environment = {"MOLONARI_THRESHOLDS": str(path)}
    assert dict(zip(*[reliability(results, environment=environment).data[k].value for k in ("criteria", "levels")], strict=True))["fit"] == "green"
    answer = reliability(results, thresholds={"fit_green": 1.0, "fit_orange": 1.5}, environment=environment)
    assert dict(zip(answer.data["criteria"].value, answer.data["levels"].value, strict=True))["fit"] == "red"


def test_the_environment_variable_is_read_from_the_process_by_default(direct, tmp_path, monkeypatch):
    path = tmp_path / "t.yaml"
    path.write_text("fit_green: 2.5\nfit_orange: 4.0\n", encoding="utf-8")
    monkeypatch.setenv("MOLONARI_THRESHOLDS", str(path))
    assert dict(zip(*[reliability(with_residuals(direct, 0.2)).data[k].value for k in ("criteria", "levels")], strict=True))["fit"] == "green"


@pytest.mark.parametrize("override", [
    {"bogus": 1.0},
    {"fit_green": "high"},
    {"fit_green": True},
    {"fit_green": -1.0},
    {"fit_green": float("nan")},
    {"fit_green": 5.0},          # le vert moins exigeant que l'orange (3)
    {"ess_green": 50.0},         # pour l'ESS, le vert doit être au moins l'orange (100)
])
def test_invalid_thresholds_are_refused(direct, override):
    assert code_of(lambda: reliability(direct, thresholds=override, environment=NO_ENVIRONMENT)) == "E_THRESHOLDS_INVALID"


def test_an_invalid_threshold_file_is_refused(tmp_path):
    for text in ("fit_green: [1]\n", "unknown: 1\n", "- a\n- b\n", "key: [unclosed\n"):
        path = tmp_path / "bad.yaml"
        path.write_text(text, encoding="utf-8")
        assert code_of(lambda environment={"MOLONARI_THRESHOLDS": str(path)}: resolve_thresholds(None, environment)) == "E_THRESHOLDS_INVALID"
    assert code_of(lambda: resolve_thresholds(None, {"MOLONARI_THRESHOLDS": str(tmp_path / "missing.yaml")})) == "E_THRESHOLDS_INVALID"
    empty = tmp_path / "empty.yaml"
    empty.write_text("", encoding="utf-8")
    assert resolve_thresholds(None, {"MOLONARI_THRESHOLDS": str(empty)}) == ReliabilityThresholds()


def test_the_answer_is_an_answer_with_the_requested_level(direct):
    answer = reliability(direct, level=0.8, environment=NO_ENVIRONMENT)
    assert isinstance(answer, Answer) and answer.data["level"].value == 0.8
