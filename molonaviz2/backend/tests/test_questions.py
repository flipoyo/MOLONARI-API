"""Questions : réponses typées calculées à partir d'un résultat, sans Qt."""

import dataclasses
from pathlib import Path

import numpy as np
import pytest
from conftest import synthetic_inputs
from molonari_twin.domain.compute import ForwardConfig, InversionConfig, LayerSpec
from molonari_twin.engine.api import run_forward, run_inversion
from molonari_twin.errors import UserError
from molonari_twin.questions import (
    Answer,
    Quantity,
    chain_data,
    convergence,
    coverage,
    describe,
    energy_breakdown,
    exchange_direction,
    exchanged_volume,
    fit_quality,
    heat_budget,
    observed_at,
    posterior_samples,
    predictive_band,
    provenance,
    residuals,
    sensor_intervals,
    sensor_report,
    sensor_samples,
    soil_properties,
    temperature_at,
    twin_exchanged_volume,
    water_flux,
)
from molonari_twin.twin.results import load_results, write_results


def test_answers_and_quantities_are_frozen():
    quantity = Quantity(1.0, "m")
    answer = Answer({"x": quantity}, {"question": "test", "parameters": {}})
    with pytest.raises(dataclasses.FrozenInstanceError):
        quantity.value = 2.0
    with pytest.raises(dataclasses.FrozenInstanceError):
        answer.data = {}
    assert quantity.interval is None and answer.warnings == []


def test_sensor_intervals_at_90_percent_is_plus_or_minus_1_645_sigma():
    observed = np.array([[10.0, 11.0, 12.0], [13.0, 14.0, 15.0]])
    answer = sensor_intervals(observed, 0.3, 0.90)
    temperature = answer.data["temperature"]
    low, high = temperature.interval
    assert answer.data["z"].value == pytest.approx(1.645, abs=1e-3)
    np.testing.assert_allclose(high - observed, 1.6448536 * 0.3, rtol=1e-6)
    np.testing.assert_allclose(observed - low, 1.6448536 * 0.3, rtol=1e-6)
    assert temperature.unit == "°C" and answer.provenance == {
        "question": "sensor_intervals", "parameters": {"sigma": 0.3, "level": 0.9}}


# Volumes ------------------------------------------------------------------------------------------


def test_constant_flux_gives_flux_times_duration():
    times = np.datetime64("2024-01-01T00:00") + np.arange(0, 3601, 900).astype("timedelta64[s]")
    answer = exchanged_volume(times, np.full(len(times), 2e-6))
    assert answer.data["net"].value == pytest.approx(2e-6 * 3600, rel=1e-12)
    assert answer.data["infiltration"].value == pytest.approx(2e-6 * 3600, rel=1e-12)
    assert answer.data["exfiltration"].value == 0.0 and answer.data["net"].unit == "m³/m²"


def test_sinusoidal_flux_over_one_period():
    period, amplitude = 86_400.0, 1e-6
    times = np.linspace(0.0, period, 200_001)
    answer = exchanged_volume(times, amplitude * np.sin(2 * np.pi * times / period))
    expected = amplitude * period / np.pi
    assert answer.data["infiltration"].value == pytest.approx(expected, rel=1e-6)
    assert answer.data["exfiltration"].value == pytest.approx(-expected, rel=1e-6)
    assert abs(answer.data["net"].value) < 1e-6 * expected


def test_sign_change_is_cut_at_the_zero_crossing():
    answer = exchanged_volume([0.0, 1.0], [1.0, -1.0])
    assert (answer.data["infiltration"].value, answer.data["exfiltration"].value, answer.data["net"].value) == (0.25, -0.25, 0.0)


def test_irregular_steps_on_a_ramp_crossing_zero():
    # q(t) = t − 3 sur [0, 10] : exfiltration −4,5 sur [0, 3], infiltration 24,5 sur [3, 10] ; linéaire, donc exact.
    times = np.array([0.0, 0.5, 2.0, 2.2, 4.7, 5.0, 9.0, 10.0])
    answer = exchanged_volume(times, times - 3.0)
    assert answer.data["exfiltration"].value == pytest.approx(-4.5, rel=1e-12)
    assert answer.data["infiltration"].value == pytest.approx(24.5, rel=1e-12)
    assert answer.data["net"].value == pytest.approx(20.0, rel=1e-12)


def test_area_gives_cubic_metres():
    answer = exchanged_volume([0.0, 10.0], [1e-5, 1e-5], area=2.5)
    assert answer.data["net"].value == pytest.approx(2.5e-4) and answer.data["net"].unit == "m³"
    assert answer.provenance == {"question": "exchanged_volume", "parameters": {"area": 2.5}}


def test_volume_of_each_depth():
    times = np.array([0.0, 1.0, 2.0])
    q = np.array([[1.0, 1.0, 1.0], [-1.0, -1.0, -1.0], [1.0, -1.0, 1.0]])
    answer = exchanged_volume(times, q)
    np.testing.assert_allclose(answer.data["infiltration"].value, [2.0, 0.0, 0.5])
    np.testing.assert_allclose(answer.data["exfiltration"].value, [0.0, -2.0, -0.5])


@pytest.mark.parametrize("times", [[0.0, 1.0, 1.0], [0.0, 2.0, 1.0]])
def test_times_must_increase(times):
    with pytest.raises(UserError) as error:
        exchanged_volume(times, [1.0, 1.0, 1.0])
    assert error.value.code == "E_QUESTION_TIMES"


def test_flux_and_times_must_match():
    with pytest.raises(UserError) as error:
        exchanged_volume([0.0, 1.0, 2.0], [1.0, 1.0])
    assert error.value.code == "E_QUESTION_SHAPES"


# Bande prédictive et couverture -------------------------------------------------------------------


def small_config(**overrides):
    values = {
        "layers": [LayerSpec("a", 0.4)], "nb_cells": 20, "nb_chains": 7, "nb_iter": 10, "burn_in_max": 20, "n_field_samples": 14, "seed": 1
    }
    values.update(overrides)
    return InversionConfig(**values)


@pytest.fixture(scope="module")
def inversions():
    """Inversions courtes, une par réglage de store_fields (champs sous-échantillonnés en espace et en temps)."""
    inputs = synthetic_inputs(nb_times=60)
    return {kind: run_inversion(inputs, small_config(store_fields=kind, space_step=2, time_step=3, thin=2)) for kind in ("sensors", "full", "none")}


def twin_results(folder, inversion):
    """Résultats d'un twin écrits puis relus, comme les lit la fenêtre d'un twin."""
    metadata = {"results_format": "2.0.0", "sensor_cell_depths": [float(d) for d in inversion.best.sensor_depths]}
    write_results(folder, inversion.best, inversion, ["a"], metadata)
    return load_results(folder)


def test_same_seed_gives_the_same_band():
    samples = np.random.default_rng(0).normal(15.0, 0.5, (200, 3, 40))
    first, second = (predictive_band(samples, 0.3, 0.9, seed=7) for _ in range(2))
    for a, b in zip(first.data["temperature"].interval, second.data["temperature"].interval):
        np.testing.assert_array_equal(a, b)
    other = predictive_band(samples, 0.3, 0.9, seed=8).data["temperature"].interval[0]
    assert not np.array_equal(first.data["temperature"].interval[0], other)
    assert first.provenance["parameters"] == {"sigma": 0.3, "level": 0.9, "seed": 7, "samples": 200}


def test_without_noise_the_band_is_the_quantiles_of_the_samples():
    samples = np.random.default_rng(1).normal(15.0, 0.5, (101, 3, 20))
    temperature = predictive_band(samples, 0.0, 0.9, seed=0).data["temperature"]
    np.testing.assert_allclose(temperature.interval[0], np.quantile(samples, 0.05, axis=0))
    np.testing.assert_allclose(temperature.interval[1], np.quantile(samples, 0.95, axis=0))
    np.testing.assert_allclose(temperature.value, np.median(samples, axis=0))
    assert temperature.unit == "°C"


def test_noise_widens_the_band():
    samples = np.full((100_000, 3, 1), 10.0)  # erreur des quantiles à 5 % : environ 1,3 %
    low, high = predictive_band(samples, 0.3, 0.9, seed=0).data["temperature"].interval
    np.testing.assert_allclose(high - 10.0, 1.645 * 0.3, rtol=0.05)
    np.testing.assert_allclose(10.0 - low, 1.645 * 0.3, rtol=0.05)


@pytest.mark.parametrize("sigma, level, code", [(0.3, 0.0, "E_QUESTION_LEVEL"), (0.3, 1.0, "E_QUESTION_LEVEL"), (-0.1, 0.9, "E_QUESTION_SIGMA")])
def test_band_settings_are_checked(sigma, level, code):
    with pytest.raises(UserError) as error:
        predictive_band(np.zeros((10, 3, 4)), sigma, level, seed=0)
    assert error.value.code == code


def test_coverage_ignores_nan_and_counts_points():
    band = Quantity(np.zeros((2, 4)), "°C", (np.zeros((2, 4)), np.ones((2, 4))))
    observed = np.array([[0.5, 1.0, 2.0, np.nan], [0.0, -1.0, np.nan, np.nan]])
    answer = coverage(observed, band)
    assert answer.data["points"].value == 5 and answer.data["coverage"].value == pytest.approx(3 / 5)
    np.testing.assert_allclose(answer.data["per_sensor"].value, [2 / 3, 1 / 2])
    np.testing.assert_array_equal(answer.data["points_per_sensor"].value, [3, 2])
    assert answer.warnings == []


def test_coverage_without_points_warns():
    band = Quantity(np.zeros(3), "°C", (np.zeros(3), np.ones(3)))
    answer = coverage(np.full(3, np.nan), band)
    assert np.isnan(answer.data["coverage"].value) and answer.data["points"].value == 0
    assert [w.code for w in answer.warnings] == ["W_COVERAGE_NO_POINTS"]


def test_coverage_needs_matching_shapes():
    band = Quantity(np.zeros((3, 4)), "°C", (np.zeros((3, 4)), np.ones((3, 4))))
    with pytest.raises(UserError) as error:
        coverage(np.zeros((4, 3)), band)
    assert error.value.code == "E_QUESTION_SHAPES"


def test_coverage_of_the_measurement_interval():
    observed = np.array([[10.0, 11.0], [12.0, 13.0]])
    assert coverage(observed, sensor_intervals(observed, 0.1, 0.9).data["temperature"]).data["coverage"].value == 1.0


def test_observed_at_subsampled_dates():
    times = np.datetime64("2024-01-01T00:00", "s") + np.arange(10) * np.timedelta64(15, "m")
    observed = np.arange(30.0).reshape(10, 3)
    at = np.append(times[::3], np.datetime64("2030-01-01T00:00", "s"))
    values = observed_at(times, observed, at).data["temperature"].value
    assert values.shape == (3, 5)
    np.testing.assert_array_equal(values[:, :4], observed[::3].T)
    assert np.isnan(values[:, 4]).all()


def test_sensor_samples_stored_at_the_sensors(inversions):
    inversion = inversions["sensors"]
    answer = sensor_samples(inversion)
    np.testing.assert_array_equal(answer.data["temperature"].value, inversion.sensor_temperature)
    np.testing.assert_array_equal(answer.data["time"].value, inversion.field_times.astype("datetime64[s]"))
    np.testing.assert_array_equal(answer.data["depth"].value, inversion.best.sensor_depths)


def test_sensor_samples_from_full_fields_take_the_nearest_rows(inversions):
    inversion = inversions["full"]
    rows = [int(np.argmin(np.abs(inversion.field_depths - d))) for d in inversion.best.sensor_depths]
    values = sensor_samples(inversion).data["temperature"].value
    assert values.shape == (inversion.field_temperature.shape[0], 3, len(inversion.field_times))
    np.testing.assert_array_equal(values, inversion.field_temperature[:, rows, :].astype(float))


@pytest.mark.parametrize("kind", ["sensors", "full"])
def test_sensor_samples_of_twin_results_match_the_inversion(tmp_path, inversions, kind):
    expected = sensor_samples(inversions[kind]).data
    found = sensor_samples(twin_results(tmp_path, inversions[kind])).data
    np.testing.assert_array_equal(found["temperature"].value, expected["temperature"].value)
    np.testing.assert_array_equal(found["time"].value, expected["time"].value)


def test_sensor_samples_without_stored_fields(tmp_path, inversions):
    for result in (inversions["none"], twin_results(tmp_path, inversions["none"])):
        with pytest.raises(UserError) as error:
            sensor_samples(result)
        assert error.value.code == "E_QUESTION_NO_FIELDS"


@pytest.mark.slow
def test_predictive_coverage_of_a_synthetic_column_is_close_to_its_level():
    """Mesures = modèle direct + bruit i.i.d. σ = 0,3 ; la bande prédictive à 90 % doit couvrir 87 à 93 % des mesures.

    3 capteurs × 599 dates ≈ 1 800 points : l'écart-type d'échantillonnage de la couverture vaut environ
    0,7 %, donc ± 3 % fait plus de 4σ. Le bruit n'a pas d'autocorrélation par construction ; la
    dépendance qui reste vient des paramètres a posteriori, partagés par tous les points.

    Les paramètres sont corrélés le long d'une crête (k, n, λs, ρs·cs) : les chaînes, toutes dans le
    même mode, la parcourent lentement et R-hat reste au-dessus de 1,2. La bande prédictive, dominée
    par le bruit, n'en dépend presque pas (89,4 % avec 600 itérations, 89,3 % avec 4 000)."""
    sigma, layers = 0.3, [LayerSpec("a", 0.4, neg_log10_k=11.5, porosity=0.2, lambda_s=2.5, rhos_cs=4e6)]
    inputs = synthetic_inputs(nb_times=600, sigma_t=sigma)
    truth = run_forward(inputs, ForwardConfig(layers, nb_cells=40)).simulated_at_sensors()
    measured = truth + np.random.default_rng(42).normal(0.0, sigma, truth.shape)
    measured[0] = truth[0]  # même condition initiale (profil interpolé à t0) que la vérité
    noisy = dataclasses.replace(inputs, t_sensors=measured)
    config = InversionConfig(layers=[LayerSpec("a", 0.4)], nb_cells=40, sigma_t=sigma, nb_chains=8, nb_iter=4000,
                             burn_in_max=1000, fields_per_chain=50, store_fields="sensors", seed=1)
    inversion = run_inversion(noisy, config)
    energies = inversion.energies.mean(axis=0)
    assert energies.max() - energies.min() < 10, energies  # toutes les chaînes dans le même mode

    samples = sensor_samples(inversion).data
    band = predictive_band(samples["temperature"].value, sigma, 0.90, seed=3).data["temperature"]
    observed = observed_at(noisy.times, measured, samples["time"].value).data["temperature"].value
    answer = coverage(observed[:, 1:], Quantity(band.value[:, 1:], "°C", (band.interval[0][:, 1:], band.interval[1][:, 1:])))
    assert answer.data["points"].value == 3 * 599
    assert 0.87 <= answer.data["coverage"].value <= 0.93, answer.data


# Données par chaîne -------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def unthinned():
    """7 chaînes, nb_iter = 10, thin du calcul = 1."""
    return run_inversion(synthetic_inputs(nb_times=60), small_config(thin=1))


def test_bn36_when_nb_iter_is_not_a_multiple_of_the_display_thin(unthinned):
    answer = chain_data(unthinned, thin=3)
    assert answer.data["count"].value == 7 * (10 // 3) == 21
    np.testing.assert_array_equal(answer.data["iteration"].value, [3, 6, 9])
    np.testing.assert_array_equal(answer.data["samples.porosity"].value, unthinned.samples[2::3, :, :, 1])
    np.testing.assert_array_equal(answer.data["energy"].value, unthinned.energies[2::3])
    assert answer.data["samples.lambda_s"].unit == "W·m⁻¹·K⁻¹" and answer.data["samples.lambda_s"].value.shape == (3, 7, 1)


def test_display_thin_follows_the_thin_of_the_computation(inversions):
    inversion = inversions["sensors"]  # nb_iter = 10, thin du calcul = 2 : itérations 2, 4, 6, 8, 10
    assert chain_data(inversion).data["count"].value == 7 * (10 // 2)
    answer = chain_data(inversion, thin=2)
    assert answer.data["count"].value == 7 * (10 // (2 * 2)) == 14
    np.testing.assert_array_equal(answer.data["iteration"].value, [4, 8])


def test_chain_selection(unthinned):
    answer = chain_data(unthinned, chains=[2, 0])
    np.testing.assert_array_equal(answer.data["chain"].value, [2, 0])
    np.testing.assert_array_equal(answer.data["samples.neg_log10_k"].value, unthinned.samples[:, [2, 0]][..., 0])
    np.testing.assert_array_equal(answer.data["acceptance"].value, unthinned.acceptance[[2, 0]])
    np.testing.assert_array_equal(answer.data["resets"].value, unthinned.resets[[2, 0]])
    assert answer.data["count"].value == 2 * 10
    assert answer.provenance["parameters"] == {"chains": [2, 0], "thin": 1, "iterations": None}


def test_iteration_range_is_inclusive(unthinned):
    answer = chain_data(unthinned, iterations=(4, 7))
    np.testing.assert_array_equal(answer.data["iteration"].value, [4, 5, 6, 7])
    np.testing.assert_array_equal(answer.data["energy"].value, unthinned.energies[3:7])
    np.testing.assert_array_equal(chain_data(unthinned, thin=2, iterations=(5, None)).data["iteration"].value, [6, 8, 10])


def test_chain_data_of_twin_results_match_the_inversion(tmp_path, inversions):
    inversion = inversions["full"]
    expected = chain_data(inversion, chains=[1, 3], thin=2).data
    found = chain_data(twin_results(tmp_path, inversion), chains=[1, 3], thin=2).data
    assert found.keys() == expected.keys()
    for name in expected:
        np.testing.assert_array_equal(found[name].value, expected[name].value)
        assert found[name].unit == expected[name].unit


@pytest.mark.parametrize("kwargs, code", [
    ({"chains": [7]}, "E_QUESTION_CHAINS"),
    ({"chains": [-1]}, "E_QUESTION_CHAINS"),
    ({"thin": 0}, "E_QUESTION_THIN"),
    ({"thin": 1.5}, "E_QUESTION_THIN"),
])
def test_chain_data_settings_are_checked(unthinned, kwargs, code):
    with pytest.raises(UserError) as error:
        chain_data(unthinned, **kwargs)
    assert error.value.code == code


def test_results_without_posterior(tmp_path, inversions):
    results = twin_results(tmp_path, inversions["none"])
    results = dataclasses.replace(results, posterior=None)  # mode direct, ou résultat 1.x
    with pytest.raises(UserError) as error:
        chain_data(results)
    assert error.value.code == "E_QUESTION_NO_POSTERIOR"


# Questions sur les résultats d'un twin (lot C2) ---------------------------------------------------

EXAMPLES = Path(__file__).resolve().parents[2] / "examples" / "twins"


def results_of(folder, forward, inversion=None, mode="direct", sigma=0.1):
    """Résultats écrits puis relus, avec les métadonnées que lisent les questions (mode, σ de l'instrument)."""
    metadata = {"results_format": "2.0.0", "mode": mode, "name": "twin_x", "twin_id": "abc", "code_version": "v",
                "created_at": "2026-10-23T10:00:00", "device": "cpu", "duration_s": 1.5,
                "data": {"file": "m.csv", "sha256": "f00"}, "provenance": {"twin": {"commit": "1234"}},
                "sensor_cell_depths": [float(d) for d in forward.sensor_depths]}
    if sigma is not None:
        metadata["instruments"] = {"sigma_temperature_C": sigma, "sigma_pressure_m": 0.01}
    write_results(folder, forward, inversion, ["a"], metadata)
    return load_results(folder)


@pytest.fixture(scope="module")
def direct(tmp_path_factory):
    forward = run_forward(synthetic_inputs(nb_times=80), ForwardConfig([LayerSpec("a", 0.4)], nb_cells=20))
    return results_of(tmp_path_factory.mktemp("direct"), forward)


def code_of(call):
    with pytest.raises(UserError) as error:
        call()
    return error.value.code


def test_water_flux_statistics_over_the_period(direct):
    q = direct.flux["q_riverbed_m_per_s"].to_numpy(float)
    data = water_flux(direct).data
    assert data["mean"].value == pytest.approx(q.mean())
    assert data["median"].value == pytest.approx(np.median(q))
    assert (data["p5"].value, data["p95"].value) == pytest.approx(tuple(np.percentile(q, [5, 95])))
    assert data["points"].value == q.size and data["mean"].unit == "m/s"
    times = direct.flux["date"].to_numpy().astype("datetime64[s]")
    part = water_flux(direct, start=times[10], end=times[19]).data
    assert part["points"].value == 10 and part["mean"].value == pytest.approx(q[10:20].mean())
    assert (part["start"].value, part["end"].value) == (times[10], times[19])
    assert code_of(lambda: water_flux(direct, start=times[-1] + np.timedelta64(1, "D"))) == "E_QUESTION_PERIOD_EMPTY"


def test_volume_of_a_constant_flux_through_the_bed(direct):
    results = dataclasses.replace(direct, flux=direct.flux.assign(q_riverbed_m_per_s=2e-6))
    times = results.flux["date"].to_numpy().astype("datetime64[s]")
    duration = (times[-1] - times[0]) / np.timedelta64(1, "s")
    data = twin_exchanged_volume(results).data
    assert data["infiltration"].value == pytest.approx(2e-6 * duration)
    assert data["exfiltration"].value == 0.0
    assert twin_exchanged_volume(results, area=3.0).data["net"].value == pytest.approx(6e-6 * duration)


def test_temperature_value_profile_and_series(direct):
    depths, times, field = direct.depths, direct.times.astype("datetime64[s]"), direct.temperature
    value = temperature_at(direct, z=float(depths[3]), t=times[5]).data["temperature"].value
    assert value == pytest.approx(field[3, 5])
    profile = temperature_at(direct, t=times[5]).data
    assert profile["temperature"].value.shape == depths.shape and profile["depth"].value.shape == depths.shape
    np.testing.assert_allclose(profile["temperature"].value, field[:, 5])
    series = temperature_at(direct, z=float(depths[3])).data
    np.testing.assert_allclose(series["temperature"].value, field[3])
    middle = (float(depths[3]) + float(depths[4])) / 2
    np.testing.assert_allclose(temperature_at(direct, z=middle).data["temperature"].value, (field[3] + field[4]) / 2)


def test_temperature_needs_a_point_inside_the_results(direct):
    assert code_of(lambda: temperature_at(direct)) == "E_QUESTION_Z_OR_T"
    assert code_of(lambda: temperature_at(direct, z=99.0)) == "E_QUESTION_OUT_OF_RANGE"
    assert code_of(lambda: temperature_at(direct, t=np.datetime64("1990-01-01"))) == "E_QUESTION_OUT_OF_RANGE"


def test_sensor_report_in_direct_mode_is_the_measurement_interval(direct):
    answer = sensor_report(direct, level=0.9)
    data = answer.data
    low, high = data["measured"].interval
    np.testing.assert_allclose(high - data["measured"].value, 1.645 * 0.1, rtol=1e-3)
    np.testing.assert_allclose(data["measured"].value - low, 1.645 * 0.1, rtol=1e-3)
    assert data["measured"].value.shape == (3, len(data["time"].value)) == data["simulated"].value.shape
    assert data["sigma"].value == 0.1
    assert "credible" not in data and "predictive" not in data and answer.warnings == []
    assert sensor_report(direct, sigma=0.3).data["sigma"].value == 0.3


def test_sensor_report_needs_sigma_when_results_do_not_record_it(tmp_path, direct):
    forward = run_forward(synthetic_inputs(nb_times=40), ForwardConfig([LayerSpec("a", 0.4)], nb_cells=20))
    results = results_of(tmp_path, forward, sigma=None)
    assert code_of(lambda: sensor_report(results)) == "E_QUESTION_SIGMA_REQUIRED"
    assert sensor_report(results, sigma=0.2).data["sigma"].value == 0.2


def test_sensor_report_in_mcmc_mode_has_both_bands_and_the_coverage(tmp_path, inversions):
    inversion = inversions["sensors"]
    results = results_of(tmp_path, inversion.best, inversion, mode="mcmc", sigma=0.1)
    data = sensor_report(results, level=0.9).data
    credible, predictive = data["credible"], data["predictive"]
    assert credible.value.shape == predictive.value.shape == (3, len(data["field_time"].value))
    width = lambda band: float(np.mean(band.interval[1] - band.interval[0]))
    assert width(predictive) > width(credible)  # le bruit de mesure élargit la bande, en moyenne
    # Même calcul que l'application (molonaviz2.ui.predictive) : même graine, même bande, même couverture.
    samples = sensor_samples(results).data
    band = predictive_band(samples["temperature"].value, 0.1, 0.9, seed=0).data["temperature"]
    times, measured = results.residuals["date"].to_numpy(), np.stack([results.residuals[f"t{i}_measured"].to_numpy(float) for i in (1, 2, 3)], 1)
    expected = coverage(observed_at(times, measured, samples["time"].value).data["temperature"].value, band).data
    np.testing.assert_array_equal(predictive.interval[0], band.interval[0])
    assert data["coverage"].value == expected["coverage"].value and data["points"].value == expected["points"].value


def test_sensor_report_without_stored_fields_warns(tmp_path, inversions):
    inversion = inversions["none"]
    answer = sensor_report(results_of(tmp_path, inversion.best, inversion, mode="mcmc"))
    assert "predictive" not in answer.data
    assert [w.code for w in answer.warnings] == ["E_QUESTION_NO_FIELDS"]


def test_residuals_and_their_intervals(tmp_path, direct, inversions):
    data = residuals(direct).data
    np.testing.assert_allclose(data["residual"].value, data["simulated"].value - data["measured"].value)
    np.testing.assert_allclose(data["rmse"].value, np.sqrt(np.mean(data["residual"].value ** 2, axis=1)))
    assert "predictive" not in data
    inversion = inversions["sensors"]
    with_bands = residuals(results_of(tmp_path, inversion.best, inversion, mode="mcmc"), intervals=True).data
    assert {"credible", "predictive", "coverage"} <= set(with_bands)


@pytest.mark.parametrize("name", ["twin_01", "twin_03"])
def test_describe_reads_the_configuration(name):
    from molonari_twin.twin.config import load_twin

    config = load_twin(EXAMPLES / name)
    data = describe(config).data
    assert data["name"].value == name and data["mode"].value == config.mode
    np.testing.assert_allclose(data["sensor_depths"].value, config.depths)
    assert list(data["layers"].value) == [layer.name for layer in config.layers]
    assert ("nb_chains" in data) == (config.mode == "mcmc")


def test_provenance_comes_from_the_metadata(direct):
    answer = provenance(direct)
    assert answer.data["twin_id"].value == "abc" and answer.data["data_sha256"].value == "f00"
    assert answer.data["duration"].value == 1.5 and answer.data["duration"].unit == "s"
    assert answer.data["reproduce"].value == "molonari-twin run twin_x"
    assert answer.provenance["repositories"] == {"twin": {"commit": "1234"}}


# Sens des échanges, bilan de chaleur et qualité de l'ajustement (lot C3) --------------------------


def with_flux(results, q):
    return dataclasses.replace(results, flux=results.flux.assign(q_riverbed_m_per_s=q))


def test_infiltration_everywhere_has_no_reversal(direct):
    data = exchange_direction(with_flux(direct, 1e-6)).data
    assert data["infiltration_share"].value == 1.0 and data["exfiltration_share"].value == 0.0
    assert data["reversals"].value == 0 and data["dominant"].value == "infiltration"
    data = exchange_direction(with_flux(direct, -1e-6)).data
    assert (data["infiltration_share"].value, data["dominant"].value) == (0.0, "exfiltration")


@pytest.mark.parametrize("periods", [1, 2, 3])
def test_a_sinusoid_over_k_periods_reverses_about_twice_per_period(direct, periods):
    n = len(direct.flux)
    q = np.sin(2 * np.pi * periods * (np.arange(n) + 0.5) / n)
    data = exchange_direction(with_flux(direct, q)).data
    assert data["reversals"].value in (2 * periods - 1, 2 * periods)
    assert data["infiltration_share"].value == pytest.approx(0.5, abs=0.02)
    assert data["infiltration_share"].value + data["exfiltration_share"].value == pytest.approx(1.0)


def test_shares_are_weighted_by_the_time_intervals(direct):
    n = len(direct.flux)
    dates = (np.datetime64("2016-06-27T00:00:00") + np.array([0, 1, 2, 3, 100]).astype("timedelta64[h]"))
    flux = direct.flux.iloc[:5].assign(date=dates, q_riverbed_m_per_s=[1.0, 1.0, 1.0, -1.0, -1.0])
    share = exchange_direction(dataclasses.replace(direct, flux=flux)).data["infiltration_share"].value
    assert 0.0 < share < 0.1 and n > 5  # le long dernier intervalle compte pour presque tout


def test_exchange_direction_in_direct_mode_has_no_probability(direct):
    assert "p_infiltration" not in exchange_direction(direct).data


def test_exchange_direction_period_is_checked(direct):
    assert code_of(lambda: exchange_direction(direct, start=np.datetime64("2030-01-01"))) == "E_QUESTION_PERIOD_EMPTY"


def constant_heat(results, advective, conductive):
    shape = results.temperature.shape
    return dataclasses.replace(results, heat_flux_advective=np.full(shape, advective), heat_flux_conductive=np.full(shape, conductive))


def test_constant_heat_fluxes_have_exact_means_and_shares(direct):
    answer = heat_budget(constant_heat(direct, -6.0, 2.0), z=0.2)
    data = answer.data
    assert (data["advective"].value, data["conductive"].value, data["total"].value) == (-6.0, 2.0, -4.0)
    assert data["advective_share"].value == pytest.approx(0.75) and data["conductive_share"].value == pytest.approx(0.25)
    assert data["dominant"].value == "advection" and data["advective"].unit == "W/m²"
    assert data["lateral"].value == 0.0 and data["lateral_share"].value == 0.0
    assert "W_LATERAL_NOT_MODELLED" in [w.code for w in answer.warnings]
    assert heat_budget(constant_heat(direct, 1.0, -3.0)).data["dominant"].value == "conduction"


def test_heat_budget_depth_and_period(direct):
    results = dataclasses.replace(direct, heat_flux_advective=np.tile(direct.depths[:, None], (1, direct.temperature.shape[1])),
                                  heat_flux_conductive=np.zeros_like(direct.temperature))
    middle = float(direct.depths[2] + direct.depths[3]) / 2
    assert heat_budget(results, z=middle).data["advective"].value == pytest.approx(middle)
    assert heat_budget(results, z=0.0).data["advective"].value == pytest.approx(float(direct.depths[0]))  # première cellule
    assert code_of(lambda: heat_budget(results, z=99.0)) == "E_QUESTION_OUT_OF_RANGE"
    times = direct.times.astype("datetime64[s]")
    assert heat_budget(results, start=times[10], end=times[19]).data["points"].value == 10
    assert code_of(lambda: heat_budget(results, start=times[-1] + np.timedelta64(1, "D"))) == "E_QUESTION_PERIOD_EMPTY"


def test_heat_budget_needs_format_2(direct):
    old = dataclasses.replace(direct, heat_flux_advective=None, heat_flux_conductive=None)
    assert code_of(lambda: heat_budget(old)) == "E_QUESTION_NO_FIELDS"


def with_residuals(results, residual):
    """Résultats dont le simulé vaut le mesuré + `residual` (°C), à chaque capteur."""
    frame = results.residuals.copy()
    for i in (1, 2, 3):
        frame[f"t{i}_simulated"] = frame[f"t{i}_measured"] + residual
        frame[f"t{i}_residual"] = residual
    return dataclasses.replace(results, residuals=frame)


def test_zero_residuals_have_zero_rmse_and_bias(direct):
    data = fit_quality(with_residuals(direct, 0.0)).data
    np.testing.assert_array_equal(data["rmse"].value, np.zeros(3))
    np.testing.assert_array_equal(data["bias"].value, np.zeros(3))
    np.testing.assert_array_equal(data["rmse_over_sigma"].value, np.zeros(3))


def test_a_constant_offset_gives_bias_and_rmse_equal_to_the_offset(direct):
    data = fit_quality(with_residuals(direct, 0.2)).data
    np.testing.assert_allclose(data["bias"].value, 0.2)
    np.testing.assert_allclose(data["rmse"].value, 0.2)
    np.testing.assert_allclose(data["rmse_over_sigma"].value, 2.0)  # σ = 0,1
    assert np.isnan(data["lag1_autocorrelation"].value).all()  # résidu constant : autocorrélation indéfinie
    assert fit_quality(with_residuals(direct, 0.2), sigma=0.4).data["rmse_over_sigma"].value[0] == pytest.approx(0.5)


def test_residual_autocorrelation_separates_noise_from_structure(direct):
    n = len(direct.residuals)
    noise = np.random.default_rng(0).normal(0, 0.1, n)
    assert abs(fit_quality(with_residuals(direct, noise)).data["lag1_autocorrelation"].value[0]) < 0.3
    wave = 0.1 * np.sin(np.arange(n) / 30)
    assert fit_quality(with_residuals(direct, wave)).data["lag1_autocorrelation"].value[0] > 0.9


def test_fit_quality_in_direct_mode_has_no_coverage(direct):
    assert "coverage_per_sensor" not in fit_quality(direct).data


# Propriétés du sol, échantillons, convergence et énergie (lot C3) ---------------------------------

PRIOR = {"neg_log10_k": (11.0, 15.0), "porosity": (0.01, 0.25), "lambda_s": (1.0, 5.0), "rhos_cs": (1e6, 1e7)}
PARAMETERS = tuple(PRIOR)


def with_posterior(results, samples, energy=None, acceptance=None, best=None, layers=("a",)):
    """Résultats mcmc synthétiques : `samples` (I, N, L, 4), priors uniformes de PRIOR, meilleurs paramètres `best`."""
    kept, chains = samples.shape[:2]
    best = best if best is not None else samples.reshape(-1, *samples.shape[2:])[0]
    posterior = {
        "samples": samples,
        "energy": np.zeros((kept, chains)) if energy is None else energy,
        "acceptance": np.full(chains, 0.2) if acceptance is None else acceptance,
        "resets": np.zeros(chains, np.int64),
        "iteration": np.arange(1, kept + 1),
        "layer": np.array(layers),
        "parameter": np.array(PARAMETERS),
    }
    mcmc = {"best_parameters": {layer: dict(zip(PARAMETERS, map(float, best[i]), strict=True)) for i, layer in enumerate(layers)},
            "burn_in_converged": True, "burn_in_iterations": 12}
    config = {"mcmc": {"priors": [{name: list(bounds) for name, bounds in PRIOR.items()} for _ in layers]}}
    metadata = {**direct_metadata(results), "mode": "mcmc", "mcmc": mcmc, "config": config}
    return dataclasses.replace(results, posterior=posterior, metadata=metadata)


def direct_metadata(results):
    return {key: value for key, value in results.metadata.items() if key not in ("mcmc", "config")}


def uniform_samples(kept=400, chains=4, layers=1, seed=0):
    rng = np.random.default_rng(seed)
    columns = [rng.uniform(low, high, (kept, chains, layers)) for low, high in PRIOR.values()]
    return np.stack(columns, axis=-1)


def test_direct_soil_properties_are_the_configuration_values(direct):
    layer = {"z_low": 0.4, "neg_log10_k": 12.0, "porosity": 0.15, "lambda_s": 3.4, "rhos_cs": 5e6}
    results = dataclasses.replace(direct, metadata={**direct.metadata, "config": {"layers": [layer]}})
    data = soil_properties(results, layer_names=["sable"]).data
    assert list(data["layers"].value) == ["sable"]
    assert data["k"].value[0] == pytest.approx(1e-12) and data["k"].unit == "m²"
    assert data["K"].value[0] == pytest.approx(1e-12 * 1000.0 * 9.81 / 1.0e-3) and data["K"].unit == "m/s"
    assert (data["n"].value[0], data["lambda_s"].value[0], data["rhos_cs"].value[0]) == (0.15, 3.4, 5e6)
    assert data["k"].interval is None and "k_median" not in data and "n_identifiability" not in data


def test_samples_equal_to_the_prior_are_not_identifiable(direct):
    samples = uniform_samples(kept=2000, chains=4)
    data = soil_properties(with_posterior(direct, samples), level=0.9).data
    for name in ("n", "lambda_s", "rhos_cs", "k", "K"):
        assert data[f"{name}_identifiability"].value[0] == pytest.approx(1.0, abs=0.05), name
    low, high = data["n"].interval
    assert (high[0] - low[0]) / (0.25 - 0.01) == pytest.approx(0.9, abs=0.03)  # le HDI à 90 % d'une uniforme en couvre 90 %
    assert data["n_median"].value[0] == pytest.approx(0.13, abs=0.01)


def test_a_narrow_posterior_is_identifiable_and_k_bounds_are_ordered(direct):
    rng = np.random.default_rng(3)
    samples = uniform_samples(kept=500)
    samples[..., 0] = rng.normal(13.0, 0.05, samples.shape[:3])  # −log₁₀ k = 13 ± 0,05
    data = soil_properties(with_posterior(direct, samples, best=np.array([[13.0, 0.1, 3.0, 4e6]]))).data
    assert data["k_identifiability"].value[0] < 0.05
    low, high = data["k"].interval
    assert low[0] < data["k"].value[0] < high[0] and data["k"].value[0] == pytest.approx(1e-13)  # la borne basse de k vient du haut de −log₁₀ k
    assert data["K"].value[0] == pytest.approx(1e-13 * 9.81e6)


def test_soil_level_is_checked(direct):
    assert code_of(lambda: soil_properties(with_posterior(direct, uniform_samples()), level=1.0)) == "E_QUESTION_LEVEL"


def test_posterior_samples_are_chain_first(direct):
    samples = uniform_samples(kept=30, chains=5, layers=2)
    results = with_posterior(direct, samples, best=samples[0, 0], layers=("a", "b"))
    data = posterior_samples(results).data
    assert data["samples"].value.shape == (5, 30, 2, 4) and data["space"].value == "prior"
    np.testing.assert_array_equal(data["samples"].value[3, 7], samples[7, 3])
    assert list(data["parameter"].value) == list(PARAMETERS) and list(data["layer"].value) == ["a", "b"]
    selected = posterior_samples(results, chains=[4, 1], iterations=(5, 20), burn=2, thin=3).data
    assert list(selected["chain"].value) == [4, 1]
    assert list(selected["iteration"].value) == [5, 8, 11, 14, 17, 20]  # burn 2, puis une sur 3 (la 3ᵉ, la 6ᵉ…), puis 5 ≤ i ≤ 20
    np.testing.assert_array_equal(selected["samples"].value[0, 0], samples[4, 4])
    assert code_of(lambda: posterior_samples(results, chains=[9])) == "E_QUESTION_CHAINS"
    assert code_of(lambda: posterior_samples(results, thin=0)) == "E_QUESTION_THIN"


def test_posterior_samples_need_a_posterior(direct):
    assert code_of(lambda: posterior_samples(direct)) == "E_QUESTION_NO_POSTERIOR"
    assert code_of(lambda: convergence(direct)) == "E_QUESTION_NO_POSTERIOR"


def test_convergence_of_independent_chains(direct):
    results = with_posterior(direct, uniform_samples(kept=400, chains=5, layers=2), layers=("a", "b"))
    data = convergence(results).data
    for name in ("rhat", "ess_bulk", "ess_tail", "mcse_mean"):
        assert data[name].value.shape == (2, 4), name
    assert (data["rhat"].value < 1.05).all() and (data["ess_bulk"].value > 1000).all()
    assert data["rhat_max"].value == pytest.approx(data["rhat_max_all"].value)
    assert data["chain_ess_bulk"].value.shape == (5, 2, 4) and data["rhat_without_chain"].value.shape == (5,)
    np.testing.assert_allclose(data["acceptance"].value, 0.2)
    assert data["burn_in_converged"].value is True and data["burn_in_iterations"].value == 12


def test_a_stray_chain_is_found_by_the_r_hat_without_it(direct):
    samples = uniform_samples(kept=300, chains=5)
    samples[:, 2] += 0.5 * (np.array(list(PRIOR.values()))[:, 1] - np.array(list(PRIOR.values()))[:, 0])  # chaîne 2 décalée
    energy = np.zeros((300, 5))
    energy[:, 2] = 7.0
    data = convergence(with_posterior(direct, samples, energy=energy)).data
    assert data["rhat_max"].value > 1.2
    without = data["rhat_without_chain"].value
    assert without[2] < 1.05 and (np.delete(without, 2) > 1.2).all()
    assert data["mean_energy"].value[2] == 7.0 and data["mean_energy"].value[0] == 0.0


def test_convergence_follows_the_selection(direct):
    samples = uniform_samples(kept=300, chains=5)
    samples[:, 2] += 1.0  # chaîne 2 aberrante
    results = with_posterior(direct, samples)
    everything = convergence(results).data
    selected = convergence(results, chains=[0, 1, 3, 4]).data
    assert selected["rhat_max"].value < 1.05 < everything["rhat_max"].value
    assert selected["rhat_max_all"].value == pytest.approx(everything["rhat_max_all"].value)  # indépendant de la sélection
    assert convergence(results, burn=100, thin=2).data["samples_per_chain"].value == 100
    assert np.isnan(convergence(results, chains=[0, 1]).data["rhat_without_chain"].value).all()  # R̂ demande 2 chaînes


def test_energy_terms_add_up_and_the_implicit_rmse_is_the_rms_residual(direct):
    data = energy_breakdown(direct).data
    _, measured, simulated = __import__("molonari_twin.questions.sensors", fromlist=["sensor_series"]).sensor_series(direct)
    residual = simulated - measured
    n, sigma = residual.size, 0.1
    assert data["n_points"].value == n and data["reference"].value == n / 2
    assert data["log_term"].value == pytest.approx(n * np.log(sigma**2))
    assert data["residual_term"].value == pytest.approx(np.sum(residual**2) / (2 * sigma**2))
    assert data["log_term"].value + data["residual_term"].value == pytest.approx(data["energy"].value)
    assert data["implicit_rmse"].value == pytest.approx(np.sqrt(np.mean(residual**2)))
    assert data["ratio"].value == pytest.approx(data["residual_term"].value / (n / 2))
    assert energy_breakdown(direct, sigma=0.2).data["sigma"].value == 0.2


def test_with_the_right_sigma_the_residual_term_is_near_its_reference(direct):
    noise = np.random.default_rng(5).normal(0.0, 0.1, len(direct.residuals))
    data = energy_breakdown(with_residuals(direct, noise)).data
    assert data["ratio"].value == pytest.approx(1.0, abs=0.25)  # 240 résidus : écart-type du rapport √(2/240) ≈ 0,09


@pytest.mark.slow
def test_energy_breakdown_equals_the_energy_of_the_engine(tmp_path):
    import jax
    import jax.numpy as jnp
    from molonari_twin.engine.jax.forward import build_geometry, energy, make_forward

    inputs, layers = synthetic_inputs(nb_times=80), [LayerSpec("a", 0.4)]
    forward = run_forward(inputs, ForwardConfig(layers, nb_cells=20))
    results = results_of(tmp_path, forward)
    geom = build_geometry(inputs, [0.4], 20)
    fields = jax.jit(make_forward(geom))(jnp.asarray([layer.physical() for layer in layers], jnp.float64))
    expected = float(energy(fields["temperature"], geom, inputs.sigma_temperature**2))
    assert energy_breakdown(results).data["energy"].value == pytest.approx(expected, rel=1e-9)
