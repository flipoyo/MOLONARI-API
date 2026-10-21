"""Questions : réponses typées calculées à partir d'un résultat, sans Qt."""

import dataclasses

import numpy as np
import pytest
from conftest import synthetic_inputs

from molonaviz2.domain.compute import InversionConfig, LayerSpec
from molonaviz2.engine.api import run_inversion
from molonaviz2.errors import UserError
from molonaviz2.questions import (
    MEASUREMENT_LEVELS,
    Answer,
    Quantity,
    coverage,
    exchanged_volume,
    measurement_interval,
    observed_at,
    predictive_band,
    sensor_intervals,
    sensor_samples,
    z_score,
)
from molonaviz2.twin.results import load_results, write_results


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


def test_measurement_interval_is_reexported_by_the_interface():
    units = pytest.importorskip("molonaviz2.ui.units")
    assert units.z_score is z_score and units.measurement_interval is measurement_interval
    assert units.MEASUREMENT_LEVELS is MEASUREMENT_LEVELS


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
