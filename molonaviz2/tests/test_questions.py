"""Questions : réponses typées calculées à partir d'un résultat, sans Qt."""

import dataclasses

import numpy as np
import pytest

from molonaviz2.errors import UserError
from molonaviz2.questions import (
    MEASUREMENT_LEVELS,
    Answer,
    Quantity,
    exchanged_volume,
    measurement_interval,
    sensor_intervals,
    z_score,
)


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
