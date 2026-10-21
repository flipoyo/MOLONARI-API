"""Questions : réponses typées calculées à partir d'un résultat, sans Qt."""

import dataclasses

import numpy as np
import pytest

from molonaviz2.questions import (
    MEASUREMENT_LEVELS,
    Answer,
    Quantity,
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
