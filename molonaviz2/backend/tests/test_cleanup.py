"""Nettoyage des mesures : conversion tension → charge, fenêtre de dates, valeurs aberrantes."""

from datetime import datetime

import numpy as np
import pandas as pd
import pytest
from conftest import FIXTURES
from molonari_twin.domain.models import PressureSensor
from molonari_twin.errors import UserError
from molonari_twin.io.point_import import read_pressure_file, read_temperature_file
from molonari_twin.processing.cleanup import (
    CleanupSettings,
    clean,
    read_cleaned_csv,
    write_cleaned_csv,
)

SENSOR = PressureSensor("P", "Hobo", None, intercept=1.5, du_dh=-8.0, du_dt=0.01, sigma=0.01, thermometer="T")


def raw(n=50, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=n, freq="15min")
    temperature = pd.DataFrame({"date": dates, **{f"t{i}": 12 + rng.normal(0, 0.1, n) for i in range(1, 5)}})
    pressure = pd.DataFrame({"date": dates, "voltage": 1.4 + rng.normal(0, 0.01, n), "temp_bed": 14 + rng.normal(0, 0.1, n)})
    return temperature, pressure


def test_pressure_formula():
    temperature, pressure = raw(3)
    pressure.loc[:, ["voltage", "temp_bed"]] = [[1.42, 14.0], [1.30, 20.0], [1.5, 0.0]]
    result = clean(temperature, pressure, SENSOR, CleanupSettings())
    expected = [(1.42 - 14.0 * 0.01 - 1.5) / -8.0, (1.30 - 20.0 * 0.01 - 1.5) / -8.0, 0.0]
    np.testing.assert_allclose(result.cleaned["pressure"], expected)


def test_only_common_dates_are_kept():
    temperature, pressure = raw(10)
    pressure = pressure.iloc[2:]
    result = clean(temperature, pressure, SENSOR, CleanupSettings())
    assert len(result.cleaned) == 8
    assert (result.unmatched_temperature, result.unmatched_pressure) == (2, 0)


def test_iqr_and_zscore_remove_known_outliers():
    temperature, pressure = raw(200)
    temperature.loc[50, "t2"] = 40.0  # pic évident
    pressure.loc[120, "voltage"] = 3.0
    settings = CleanupSettings(methods={"temp_bed": "none", "t1": "none", "t2": "iqr", "t3": "none", "t4": "none", "pressure": "zscore"})
    result = clean(temperature, pressure, SENSOR, settings)
    assert result.reason[50] == "iqr:t2"
    assert result.reason[120] == "zscore:pressure"
    assert result.kept.sum() >= 190  # bruit gaussien : quelques faux positifs IQR au plus
    assert not result.cleaned["date"].isin([temperature["date"][50], temperature["date"][120]]).any()


def test_date_window_and_manual_exclusion():
    temperature, pressure = raw(20)
    dates = temperature["date"]
    settings = CleanupSettings(start=dates[5].to_pydatetime(), end=dates[14].to_pydatetime(), excluded=[dates[7].to_pydatetime()])
    result = clean(temperature, pressure, SENSOR, settings)
    assert len(result.cleaned) == 9
    assert result.counts() == {"outside_window": 10, "manual": 1}


def test_outliers_are_computed_on_the_window_only():
    temperature, pressure = raw(100)
    temperature.loc[:9, "t1"] = 100.0  # période aberrante hors fenêtre : ne doit pas fausser les quartiles
    temperature.loc[60, "t1"] = 13.5
    settings = CleanupSettings(start=temperature["date"][10].to_pydatetime(), methods={"t1": "iqr"})
    result = clean(temperature, pressure, SENSOR, settings)
    assert result.reason[60] == "iqr:t1"


def test_settings_round_trip():
    settings = CleanupSettings(start=datetime(2020, 1, 1), methods={"t1": "iqr"}, excluded=[datetime(2020, 1, 2)])
    again = CleanupSettings.from_dict(settings.to_dict())
    assert again.start == settings.start and again.excluded == settings.excluded
    assert again.methods["t1"] == "iqr" and again.methods["pressure"] == "none"


def test_unusable_calibration_and_disjoint_files():
    temperature, pressure = raw(5)
    with pytest.raises(UserError, match="dU/dH"):
        clean(temperature, pressure, PressureSensor("P", "", None, 1, 0.0, 0, 0.01, "T"), CleanupSettings())
    pressure["date"] = pressure["date"] + pd.Timedelta(minutes=1)
    with pytest.raises(UserError, match="no date in common"):
        clean(temperature, pressure, SENSOR, CleanupSettings())


def test_real_point034_cleanup():
    temperature = read_temperature_file(FIXTURES / "point034" / "point034_T_measures.csv").frame
    pressure = read_pressure_file(FIXTURES / "point034" / "point034_P_measures.csv").frame
    sensor = PressureSensor("P508", "", None, intercept=1.5, du_dh=-8.9, du_dt=-0.014, sigma=0.01, thermometer="T001")
    result = clean(temperature, pressure, sensor, CleanupSettings(methods={v: "zscore" for v in ["t1", "t2", "t3", "t4"]}))
    assert 1300 < len(result.cleaned) <= 1436
    assert result.cleaned["date"].is_monotonic_increasing
    assert not result.cleaned.isna().any().any()


def test_csv_export_round_trip(tmp_path):
    temperature, pressure = raw(10)
    cleaned = clean(temperature, pressure, SENSOR, CleanupSettings()).cleaned
    path = tmp_path / "cleaned.csv"
    write_cleaned_csv(cleaned, path)
    again = read_cleaned_csv(path)
    pd.testing.assert_frame_equal(again, cleaned.astype({"date": "datetime64[ns]"}), check_exact=False, rtol=1e-5)
