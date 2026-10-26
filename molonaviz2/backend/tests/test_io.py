"""Lecture des fichiers de terrain : dates, laboratoire, point (info + mesures HOBO)."""

from datetime import datetime

import pandas as pd
import pytest
from conftest import FIXTURES
from molonari_twin.errors import UserError
from molonari_twin.io.dates import parse_date, parse_dates
from molonari_twin.io.lab_import import import_lab_directory, parse_shaft
from molonari_twin.io.point_import import (
    find_point_files,
    read_point_info,
    read_pressure_file,
    read_temperature_file,
)

# Dates ---------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "values, expected_first, expected_format",
    [
        (["06/27/16 12:00:00 PM", "06/27/16 12:15:00 PM"], datetime(2016, 6, 27, 12, 0), "%m/%d/%y %I:%M:%S %p"),
        (["27/08/2019 14:00", "27/08/2019 14:15"], datetime(2019, 8, 27, 14, 0), "%d/%m/%Y %H:%M"),
        (["2009/9/24 16:00", "2009/9/24 16:15"], datetime(2009, 9, 24, 16, 0), "%Y/%m/%d %H:%M"),
        (["2016-06-27 12:00:00", "2016-06-27 12:15:00"], datetime(2016, 6, 27, 12, 0), "%Y-%m-%d %H:%M:%S"),
    ],
)
def test_date_formats(values, expected_first, expected_format):
    dates, fmt, chronological = parse_dates(values)
    assert chronological
    assert dates.iloc[0] == pd.Timestamp(expected_first)
    assert fmt == expected_format


def test_day_month_ambiguity_resolved_by_chronology():
    # Lu en mois/jour, la série reculerait (01/02 → 13/01 impossible) : seul jour/mois est chronologique.
    dates, fmt, _ = parse_dates(["12/01/2020 10:00", "13/01/2020 10:00", "01/02/2020 10:00"])
    assert fmt.startswith("%d/%m")
    assert dates.is_monotonic_increasing


def test_mixed_formats_in_one_column():
    dates, fmt, chronological = parse_dates(["06/27/16 11:45:00 PM", "2016/6/28 0:00", "2016/6/28 0:15"])
    assert chronological
    assert " + " in fmt
    assert list(dates) == [pd.Timestamp(2016, 6, 27, 23, 45), pd.Timestamp(2016, 6, 28, 0, 0), pd.Timestamp(2016, 6, 28, 0, 15)]


def test_unknown_date_format_names_the_value():
    with pytest.raises(UserError, match="hier soir"):
        parse_dates(["hier soir", "ce matin"])


def test_single_dates():
    assert parse_date("2020/04/09  12:02:35") == datetime(2020, 4, 9, 12, 2, 35)  # double espace (fichiers labo)
    assert parse_date("13_10_23") == datetime(2023, 10, 13)  # jour_mois_an, comme les noms de dossiers
    assert parse_date("") is None


# Laboratoire ---------------------------------------------------------------------------


def test_import_real_laboratory():
    result = import_lab_directory(FIXTURES / "lab", name="Montberneux")
    lab = result.lab
    assert lab.name == "Montberneux"
    assert [t.name for t in lab.thermometers] == ["T001", "T002", "T003"]
    assert len(lab.pressure_sensors) == 9 and len(lab.shafts) == 7
    assert result.issues == []
    t001 = lab.thermometers[0]
    assert (t001.manufacturer, t001.reference, t001.sigma) == ("Mines", "Corefge", 0.03)
    p001 = next(p for p in lab.pressure_sensors if p.name == "P001")
    assert p001.calibration_date == datetime(2020, 4, 9, 12, 2, 35)
    assert (p001.intercept, p001.du_dh, p001.du_dt, p001.sigma, p001.thermometer) == (0.36, -12.7, 0.04, 0.01, "T001")
    p505 = next(p for p in lab.pressure_sensors if p.name == "P505")  # clé « Sensor_Name » et « dU/dH »
    assert p505.du_dh == pytest.approx(-8.89963363336198)
    s001 = next(s for s in lab.shafts if s.name == "S001")
    assert s001.depths == (0.2, 0.4, 0.6, 0.8) and s001.thermometer == "T001"


def test_lab_import_reports_invalid_files_and_keeps_the_rest(tmp_path):
    (tmp_path / "temperature_sensors").mkdir()
    (tmp_path / "shafts").mkdir()
    (tmp_path / "temperature_sensors" / "good.csv").write_text("Cons_Name,A\nCons_Ref,B\nT_Sensor_Name,T9\nSigma_Meas_T,\"0,05\"\n")
    (tmp_path / "temperature_sensors" / "short.csv").write_text("Cons_Name,A\nCons_Ref,B\n")
    (tmp_path / "shafts" / "bad_depths.csv").write_text("Shaft_Name,S9\nDatalogger,Hobo\nT_Sensor_Name,T9\nSensors_Depth,\"[0.4, 0.2, 0.6, 0.8]\"\n")
    result = import_lab_directory(tmp_path)
    assert [t.name for t in result.lab.thermometers] == ["T9"]
    assert result.lab.thermometers[0].sigma == 0.05  # virgule décimale
    files = {issue.file: issue for issue in result.issues}
    assert "temperature_sensors/short.csv" in files and "4" in files["temperature_sensors/short.csv"].message
    assert files["shafts/bad_depths.csv"].line == 4


def test_lab_import_rejects_unrelated_folder(tmp_path):
    with pytest.raises(UserError):
        import_lab_directory(tmp_path)


def test_shaft_depths_with_semicolons(tmp_path):
    path = tmp_path / "s.csv"
    path.write_text("Shaft_Name,S1\nDatalogger,Hobo\nT_Sensor_Name,T1\nSensors_Depth,[0.1; 0.2; 0.3; 0.4]\n")
    assert parse_shaft(path).depths == (0.1, 0.2, 0.3, 0.4)


# Point ---------------------------------------------------------------------------------


def test_find_point034_files():
    files = find_point_files(FIXTURES / "point034")
    assert files.info.name == "Point034_info.csv"
    assert files.pressure.name == "point034_P_measures.csv"
    assert files.temperature.name == "point034_T_measures.csv"
    assert files.notice.name == "notice034.txt"
    assert files.scheme.name == "point34_schema.png"


def test_point034_info():
    info = read_point_info(FIXTURES / "point034" / "Point034_info.csv")
    assert (info.name, info.pressure_sensor, info.shaft) == ("Point034", "P508", "S007")
    assert info.setup_date == datetime(2016, 6, 27, 12, 0)
    assert info.measure_date == datetime(2016, 7, 12, 10, 30)
    assert (info.river_bed, info.offset) == (0.505, 0.0)


def test_point034_hobo_measures():
    pressure = read_pressure_file(FIXTURES / "point034" / "point034_P_measures.csv")
    temperature = read_temperature_file(FIXTURES / "point034" / "point034_T_measures.csv")
    assert list(pressure.frame.columns) == ["date", "voltage", "temp_bed"]
    assert list(temperature.frame.columns) == ["date", "t1", "t2", "t3", "t4"]
    assert pressure.kept_rows == 1460  # ce fichier mélange deux écritures de dates
    assert " + " in pressure.date_format
    assert temperature.kept_rows == 1436 and len(temperature.dropped) == 2
    first = pressure.frame.iloc[0]
    assert first["date"] == pd.Timestamp(2016, 6, 27, 12, 0)
    assert (first["voltage"], first["temp_bed"]) == (1.26787, 18.342)
    assert temperature.frame["date"].is_monotonic_increasing
    assert temperature.frame.iloc[0][["t1", "t2", "t3", "t4"]].tolist() == [13.810, 13.834, 13.858, 13.810]


def test_fahrenheit_is_converted_and_extra_columns_ignored():
    parsed = read_pressure_file(FIXTURES / "hobo_fahrenheit_P.csv")
    assert parsed.converted_units == ["°F → °C"]
    first = parsed.frame.iloc[0]
    assert first["voltage"] == 0.00835
    assert first["temp_bed"] == pytest.approx((487.387 - 32) * 5 / 9)  # 1re colonne de température, pas la « boîte »


def test_spreadsheet_altered_file_is_refused_with_line_numbers():
    """Extrait réel (Point2024-3) : des dates « 09/01/24 » reconverties en « 2009/1/24 » par un tableur.
    Trier ces lignes enverrait des mesures de 2024 en 2009 : le fichier doit être refusé."""
    with pytest.raises(UserError, match=r"line 13 \(2009/1/24 0:00\) comes after line 12"):
        read_temperature_file(FIXTURES / "hobo_spreadsheet_altered_T.csv")


def test_logger_event_rows_are_reported_with_line_numbers(tmp_path):
    path = tmp_path / "t.csv"
    path.write_text(
        "\"Titre de tracé : t1\"\n"
        "\"#\",\"Date Heure, GMT+01:00\",\"Temp., °C (a)\",\"Temp., °C (b)\",\"Temp., °C (c)\",\"Temp., °C (d)\",\"Arrêté\"\n"
        "1,10/13/23 01:45:00 PM,10.1,10.2,10.3,10.4,\n"
        "2,10/13/23 01:50:39 PM,,,,,Enregistré\n"
        "3,10/13/23 02:00:00 PM,10.5,abc,10.7,10.8,\n"
        "4,10/13/23 02:15:00 PM,10.9,11.0,11.1,11.2,\n"
    )
    parsed = read_temperature_file(path)
    assert parsed.kept_rows == 2
    reasons = dict(parsed.dropped)
    assert set(reasons) == {4, 5}
    assert "abc" in reasons[5]


def test_clean_format_without_title_and_semicolon_separator(tmp_path):
    path = tmp_path / "p.csv"
    path.write_text("Date;Voltage;Temperature\n27/06/2016 12:00;1,25;18,3\n27/06/2016 12:15;1,26;18,1\n")
    parsed = read_pressure_file(path)
    assert parsed.frame["voltage"].tolist() == [1.25, 1.26]
    assert parsed.frame["temp_bed"].tolist() == [18.3, 18.1]


def test_file_without_measures_is_rejected(tmp_path):
    path = tmp_path / "empty.csv"
    path.write_text("Titre\nDate,Temp\n")
    with pytest.raises(UserError):
        read_temperature_file(path)


# Incertitudes σ des instruments (P2.1c) ---------------------------------------------------


def test_sigma_is_read_by_name(tmp_path):
    """σ est lu sous son nom (Sigma_Meas_T, Sigma_Meas_P), sans tenir compte de la casse ni de la position."""
    (tmp_path / "pressure_sensors").mkdir()
    (tmp_path / "temperature_sensors").mkdir()
    # Sigma_Meas_P est échangé avec dU/dT : lu par position, σ vaudrait 0.5
    (tmp_path / "pressure_sensors" / "P9.csv").write_text(
        "Sensor_Name,P9\nDatalogger,Hobo\nCalibration_Date,2016/01/12 10:10:10\nIntercept,1.2\ndU/dH,-1.4\n"
        "sigma_meas_p,0.02\ndU/dT,0.5\nThermometer name,T9\n"
    )
    (tmp_path / "temperature_sensors" / "T9.csv").write_text("Cons_Name,A\nCons_Ref,B\nT_Sensor_Name,T9\nSigma_Meas_T,0.07\n")
    lab = import_lab_directory(tmp_path).lab
    assert lab.pressure_sensors[0].sigma == 0.02 and lab.thermometers[0].sigma == 0.07


def test_sigma_falls_back_to_its_position_without_its_name(tmp_path):
    (tmp_path / "temperature_sensors").mkdir()
    (tmp_path / "temperature_sensors" / "T9.csv").write_text("Cons_Name,A\nCons_Ref,B\nT_Sensor_Name,T9\nIncertitude,0.09\n")
    assert import_lab_directory(tmp_path).lab.thermometers[0].sigma == 0.09


def test_sigma_that_is_not_a_number_discards_the_file_and_names_the_key(tmp_path):
    (tmp_path / "temperature_sensors").mkdir()
    (tmp_path / "temperature_sensors" / "T9.csv").write_text("Cons_Name,A\nCons_Ref,B\nT_Sensor_Name,T9\nSigma_Meas_T,beaucoup\n")
    result = import_lab_directory(tmp_path)
    assert result.lab.thermometers == []
    issue = result.issues[0].message
    assert (issue.code, issue.params) == ("E_LAB_FILE_NOT_A_NUMBER", {"key": "Sigma_Meas_T", "value": "beaucoup"})


# Unités non prises en charge (P2.1d) --------------------------------------------------------


@pytest.mark.parametrize("unit", ["kPa", "psi", "bar", "mbar", "Pa"])
def test_pressure_in_a_pressure_unit_is_refused(tmp_path, unit):
    path = tmp_path / "p.csv"
    path.write_text(
        '"Titre de tracé : 1"\n'
        f'"#","Date Heure, GMT+01:00","Pression absolue, {unit} (LGR S/N: 1)","Temp., °C (LGR S/N: 1)"\n'
        "1,06/27/16 12:00:00 PM,101.3,18.3\n2,06/27/16 12:15:00 PM,101.4,18.1\n"
    )
    with pytest.raises(UserError) as error:
        read_pressure_file(path)
    assert error.value.code == "E_MEASURES_UNIT_UNSUPPORTED"
    assert error.value.params["file"] == "p.csv" and unit in error.value.params["column"]
