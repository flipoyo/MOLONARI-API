"""Pipeline du brut au résultat (P2.1) : du dossier d'un point et d'un laboratoire aux entrées d'une colonne,
et refus explicite de ce qui n'est pas pris en charge."""

import shutil
import subprocess
import sys
import textwrap
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest
from conftest import EXAMPLE_TWINS, FIXTURES
from molonari_twin.domain.models import CLEANED_COLUMNS
from molonari_twin.errors import Issue, UserError
from molonari_twin.io.dates import PROVISIONAL_UTC_OFFSET
from molonari_twin.processing.cleanup import CleanupSettings, read_cleaned_csv
from molonari_twin.twin.pipeline import (
    clean_point,
    column_inputs,
    load_point,
    read_raw_point,
    resolve_instruments,
    run_pipeline,
)

LAB = FIXTURES / "lab"
POINT034 = FIXTURES / "point034"
# Fenêtre de twin_01 : installation et retrait du dispositif exclus.
WINDOW = CleanupSettings(start=datetime(2016, 6, 27, 13, 0), end=datetime(2016, 7, 12, 10, 15))


@pytest.fixture
def point(tmp_path):
    """Copie modifiable du dossier de Point034 (mesures et fichier info seulement)."""
    folder = tmp_path / "Point034"
    folder.mkdir()
    for name in ("Point034_info.csv", "point034_P_measures.csv", "point034_T_measures.csv"):
        shutil.copy(POINT034 / name, folder / name)
    return folder


def refusal(call) -> UserError:
    with pytest.raises(UserError) as error:
        call()
    return error.value


# Du brut au résultat ---------------------------------------------------------------------------------------------


def test_raw_point034_gives_the_cleaned_measures_of_the_example_twin():
    """Critère de P2.1 : le brut de Point034 redonne les mesures nettoyées (twin_01/measurements.csv, exportées
    avec 6 chiffres significatifs, d'où la tolérance sur la charge)."""
    expected = read_cleaned_csv(EXAMPLE_TWINS / "twin_01" / "measurements.csv")
    result = run_pipeline(POINT034, LAB)
    assert len(result.cleaned) == len(expected) == 1436
    pd.testing.assert_frame_equal(result.cleaned, expected, check_exact=False, rtol=1e-5, atol=1e-7)
    assert list(result.cleaned.columns) == CLEANED_COLUMNS


def test_the_window_of_the_example_twin_keeps_the_same_measures():
    expected = read_cleaned_csv(EXAMPLE_TWINS / "twin_01" / "measurements.csv")
    expected = expected[(expected["date"] >= WINDOW.start) & (expected["date"] <= WINDOW.end)].reset_index(drop=True)
    result = run_pipeline(POINT034, LAB, WINDOW)
    assert len(result.cleaned) == len(expected) == 1430
    pd.testing.assert_frame_equal(result.cleaned, expected, check_exact=False, rtol=1e-5, atol=1e-7)
    assert result.cleanup.counts() == {"outside_window": 6}


def test_column_inputs_come_from_the_laboratory():
    """Profondeurs de la tige S007, σ de pression du capteur P508 (Sigma_Meas_P) et σ de température du
    thermomètre T001 (Sigma_Meas_T)."""
    result = run_pipeline(POINT034, LAB, WINDOW)
    inputs = result.inputs
    assert inputs.sensor_depths == (0.1, 0.2, 0.3, 0.4) and inputs.offset == 0.0
    assert (inputs.sigma_pressure, inputs.sigma_temperature) == (0.01, 0.03)
    assert inputs.nb_times == 1430 and inputs.t_sensors.shape == (1430, 3)
    instruments = result.point.instruments
    assert (instruments.pressure_sensor.name, instruments.shaft.name, instruments.thermometer.name) == ("P508", "S007", "T001")
    assert result.point.raw.name == "Point034" and result.point.lab_issues == []


def test_the_steps_chain_like_run_pipeline():
    loaded = load_point(POINT034, LAB)
    cleanup = clean_point(loaded, WINDOW)
    inputs = column_inputs(loaded, cleanup)
    whole = run_pipeline(POINT034, LAB, WINDOW).inputs
    np.testing.assert_array_equal(inputs.head, whole.head)
    np.testing.assert_array_equal(inputs.times, whole.times)


def test_cleanup_settings_are_applied():
    settings = CleanupSettings(methods={**CleanupSettings().methods, "t1": "iqr"}, excluded=[datetime(2016, 6, 28, 0, 0)])
    result = run_pipeline(POINT034, LAB, settings)
    counts = result.cleanup.counts()
    assert counts["manual"] == 1 and counts.get("iqr:t1", 0) > 0
    assert len(result.cleaned) == 1436 - sum(counts.values())


def test_sigmas_can_be_given_explicitly():
    inputs = run_pipeline(POINT034, LAB, sigma_temperature=0.2, sigma_pressure=0.02).inputs
    assert (inputs.sigma_pressure, inputs.sigma_temperature) == (0.02, 0.2)


@pytest.mark.parametrize("name, value", [("sigma_temperature", 0.0), ("sigma_pressure", -0.01)])
def test_a_sigma_that_is_not_positive_is_refused(name, value):
    error = refusal(lambda: run_pipeline(POINT034, LAB, **{name: value}))
    assert error.code == "E_PIPELINE_SIGMA_POSITIVE" and error.params == {"name": name, "sigma": value}


def test_a_laboratory_sigma_that_is_not_positive_is_refused(tmp_path):
    lab = tmp_path / "lab"
    shutil.copytree(LAB, lab)
    path = lab / "pressure_sensors" / "P508.csv"
    path.write_text(path.read_text(encoding="utf-8").replace("Sigma_Meas_P,0.01", "Sigma_Meas_P,0"), encoding="utf-8")
    error = refusal(lambda: run_pipeline(POINT034, lab))
    assert (error.code, error.params["name"]) == ("E_PIPELINE_SIGMA_POSITIVE", "P508")


# Ce qui n'est pas pris en charge ---------------------------------------------------------------------------------


# Fuseau des dates (P2.1f) ----------------------------------------------------------------------------------------
#
# Convention : celle des données brutes. L'en-tête de la colonne des dates déclare un décalage fixe à UTC
# (« Date Heure, GMT+01:00 »), qu'une date peut aussi porter (« 2016-06-27T12:00:00+01:00 »).

DECLARED = "GMT+01:00"
PLUS_ONE = timedelta(hours=1)
PLUS_TWO = timedelta(hours=2)


def read_date(text):
    """Les exports de Point034 mêlent « 06/27/16 12:00:00 PM » et, à partir du 1er juillet, « 07/01/2016 00:00 »."""
    for fmt in ("%m/%d/%y %I:%M:%S %p", "%m/%d/%Y %H:%M"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    raise ValueError(text)


def write_date(moment):
    return moment.strftime("%m/%d/%y %I:%M:%S %p")


def rewrite_dates(path, change, header=None):
    """Réécrit les dates d'un export HOBO de Point034 (« MM/DD/YY hh:mm:ss AM/PM ») avec `change(datetime) -> texte`,
    et remplace éventuellement le décalage de l'en-tête de la colonne des dates."""
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    for i, line in enumerate(lines):
        if DECLARED in line and header is not None:
            lines[i] = line.replace(DECLARED, header)
        cells = line.split(",", 2)
        if len(cells) == 3 and cells[0].replace(".", "").isdigit():
            cells[1] = change(read_date(cells[1]))
            lines[i] = ",".join(cells)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def as_iso(offset):
    return lambda moment: moment.strftime("%Y-%m-%dT%H:%M:%S") + offset


def same_as_default(point, **kwargs):
    default = run_pipeline(POINT034, LAB)
    result = run_pipeline(point, LAB, **kwargs)
    pd.testing.assert_frame_equal(result.cleaned, default.cleaned)
    return result


def timezone_warnings(raw, code=None):
    return [w for w in raw.warnings if "TIMEZONE" in w.code and (code is None or w.code == code)]


def test_the_raw_data_declare_gmt_plus_1_and_the_provisional_convention_reads_it_as_utc_plus_2():
    """Tous les bruts réels (Point034, 047, 051, 52, 53) déclarent « GMT+01:00 », en été comme en hiver. Convention
    PROVISOIRE (TODO(fuseau)) : UTC+2 toute l'année. Les dates restent écrites telles quelles, seul `utc_offset` change."""
    assert PROVISIONAL_UTC_OFFSET == PLUS_TWO
    raw = read_raw_point(POINT034)
    assert (raw.temperature.utc_offset, raw.pressure.utc_offset, raw.utc_offset) == (PLUS_TWO, PLUS_TWO, PLUS_TWO)
    overridden = timezone_warnings(raw)
    assert sorted((w.code, w.params["file"], w.params["declared"], w.params["applied"]) for w in overridden) == [
        ("W_MEASURES_TIMEZONE_OVERRIDDEN", "point034_P_measures.csv", "+01:00", "+02:00"),
        ("W_MEASURES_TIMEZONE_OVERRIDDEN", "point034_T_measures.csv", "+01:00", "+02:00"),
    ]
    result = run_pipeline(POINT034, LAB)
    assert result.utc_offset == PLUS_TWO and result.point.utc_offset == PLUS_TWO
    assert result.cleaned["date"].iloc[0] == pd.Timestamp("2016-06-27 12:00")  # toujours dans le repère du fichier


@pytest.mark.parametrize("timezone", ["+02:00", "+0200", "GMT+2", "UTC+02:00", "utc+2"])
def test_a_requested_time_zone_equal_to_the_applied_one_is_accepted(timezone):
    assert same_as_default(POINT034, timezone=timezone).utc_offset == PLUS_TWO
    assert load_point(POINT034, LAB, timezone=timezone).utc_offset == PLUS_TWO


@pytest.mark.parametrize("timezone", ["UTC", "Z", "+01:00", "GMT-05:30"])
def test_a_requested_time_zone_that_contradicts_the_files_is_a_conflict(timezone):
    error = refusal(lambda: run_pipeline(POINT034, LAB, timezone=timezone))
    assert error.code == "E_PIPELINE_TIMEZONE_CONFLICT" and error.params["declared"] == "+02:00"


@pytest.mark.parametrize("timezone", ["Europe/Paris", "CET", "1", "GMT+15:00", "GMT+01:75"])
def test_a_time_zone_that_is_not_a_fixed_offset_is_refused(timezone):
    """Un nom de zone a un changement d'heure : il ne peut pas être le décalage fixe que déclarent les bruts."""
    for call in (lambda: run_pipeline(POINT034, LAB, timezone=timezone), lambda: load_point(POINT034, LAB, timezone=timezone)):
        error = refusal(call)
        assert error.code == "E_PIPELINE_TIMEZONE_INVALID" and error.params == {"value": timezone}


def test_dates_without_a_declaration_are_kept_as_written_and_flagged(point):
    """Rien n'est supposé : sans fuseau déclaré, les dates ne changent pas, `utc_offset` est None et c'est signalé."""
    for name in ("point034_P_measures.csv", "point034_T_measures.csv"):
        path = point / name
        path.write_text(path.read_text(encoding="utf-8-sig").replace(", " + DECLARED, ""), encoding="utf-8")
    raw = read_raw_point(point)
    assert raw.utc_offset is None
    assert sorted(w.params["file"] for w in raw.warnings if w.code == "W_MEASURES_TIMEZONE_UNDECLARED") == [
        "point034_P_measures.csv", "point034_T_measures.csv"]
    assert same_as_default(point).utc_offset is None


def test_a_requested_time_zone_is_refused_when_the_files_declare_none(point):
    """La convention est celle des données brutes : le paramètre ne peut pas combler un fuseau absent."""
    for name in ("point034_P_measures.csv", "point034_T_measures.csv"):
        path = point / name
        path.write_text(path.read_text(encoding="utf-8-sig").replace(", " + DECLARED, ""), encoding="utf-8")
    error = refusal(lambda: run_pipeline(point, LAB, timezone="+01:00"))
    assert (error.code, error.params) == ("E_PIPELINE_TIMEZONE_UNDECLARED", {"requested": "+01:00"})


@pytest.mark.parametrize("offset", ["+01:00", "+0100", "+01"])
def test_dates_with_an_explicit_offset_are_read_in_that_frame(point, offset):
    """Jeu avec décalage explicite : « 2016-06-27T12:00:00+01:00 » partout, et un en-tête sans fuseau."""
    for name in ("point034_P_measures.csv", "point034_T_measures.csv"):
        rewrite_dates(point / name, as_iso(offset), header="")
    raw = read_raw_point(point)
    assert raw.utc_offset == PLUS_ONE
    assert not [w for w in raw.warnings if "TIMEZONE" in w.code]
    assert same_as_default(point, timezone="UTC+1").utc_offset == PLUS_ONE


def test_an_explicit_offset_that_matches_the_header_is_accepted(point):
    for name in ("point034_P_measures.csv", "point034_T_measures.csv"):
        rewrite_dates(point / name, as_iso("+01:00"))
    assert same_as_default(point).utc_offset == PLUS_ONE


def test_an_explicit_offset_in_the_east_of_utc_gives_the_same_instants_as_utc(point):
    """UTC = date - décalage : « 12:00Z » et « 13:00+01:00 » sont le même instant, lus dans leur repère déclaré."""
    for name in ("point034_P_measures.csv", "point034_T_measures.csv"):
        rewrite_dates(point / name, lambda moment: (moment - PLUS_ONE).strftime("%Y-%m-%dT%H:%M:%SZ"), header="")
    raw = read_raw_point(point)
    assert raw.utc_offset == timedelta(0)
    utc = raw.temperature.frame["date"]
    reference = read_raw_point(POINT034).temperature.frame["date"] - PLUS_ONE
    assert utc.equals(reference)


def test_an_explicit_offset_that_contradicts_the_header_is_a_conflict(point):
    calls = iter(range(10**6))

    def change(moment):
        return as_iso("+02:00" if next(calls) == 4 else "+01:00")(moment)

    rewrite_dates(point / "point034_T_measures.csv", change)
    error = refusal(lambda: run_pipeline(point, LAB))
    assert error.code == "E_MEASURES_TIMEZONE_CONFLICT"
    assert error.params["file"] == "point034_T_measures.csv"
    assert (error.params["offset"], error.params["declared"]) == ("+02:00", "+01:00")
    assert error.params["value"] == "2016-06-27T13:00:00+02:00" and error.params["line"] == 7


def test_explicit_offsets_that_differ_without_a_declaration_are_ambiguous(point):
    calls = iter(range(10**6))
    rewrite_dates(point / "point034_T_measures.csv", lambda m: as_iso("+02:00" if next(calls) % 2 else "+01:00")(m), header="")
    error = refusal(lambda: read_raw_point(point))
    assert error.code == "E_MEASURES_TIMEZONE_MIXED" and error.params["offsets"] == "+01:00, +02:00"


def test_an_offset_carried_by_only_some_dates_without_a_declaration_is_ambiguous(point):
    calls = iter(range(10**6))
    rewrite_dates(point / "point034_T_measures.csv", lambda m: as_iso("+01:00" if next(calls) else "")(m), header="")
    assert refusal(lambda: read_raw_point(point)).code == "E_MEASURES_TIMEZONE_PARTIAL"


def test_an_offset_beyond_the_possible_range_is_refused(point):
    rewrite_dates(point / "point034_T_measures.csv", write_date, header="GMT+15:00")
    assert refusal(lambda: read_raw_point(point)).code == "E_DATE_OFFSET_INVALID"


def test_a_pressure_file_declaring_plus_2_is_no_longer_in_conflict_with_a_temperature_file_declaring_plus_1(point):
    """Cas de Point2024-1 à 3 (P en GMT+02:00, T en GMT+01:00) : la convention provisoire lit les deux en UTC+2. Il n'y
    a rien à décaler, et seul l'en-tête +01:00 est signalé."""
    rewrite_dates(point / "point034_P_measures.csv", write_date, header="GMT+02:00")
    raw = read_raw_point(point)
    assert (raw.pressure.utc_offset, raw.temperature.utc_offset, raw.utc_offset) == (PLUS_TWO, PLUS_TWO, PLUS_TWO)
    assert [(w.code, w.params["file"]) for w in timezone_warnings(raw)] == [("W_MEASURES_TIMEZONE_OVERRIDDEN", "point034_T_measures.csv")]
    assert same_as_default(point).utc_offset == PLUS_TWO


@pytest.mark.parametrize("header, offset", [("GMT+03:00", 3), ("GMT-05:30", -5.5), ("UTC", 0)])
def test_a_header_other_than_plus_1_or_plus_2_keeps_its_own_offset(point, header, offset):
    for name in ("point034_P_measures.csv", "point034_T_measures.csv"):
        path = point / name
        path.write_text(path.read_text(encoding="utf-8-sig").replace(DECLARED, header), encoding="utf-8")
    raw = read_raw_point(point)
    assert raw.utc_offset == timedelta(hours=offset)
    assert not timezone_warnings(raw)


def test_a_header_declaring_plus_2_is_kept_without_a_warning(point):
    for name in ("point034_P_measures.csv", "point034_T_measures.csv"):
        path = point / name
        path.write_text(path.read_text(encoding="utf-8-sig").replace(DECLARED, "GMT+02:00"), encoding="utf-8")
    raw = read_raw_point(point)
    assert raw.utc_offset == PLUS_TWO and not timezone_warnings(raw)


def test_two_files_in_different_offsets_are_aligned_on_the_temperature_frame(point):
    """Température aux dates écrites avec leur décalage explicite +01:00 (jamais remplacé), pression déclarée en
    GMT+02:00 avec ses dates écrites une heure plus tard : mêmes instants, donc même résultat, et un avertissement dit
    que les dates de la pression ont été décalées vers le repère de la température."""
    rewrite_dates(point / "point034_T_measures.csv", as_iso("+01:00"), header="")
    rewrite_dates(point / "point034_P_measures.csv", lambda m: write_date(m + PLUS_ONE), header="GMT+02:00")
    raw = read_raw_point(point)
    assert (raw.pressure.utc_offset, raw.temperature.utc_offset, raw.utc_offset) == (PLUS_TWO, PLUS_ONE, PLUS_ONE)
    shifted = [w for w in raw.warnings if w.code == "W_PIPELINE_TIMEZONE_SHIFTED"]
    assert len(shifted) == 1
    assert shifted[0].params == {"file": "point034_P_measures.csv", "declared": "+02:00", "reference_file": "point034_T_measures.csv",
                                 "reference": "+01:00", "shift": "-01:00"}
    assert same_as_default(point).utc_offset == PLUS_ONE


def test_only_one_file_declaring_its_time_zone_is_ambiguous(point):
    path = point / "point034_P_measures.csv"
    path.write_text(path.read_text(encoding="utf-8-sig").replace(", " + DECLARED, ""), encoding="utf-8")
    error = refusal(lambda: read_raw_point(point))
    assert error.code == "E_PIPELINE_TIMEZONE_AMBIGUOUS"
    assert error.params == {"declared_file": "point034_T_measures.csv", "declared": "+02:00", "other_file": "point034_P_measures.csv"}


def test_spreadsheet_formats_are_refused_as_unsupported(point):
    (point / "point034_P_measures.csv").unlink()
    (point / "point034_P_measures.xlsx").write_bytes(b"PK\x03\x04")
    (point / "point034_P_measures.hobo").write_bytes(b"\x00")
    error = refusal(lambda: read_raw_point(point))
    assert error.code == "E_PIPELINE_FORMAT_UNSUPPORTED"
    assert error.params["files"] == "point034_P_measures.hobo, point034_P_measures.xlsx"
    assert refusal(lambda: run_pipeline(point, LAB)).code == "E_PIPELINE_FORMAT_UNSUPPORTED"


@pytest.mark.parametrize("missing, code", [("point034_P_measures.csv", "E_PIPELINE_NO_PRESSURE_FILE"),
                                           ("point034_T_measures.csv", "E_PIPELINE_NO_TEMPERATURE_FILE")])
def test_a_missing_measure_file_is_refused(point, missing, code):
    (point / missing).unlink()
    error = refusal(lambda: read_raw_point(point))
    assert (error.code, error.params["folder"]) == (code, point)


def test_a_missing_measure_file_is_only_a_warning_when_not_required(point):
    (point / "point034_P_measures.csv").unlink()
    raw = read_raw_point(point, require_measures=False)
    assert raw.pressure is None and raw.temperature is not None
    others = [w for w in raw.warnings if "TIMEZONE" not in w.code]
    assert [w.code for w in others] == ["E_PIPELINE_NO_PRESSURE_FILE"] and isinstance(others[0], Issue)
    (point / "point034_T_measures.csv").unlink()
    assert refusal(lambda: read_raw_point(point, require_measures=False)).code == "E_PIPELINE_NO_TEMPERATURE_FILE"


def test_a_folder_that_does_not_exist_is_refused(tmp_path):
    assert refusal(lambda: read_raw_point(tmp_path / "nowhere")).code == "E_FOLDER_NOT_FOUND"
    assert refusal(lambda: run_pipeline(POINT034, tmp_path / "nowhere")).code == "E_FOLDER_NOT_FOUND"


def test_pressure_in_kilopascals_is_refused(point):
    (point / "point034_P_measures.csv").write_text(
        '"Titre de tracé : 1"\n"#","Date Heure, GMT+01:00","Pression, kPa (LGR S/N: 1)","Temp., °C (LGR S/N: 1)"\n'
        "1,06/27/16 12:00:00 PM,101.3,18.3\n2,06/27/16 12:15:00 PM,101.4,18.1\n", encoding="utf-8")
    assert refusal(lambda: run_pipeline(point, LAB)).code == "E_MEASURES_UNIT_UNSUPPORTED"


def test_a_file_altered_by_a_spreadsheet_is_refused(point):
    shutil.copy(FIXTURES / "hobo_spreadsheet_altered_T.csv", point / "point034_T_measures.csv")
    assert refusal(lambda: run_pipeline(point, LAB)).code == "E_MEASURES_NOT_CHRONOLOGICAL"


def info_without(point, key):
    path = point / "Point034_info.csv"
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if not line.startswith(key)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_a_point_that_does_not_name_its_instruments_is_refused(point):
    info_without(point, "Shaft_Name")
    assert refusal(lambda: run_pipeline(point, LAB)).code == "E_PIPELINE_NO_SHAFT"


def test_a_point_without_info_is_refused(point):
    (point / "Point034_info.csv").unlink()
    error = refusal(lambda: run_pipeline(point, LAB))
    assert (error.code, error.params["folder"]) == ("E_PIPELINE_NO_INFO", point)


def test_a_pressure_sensor_missing_from_the_laboratory_is_refused(point, tmp_path):
    lab = tmp_path / "lab"
    shutil.copytree(LAB, lab)
    (lab / "pressure_sensors" / "P508.csv").unlink()
    error = refusal(lambda: run_pipeline(point, lab))
    assert (error.code, error.params["name"]) == ("E_POINT_PRESSURE_SENSOR_UNKNOWN", "P508")


def test_a_shaft_missing_from_the_laboratory_is_refused(point, tmp_path):
    lab = tmp_path / "lab"
    shutil.copytree(LAB, lab)
    (lab / "shafts" / "S007.csv").unlink()
    assert refusal(lambda: run_pipeline(point, lab)).code == "E_POINT_SHAFT_UNKNOWN"


def test_a_thermometer_missing_from_the_laboratory_is_refused_unless_sigma_is_given(point, tmp_path):
    lab = tmp_path / "lab"
    shutil.copytree(LAB, lab)
    (lab / "temperature_sensors" / "T001.csv").unlink()
    error = refusal(lambda: run_pipeline(point, lab))
    assert (error.code, error.params["name"], error.params["shaft"]) == ("E_PIPELINE_THERMOMETER_UNKNOWN", "T001", "S007")
    assert run_pipeline(point, lab, sigma_temperature=0.05).inputs.sigma_temperature == 0.05


def test_resolve_instruments_needs_the_info_file():
    raw = read_raw_point(POINT034)
    raw.info = None
    assert refusal(lambda: resolve_instruments(raw, load_point(POINT034, LAB).lab)).code == "E_PIPELINE_NO_INFO"


def test_every_pipeline_refusal_is_structured(point):
    """Un refus est une UserError à code stable E_… : le bord (CLI, API) la traduit à partir de son code."""
    (point / "point034_P_measures.csv").unlink()
    (point / "p.xlsx").write_bytes(b"PK")
    error = refusal(lambda: run_pipeline(point, LAB))
    assert error.code.startswith("E_PIPELINE_") and error.params["folder"] == point
    assert "Only the .csv exports" in str(error)


def test_the_pipeline_does_not_load_the_engine():
    code = textwrap.dedent(f"""
        import sys
        from pathlib import Path
        from molonari_twin.twin.pipeline import run_pipeline
        run_pipeline(Path({str(POINT034)!r}), Path({str(LAB)!r}))
        print(sorted({{n.split(".")[0] for n in sys.modules}} & {{"jax", "jaxlib", "arviz", "PySide6", "fastapi"}}))
    """)
    completed = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300, check=False)
    assert completed.returncode == 0, completed.stderr[-3000:]
    assert completed.stdout.strip() == "[]"
