"""BN-03 (raw_series) et BN-04 (processing_preview) : questions sur les fichiers d'un point, sans twin ni JAX."""

import shutil
import typing
from datetime import datetime

import numpy as np
import pytest
from conftest import FIXTURES
from molonari_twin.api import TwinAPI
from molonari_twin.contract import params as p
from molonari_twin.errors import UserError
from molonari_twin.processing.cleanup import METHODS, VARIABLES
from molonari_twin.questions import processing_preview, raw_series
from molonari_twin.twin.pipeline import run_pipeline

POINT = str(FIXTURES / "point034")
LAB = str(FIXTURES / "lab")


@pytest.fixture
def api(tmp_path):
    return TwinAPI(tmp_path)  # ces questions n'ont pas besoin de twin


def refusal(call) -> UserError:
    with pytest.raises(UserError) as error:
        call()
    return error.value


def test_the_cleanup_choices_of_the_contract_are_those_of_the_processing():
    assert p.CLEANUP_VARIABLES == tuple(VARIABLES)
    assert typing.get_args(p.CleanupMethod) == METHODS


# BN-03 raw_series ---------------------------------------------------------------------------------------------------


def test_raw_series_of_point034(api):
    answer = api.ask_global("BN-03", {"point_folder": POINT})
    data = answer.data
    assert data["temperature"].value.shape == (4, 1436) and data["temperature"].unit == "°C"
    assert data["temperature_time"].value.shape == (1436,) and data["temperature_time"].unit == "datetime64[s]"
    assert data["voltage"].value.shape == data["pressure_time"].value.shape == data["pressure_temperature"].value.shape == (1460,)
    assert (data["temperature_dropped"].value, data["pressure_dropped"].value) == (2, 0)
    assert data["voltage"].unit == "V" and data["pressure_temperature"].unit == "°C"
    assert data["temperature_time"].value[0] == np.datetime64("2016-06-27T12:00:00")
    assert data["voltage"].value[0] == 1.26787 and data["temperature"].value[:, 0].tolist() == [13.81, 13.834, 13.858, 13.81]
    assert answer.provenance == {"question": "raw_series", "ref": "BN-03",
                                 "parameters": {"point_folder": POINT, "start": None, "end": None}}
    assert answer.warnings == []


def test_raw_series_period_is_included_at_both_ends():
    answer = raw_series(POINT, datetime(2016, 6, 27, 12, 15), datetime(2016, 6, 27, 12, 45))
    assert answer.data["temperature_time"].value.tolist() == [np.datetime64(f"2016-06-27T12:{m}:00") for m in (15, 30, 45)]
    assert answer.data["temperature"].value.shape == (4, 3) and answer.data["voltage"].value.shape == (3,)
    assert answer.provenance["parameters"] == {"point_folder": POINT, "start": "2016-06-27T12:15:00", "end": "2016-06-27T12:45:00"}


def test_raw_series_of_an_empty_period_is_refused():
    error = refusal(lambda: raw_series(POINT, start=datetime(2030, 1, 1)))
    assert (error.code, error.params) == ("E_QUESTION_PERIOD_EMPTY", {"start": "2030-01-01T00:00:00", "end": "…"})


def test_raw_series_answers_with_the_file_that_exists(api, tmp_path):
    folder = tmp_path / "Point034"
    shutil.copytree(FIXTURES / "point034", folder)
    (folder / "point034_P_measures.csv").unlink()
    answer = api.ask_global("raw_series", {"point_folder": str(folder)})
    assert "temperature" in answer.data and "voltage" not in answer.data
    assert [w.code for w in answer.warnings] == ["E_PIPELINE_NO_PRESSURE_FILE"]


def test_raw_series_refuses_unsupported_formats(api, tmp_path):
    (tmp_path / "t.xlsx").write_bytes(b"PK\x03\x04")
    error = refusal(lambda: api.ask_global("raw_series", {"point_folder": str(tmp_path)}))
    assert error.code == "E_PIPELINE_FORMAT_UNSUPPORTED" and error.params["files"] == "t.xlsx"
    assert refusal(lambda: api.ask_global("raw_series", {"point_folder": str(tmp_path / "nowhere")})).code == "E_FOLDER_NOT_FOUND"


@pytest.mark.parametrize("question", ["raw_series", "processing_preview"])
@pytest.mark.parametrize("params, field", [
    ({"start": "2016-06-27T12:00:00+01:00"}, "start"),  # un fuseau est refusé (P2.1f), pas ignoré
    ({"end": "2016-06-27T12:00:00Z"}, "end"),
    ({"start": "2016-07-01T00:00:00", "end": "2016-06-01T00:00:00"}, None),
    ({"point_folder": ""}, "point_folder"),
])
def test_the_period_and_the_folders_are_validated(api, question, params, field):
    full = {"point_folder": POINT, "lab_folder": LAB, **params}
    if question == "raw_series":
        full.pop("lab_folder")
    error = refusal(lambda: api.ask_global(question, full))
    assert error.code == "E_QUESTION_PARAMS" and [i.field for i in error.params["issues"]] == [field]


# BN-04 processing_preview -------------------------------------------------------------------------------------------


def preview(api, **params):
    return api.ask_global("BN-04", {"point_folder": POINT, "lab_folder": LAB, **params})


def test_preview_without_settings_keeps_every_common_date(api):
    answer = preview(api)
    data = answer.data
    assert data["time"].value.shape == (1436,) and data["temperature"].value.shape == (4, 1436)
    assert data["kept"].value.all() and data["kept_count"].value == data["total_count"].value == 1436
    assert set(data["reason"].value.tolist()) == {""} and data["reasons"].value.size == 0
    assert (data["unmatched_temperature"].value, data["unmatched_pressure"].value) == (0, 24)
    assert (data["sigma_pressure"].value, data["sigma_temperature"].value) == (0.01, 0.03)
    assert (data["pressure"].unit, data["sigma_pressure"].unit) == ("m", "m")
    assert answer.provenance["parameters"]["methods"] == {v: "none" for v in VARIABLES}
    assert answer.provenance["parameters"]["excluded"] == []


def test_preview_is_what_the_pipeline_cleans(api):
    """La prévisualisation montre exactement les mesures que le pipeline garde."""
    answer = preview(api, start="2016-06-27T13:00:00", end="2016-07-12T10:15:00", methods={"t1": "iqr"},
                     excluded=["2016-06-28T00:00:00"])
    data = answer.data
    from molonari_twin.processing.cleanup import CleanupSettings

    settings = CleanupSettings(start=datetime(2016, 6, 27, 13), end=datetime(2016, 7, 12, 10, 15),
                               methods={**CleanupSettings().methods, "t1": "iqr"}, excluded=[datetime(2016, 6, 28)])
    cleaned = run_pipeline(POINT, LAB, settings).cleaned
    kept = data["kept"].value
    assert kept.sum() == len(cleaned) == data["kept_count"].value
    np.testing.assert_array_equal(data["time"].value[kept], cleaned["date"].to_numpy().astype("datetime64[s]"))
    np.testing.assert_array_equal(data["pressure"].value[kept], cleaned["pressure"].to_numpy())
    np.testing.assert_array_equal(data["temperature"].value[:, kept].T, cleaned[["t1", "t2", "t3", "t4"]].to_numpy())
    counts = dict(zip(data["reasons"].value.tolist(), data["reason_counts"].value.tolist(), strict=True))
    assert counts["outside_window"] == 6 and counts["manual"] == 1 and counts.get("iqr:t1", 0) > 0
    assert sum(counts.values()) == data["total_count"].value - data["kept_count"].value
    assert set(data["reason"].value.tolist()) == {""} | set(counts)
    assert answer.provenance["parameters"]["methods"]["t1"] == "iqr" and answer.provenance["parameters"]["excluded"] == ["2016-06-28T00:00:00"]


def test_preview_saves_nothing_and_leaves_the_folders_alone(api, tmp_path):
    point = tmp_path / "Point034"
    shutil.copytree(FIXTURES / "point034", point)
    before = sorted(path.name for path in point.iterdir())
    processing_preview(point, LAB, methods={"pressure": "zscore"})
    assert sorted(path.name for path in point.iterdir()) == before


@pytest.mark.parametrize("params, field", [
    ({"methods": {"humidity": "iqr"}}, "methods"),
    ({"methods": {"t1": "median"}}, "methods.t1"),
    ({"excluded": ["2016-06-28T00:00:00+02:00"]}, "excluded"),
    ({"lab_folder": ""}, "lab_folder"),
])
def test_preview_validates_its_settings(api, params, field):
    error = refusal(lambda: preview(api, **params))
    assert error.code == "E_QUESTION_PARAMS" and [i.field for i in error.params["issues"]] == [field]


def test_preview_refuses_what_the_pipeline_refuses(api, tmp_path):
    lab = tmp_path / "lab"
    shutil.copytree(FIXTURES / "lab", lab)
    (lab / "pressure_sensors" / "P508.csv").unlink()
    error = refusal(lambda: preview(api, lab_folder=str(lab)))
    assert (error.code, error.params["name"]) == ("E_POINT_PRESSURE_SENSOR_UNKNOWN", "P508")
    assert refusal(lambda: preview(api, lab_folder=str(tmp_path))).code == "E_LAB_FOLDER_INVALID"


def test_the_questions_are_about_no_twin(api):
    for question, params in (("raw_series", {"point_folder": POINT}), ("processing_preview", {"point_folder": POINT, "lab_folder": LAB})):
        assert api.question(question).requires == "none"
        assert refusal(lambda q=question, a=params: api.ask("twin_01", q, a)).code == "E_QUESTION_NO_TWIN"
