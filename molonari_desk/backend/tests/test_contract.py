"""Contrat de l'API /v1 : version, codec des réponses (JSON et npz), et légèreté du contrat."""

import json
import math
import re
import subprocess
import sys

import numpy as np
import pytest
from molonari_twin.contract import API_VERSION
from molonari_twin.contract.codec import (
    answer_from_json,
    answer_from_npz,
    answer_to_json,
    answer_to_npz,
)
from molonari_twin.contract.models import AnswerOut
from molonari_twin.errors import Issue
from molonari_twin.questions.answer import Answer, Quantity

SEMVER = re.compile(r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)(-[0-9A-Za-z.-]+)?$")


def sample_answer():
    times = np.array(["2016-06-27T12:00", "2016-06-27T12:15", "NaT"], dtype="datetime64[s]")
    values = np.array([[1.5, np.nan, np.inf], [-np.inf, 0.0, 2.25]])
    return Answer(
        data={
            "temperature": Quantity(values, "°C", (values - 1, values + 1)),
            "time": Quantity(times, "datetime64[s]"),
            "counts": Quantity(np.arange(24, dtype=np.int64).reshape(2, 3, 4), "1"),
            "flags": Quantity(np.array([True, False]), "1"),
            "names": Quantity(np.array(["clay", "sand"]), "1"),
            "level": Quantity(0.9, "1"),
            "points": Quantity(np.int64(12), "1"),
            "coverage": Quantity(math.nan, "1"),
            "mode": Quantity("mcmc", "1"),
            "converged": Quantity(np.bool_(True), "1"),
            "start": Quantity(np.datetime64("2016-06-27T12:00:00"), "datetime64[s]"),
            "volume": Quantity(np.float64(3.5), "m³/m²", (np.float64(3.0), np.float64(4.0))),
        },
        provenance={"question": "test", "parameters": {"level": 0.9}, "twin": "twin_03"},
        warnings=[Issue(None, "W_COVERAGE_NO_POINTS")],
    )


def assert_same(decoded: Answer, original: Answer):
    assert decoded.provenance == original.provenance
    assert [(w.code, w.field, w.params) for w in decoded.warnings] == [(w.code, w.field, w.params) for w in original.warnings]
    assert decoded.data.keys() == original.data.keys()
    for name, quantity in original.data.items():
        other = decoded.data[name]
        assert other.unit == quantity.unit, name
        pairs = [(other.value, quantity.value)]
        if quantity.interval is not None:
            pairs += list(zip(other.interval, quantity.interval, strict=True))
        else:
            assert other.interval is None, name
        for got, expected in pairs:
            if isinstance(expected, np.ndarray):
                assert isinstance(got, np.ndarray), name
                assert got.dtype == expected.dtype and got.shape == expected.shape, name
                np.testing.assert_array_equal(got, expected)
            elif isinstance(expected, float) and math.isnan(expected):
                assert isinstance(got, float) and math.isnan(got), name
            else:
                assert got == expected and type(np.asarray(got).item()) is type(np.asarray(expected).item()), name


def test_api_version_is_semver():
    assert SEMVER.match(API_VERSION)


def test_json_round_trip_keeps_values_types_and_shapes():
    original = sample_answer()
    document = answer_to_json(original)
    text = json.dumps(document, allow_nan=False)  # du JSON strict : ni NaN ni Infinity
    assert_same(answer_from_json(text), original)
    assert_same(answer_from_json(document), original)


def test_json_encoding_of_special_values():
    document = answer_to_json(sample_answer())
    temperature = document["data"]["temperature"]["value"]
    assert temperature == {"dtype": "float64", "shape": [2, 3], "data": [1.5, None, "inf", "-inf", 0.0, 2.25]}
    assert document["data"]["time"]["value"]["data"] == ["2016-06-27T12:00:00", "2016-06-27T12:15:00", None]
    assert document["data"]["coverage"]["value"] == {"dtype": "float64", "shape": [], "data": [None]}
    assert document["data"]["level"]["value"] == 0.9
    assert document["data"]["points"]["value"] == 12
    assert document["warnings"] == [{"code": "W_COVERAGE_NO_POINTS", "field": None, "params": {},
                                     "message_en": "No measurement can be compared with the band: coverage is undefined."}]


def test_npz_round_trip_and_arrays_in_the_archive():
    original = sample_answer()
    content = answer_to_npz(original)
    assert_same(answer_from_npz(content), original)
    with np.load(__import__("io").BytesIO(content), allow_pickle=False) as archive:
        assert set(archive.files) == {"meta", "data/temperature/value", "data/temperature/low", "data/temperature/high",
                                      "data/time/value", "data/counts/value", "data/flags/value", "data/names/value"}
        meta = AnswerOut.model_validate_json(str(archive["meta"]))
    assert meta.data["temperature"].value.npz == "data/temperature/value"
    assert meta.data["level"].value == 0.9  # les scalaires restent dans meta


def test_npz_is_much_smaller_than_json_for_large_arrays():
    rng = np.random.default_rng(0)
    answer = Answer(data={"field": Quantity(rng.normal(size=100_000), "°C")}, provenance={})
    assert len(answer_to_npz(answer)) < len(json.dumps(answer_to_json(answer))) / 2


def test_unsupported_values_are_refused():
    with pytest.raises(TypeError):
        answer_to_json(Answer(data={"x": Quantity(object(), "1")}, provenance={}))


def test_contract_does_not_load_the_web_stack_the_engine_or_qt():
    code = (
        "import sys, molonari_twin.contract, molonari_twin.contract.models, molonari_twin.contract.codec\n"
        "print(sorted({n.split('.')[0] for n in sys.modules} & {'fastapi', 'starlette', 'uvicorn', 'jax', 'arviz', 'PySide6'}))"
    )
    completed = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300, check=False)
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "[]"
