"""La référence des questions (docs/API_V1.md) est générée à partir du catalogue, et à jour."""

import importlib.util
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("gen_api_doc", WORKSPACE / "tools" / "gen_api_doc.py")
gen_api_doc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gen_api_doc)


def test_api_reference_is_up_to_date():
    assert gen_api_doc.OUTPUT.read_text(encoding="utf-8") == gen_api_doc.render(), (
        "docs/API_V1.md n'est pas à jour : python tools/gen_api_doc.py (depuis molonaviz2/)")


def test_every_question_is_documented():
    from molonari_twin.api import CATALOG

    text = gen_api_doc.OUTPUT.read_text(encoding="utf-8")
    for question in CATALOG:
        assert f"### `{question.id}` ({question.ref})" in text
        for name in question.params.model_fields:
            assert f"| `{name}` |" in text, f"{question.id}.{name}"


def test_tables_have_a_constant_number_of_columns():
    for line in gen_api_doc.OUTPUT.read_text(encoding="utf-8").splitlines():
        if line.startswith("| `") and "Paramètre" not in line:
            assert line.count("|") in (3, 6), line
