"""Catalogue français du backend : exactement ses chaînes (messages d'erreur, ligne de commande, N_)."""

import importlib.util
import json
import re
from pathlib import Path

from molonari_twin.i18n import LANGUAGES, load_catalog

WORKSPACE = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("extract_strings", WORKSPACE / "tools" / "extract_strings.py")
extract_strings = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(extract_strings)

PLACEHOLDER = re.compile(r"\{[^{}]*\}")
PATH = Path(__file__).resolve().parents[1] / "src" / "molonari_twin" / "i18n" / "fr.json"


def test_every_backend_string_has_a_french_translation():
    missing = sorted(set(extract_strings.extract("backend")) - set(load_catalog("fr")))
    assert missing == [], f"{len(missing)} chaîne(s) sans traduction : {missing[:10]}"


def test_no_unused_translations():
    unused = sorted(set(load_catalog("fr")) - set(extract_strings.extract("backend")))
    assert unused == [], unused


def test_translations_keep_placeholders():
    for source, translation in load_catalog("fr").items():
        assert sorted(PLACEHOLDER.findall(source)) == sorted(PLACEHOLDER.findall(translation)), source
        assert translation.strip(), source


def test_languages_and_catalog_file():
    assert set(LANGUAGES) == {"fr", "en"} and load_catalog("en") == {}
    assert isinstance(json.loads(PATH.read_text(encoding="utf-8")), dict)
