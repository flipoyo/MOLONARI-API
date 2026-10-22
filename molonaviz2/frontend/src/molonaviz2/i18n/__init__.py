"""Traduction de l'interface.

Deux catalogues par langue : celui du backend (messages d'erreur, ligne de commande), lu par sa façade,
et celui de l'application (`fr.json` à côté de ce module, ses propres textes). `load_catalog` les
réunit : l'interface affiche aussi les erreurs du backend, et Qt n'installe qu'un seul traducteur.

`t()` et `install_language()` passent par le mécanisme standard de Qt (QTranslator, dans `i18n/qt.py`),
si bien que les boîtes de dialogue de Qt et nos propres textes suivent le même chemin. Ils importent
Qt au moment de l'appel seulement : `molonaviz2 twin …` n'en a pas besoin.
"""

from __future__ import annotations

import json
from functools import cache
from importlib import resources

from molonari_twin import facade
from molonari_twin.facade import DEFAULT_LANGUAGE, LANGUAGES, N_, render

CONTEXT = "molonaviz2"

__all__ = ["CONTEXT", "DEFAULT_LANGUAGE", "LANGUAGES", "N_", "install_language", "load_catalog", "own_catalog", "render", "t", "translate"]


@cache
def own_catalog(language: str) -> dict[str, str]:
    """Catalogue propre à l'application (à ne pas modifier). L'anglais est la langue source : catalogue vide."""
    if language == "en":
        return {}
    data = resources.files(__package__).joinpath(f"{language}.json").read_text(encoding="utf-8")
    return json.loads(data)


@cache
def load_catalog(language: str) -> dict[str, str]:
    """Catalogue réuni de l'interface : celui du backend, puis celui de l'application (à ne pas modifier)."""
    return {**facade.load_catalog(language), **own_catalog(language)}


def translate(text: str, language: str) -> str:
    """Traduit `text` dans `language` sans Qt, avec le catalogue réuni (retourne `text` s'il n'a pas de traduction)."""
    if language not in LANGUAGES:
        language = DEFAULT_LANGUAGE
    return load_catalog(language).get(text, text)


def t(text: str) -> str:
    """Traduit `text` dans la langue installée dans l'application Qt."""
    from .qt import t as qt_t

    return qt_t(text)


def install_language(app, language: str) -> None:
    """Installe (ou remplace) la traduction active de l'application Qt."""
    from .qt import install_language as qt_install_language

    qt_install_language(app, language)
