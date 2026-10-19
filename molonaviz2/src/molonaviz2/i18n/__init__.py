"""Traduction de l'interface.

Les chaînes sont écrites en anglais dans le code via `t("...")` et traduites à
l'exécution à partir d'un catalogue JSON par langue (`fr.json`). Le catalogue est
branché sur le mécanisme standard de Qt (QTranslator), donc les boîtes de dialogue
de Qt et nos propres textes passent par le même chemin.
"""

from __future__ import annotations

import json
from importlib import resources

from PySide6 import QtCore

CONTEXT = "molonaviz2"
LANGUAGES = {"fr": "Français", "en": "English"}
DEFAULT_LANGUAGE = "fr"


def t(text: str) -> str:
    """Traduit `text` dans la langue courante (retourne `text` s'il n'a pas de traduction)."""
    return QtCore.QCoreApplication.translate(CONTEXT, text)


def N_(text: str) -> str:  # noqa: N802 (convention gettext)
    """Marque une chaîne à traduire plus tard (constantes de module) ; la retourne telle quelle."""
    return text


def load_catalog(language: str) -> dict[str, str]:
    """Charge le catalogue d'une langue. L'anglais est la langue source : catalogue vide."""
    if language == "en":
        return {}
    data = resources.files(__package__).joinpath(f"{language}.json").read_text(encoding="utf-8")
    return json.loads(data)


class CatalogTranslator(QtCore.QTranslator):
    """QTranslator alimenté par un dictionnaire {texte source: traduction}."""

    def __init__(self, catalog: dict[str, str], parent: QtCore.QObject | None = None):
        super().__init__(parent)
        self._catalog = catalog

    def translate(self, context, source_text, disambiguation=None, n=-1):  # noqa: N802 (API Qt)
        # None : Qt retombe sur le texte source.
        return self._catalog.get(source_text)

    def isEmpty(self) -> bool:  # noqa: N802 (API Qt)
        return not self._catalog


_installed: CatalogTranslator | None = None


def install_language(app: QtCore.QCoreApplication, language: str) -> None:
    """Installe (ou remplace) la traduction active de l'application."""
    global _installed
    if language not in LANGUAGES:
        language = DEFAULT_LANGUAGE
    if _installed is not None:
        app.removeTranslator(_installed)
    _installed = CatalogTranslator(load_catalog(language), app)
    app.installTranslator(_installed)
