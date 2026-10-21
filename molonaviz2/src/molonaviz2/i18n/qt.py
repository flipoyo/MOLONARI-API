"""Traduction de l'interface par Qt : seule partie d'i18n qui importe PySide6 (interdite au cœur)."""

from __future__ import annotations

from PySide6 import QtCore

from . import CONTEXT, DEFAULT_LANGUAGE, LANGUAGES, load_catalog


def t(text: str) -> str:
    """Traduit `text` dans la langue courante (retourne `text` s'il n'a pas de traduction)."""
    return QtCore.QCoreApplication.translate(CONTEXT, text)


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
