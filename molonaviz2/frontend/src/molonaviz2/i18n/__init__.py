"""Traduction de l'interface.

Le catalogue (`fr.json`) et la traduction pure viennent du backend, par sa façade. Ce module y ajoute
la traduction par Qt : `t()` et `install_language()` passent par le mécanisme standard de Qt
(QTranslator, dans `i18n/qt.py`), si bien que les boîtes de dialogue de Qt et nos propres textes
suivent le même chemin. Ils importent Qt au moment de l'appel seulement : `molonaviz2 twin …` n'en a
pas besoin.
"""

from __future__ import annotations

from molonari_twin.facade import (
    DEFAULT_LANGUAGE,
    LANGUAGES,
    N_,
    load_catalog,
    render,
    translate,
)

CONTEXT = "molonaviz2"

__all__ = ["CONTEXT", "DEFAULT_LANGUAGE", "LANGUAGES", "N_", "install_language", "load_catalog", "render", "t", "translate"]


def t(text: str) -> str:
    """Traduit `text` dans la langue installée dans l'application Qt."""
    from .qt import t as qt_t

    return qt_t(text)


def install_language(app, language: str) -> None:
    """Installe (ou remplace) la traduction active de l'application Qt."""
    from .qt import install_language as qt_install_language

    qt_install_language(app, language)
