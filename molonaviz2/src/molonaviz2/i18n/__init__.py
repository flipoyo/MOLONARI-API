"""Traduction : partie pure, importable sans Qt.

Les chaînes sont écrites en anglais et traduites à partir d'un catalogue JSON par langue
(`fr.json`, {texte anglais: traduction}). Ce module ne dépend pas de Qt : le cœur peut l'importer
(`N_`, `LANGUAGES`), et la ligne de commande traduit avec `translate`.

Pour l'interface, `t()` et `install_language()` passent par le mécanisme standard de Qt
(QTranslator, dans `i18n/qt.py`), si bien que les boîtes de dialogue de Qt et nos propres textes
suivent le même chemin. Ils importent Qt au moment de l'appel seulement.
"""

from __future__ import annotations

import json
from functools import cache
from importlib import resources

from ..errors import Issue, UserError, message_template

CONTEXT = "molonaviz2"
LANGUAGES = {"fr": "Français", "en": "English"}
DEFAULT_LANGUAGE = "fr"


def N_(text: str) -> str:  # noqa: N802 (convention gettext)
    """Marque une chaîne à traduire plus tard (constantes de module) ; la retourne telle quelle."""
    return text


@cache
def load_catalog(language: str) -> dict[str, str]:
    """Charge le catalogue d'une langue (à ne pas modifier). L'anglais est la langue source : catalogue vide."""
    if language == "en":
        return {}
    data = resources.files(__package__).joinpath(f"{language}.json").read_text(encoding="utf-8")
    return json.loads(data)


def translate(text: str, language: str) -> str:
    """Traduit `text` dans `language` sans Qt (retourne `text` s'il n'a pas de traduction)."""
    if language not in LANGUAGES:
        language = DEFAULT_LANGUAGE
    return load_catalog(language).get(text, text)


def render(obj, tr) -> str:
    """Texte traduit d'une erreur ou d'un problème du cœur, à partir de son code et de ses paramètres.

    `tr` traduit un texte anglais : `t` dans l'interface, `lambda s: translate(s, langue)` en ligne de
    commande. Avec la fonction identité, le résultat est `str(obj)`. Gère `UserError` (y compris celles
    qui regroupent des `issues`), `Issue`, et tout objet qui sait se rendre lui-même (`localized(tr)`,
    par exemple un problème d'import rattaché à un fichier). Un autre objet est rendu par `str`.
    """
    if hasattr(obj, "localized"):
        return obj.localized(tr)
    if isinstance(obj, UserError | Issue):
        params = dict(obj.params)
        if "issues" in params:
            params["issues"] = "\n".join(render(issue, tr) for issue in params["issues"])
        message = tr(message_template(obj.code)).format(**params)
        if isinstance(obj, Issue) and obj.prefixed and obj.field:
            return f"{obj.field}: {message}"
        return message
    return str(obj)


def t(text: str) -> str:
    """Interface : traduit `text` dans la langue installée dans l'application Qt."""
    from .qt import t as qt_t

    return qt_t(text)


def install_language(app, language: str) -> None:
    """Interface : installe (ou remplace) la traduction active de l'application Qt."""
    from .qt import install_language as qt_install_language

    qt_install_language(app, language)
