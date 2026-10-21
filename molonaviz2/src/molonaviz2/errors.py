"""Exceptions et messages partagés par toutes les couches (domaine, stockage, moteur, interface).

Chaque erreur attendue (`UserError`) et chaque problème de saisie (`Issue`) porte un code stable
(`E_…` pour une erreur, `W_…` pour un avertissement), les paramètres de son message et, s'il y a
lieu, le champ concerné. Le texte est le modèle anglais de `MESSAGES`, formaté avec ces paramètres :
le cœur ne traduit rien. Le bord (interface, ligne de commande) traduit à partir du code, le modèle
anglais servant de clé dans le catalogue (`i18n.render`).
"""

from __future__ import annotations

# Code → modèle anglais. Le modèle est aussi la clé de sa traduction dans i18n/fr.json.
MESSAGES: dict[str, str] = {
    # Temporaire, le temps de la migration : `UserError(texte)` sans code.
    "E_LEGACY": "{text}",
}


def message_template(code: str) -> str:
    """Modèle anglais du code (KeyError si le code est inconnu)."""
    return MESSAGES[code]


def _is_code(text: str) -> bool:
    return len(text) > 2 and text[0] in "EW" and text[1] == "_" and all(c.isupper() or c.isdigit() or c == "_" for c in text[2:])


def format_message(code: str, params: dict) -> str:
    """Modèle anglais du code formaté avec ses paramètres ; une liste de problèmes est jointe ligne par ligne."""
    values = {key: "\n".join(map(str, value)) if key == "issues" else value for key, value in params.items()}
    return message_template(code).format(**values)


class UserError(Exception):
    """Erreur attendue, causée par une donnée ou une action de l'utilisateur.

    `UserError(code, field=None, **params)`. `str(exc)` est le message anglais formaté ; le bord le
    traduit à partir de `.code` et `.params`. Le paramètre `issues` (liste d'`Issue`) est réservé aux
    erreurs qui regroupent plusieurs problèmes.
    """

    def __init__(self, code: str, field: str | None = None, **params):
        if not _is_code(code):  # ancienne forme : texte libre (temporaire)
            code, params = "E_LEGACY", {"text": code}
        self.code = code
        self.field = field
        self.params = params
        super().__init__(format_message(code, params))

    def __reduce__(self):
        return _rebuild_error, (type(self), self.code, self.field, self.params)


def _rebuild_error(cls, code, field, params):
    error = UserError.__new__(cls)
    UserError.__init__(error, code, field, **params)
    return error


class Issue(str):
    """Problème de saisie ou de validation, rattaché à un champ.

    `Issue(field, code, prefixed=False, **params)`. C'est une chaîne : sa valeur est le message anglais
    formaté (`.message`), précédé de `"<field>: "` si `prefixed`. Elle porte `.code`, `.field` et
    `.params`. Attention : `==` compare le texte, pas le code.
    """

    code: str
    field: str | None
    params: dict
    message: str
    prefixed: bool

    def __new__(cls, field: str | None, code: str, prefixed: bool = False, **params):
        message = format_message(code, params)
        issue = super().__new__(cls, f"{field}: {message}" if prefixed and field else message)
        issue.code, issue.field, issue.params, issue.message, issue.prefixed = code, field, params, message, prefixed
        return issue

    def __getnewargs_ex__(self):
        return (self.field, self.code), {"prefixed": self.prefixed, **self.params}

    def __repr__(self) -> str:
        return f"Issue({self.field!r}, {self.code!r}, {str(self)!r})"
