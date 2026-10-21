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
    # storage
    "E_CLEANED_EMPTY_VALUES": "Cleaned measures must not contain empty values.",
    "E_COMPUTATION_NOT_FOUND": "This computation no longer exists.",
    "E_FILE_EXISTS": "A file already exists at {path}.",
    "E_FILE_NOT_FOUND": "File not found: {path}",
    "E_LAB_EXISTS": "A laboratory named \"{name}\" already exists.",
    "E_LAB_IN_USE": "This laboratory is used by: {studies}. Delete these studies first.",
    "E_LAB_ITEM_EXISTS": "\"{name}\" already exists in this laboratory.",
    "E_LAB_NAME_REQUIRED": "The laboratory needs a name.",
    "E_LAB_NOT_FOUND": "Laboratory \"{name}\" does not exist.",
    "E_POINT_EXISTS": "A point named \"{name}\" already exists in this study.",
    "E_POINT_NAME_REQUIRED": "The point needs a name.",
    "E_POINT_NOT_FOUND": "Point \"{name}\" does not exist in study \"{study}\".",
    "E_POINT_PRESSURE_SENSOR_UNKNOWN": "Pressure sensor \"{name}\" is not in laboratory \"{lab}\".",
    "E_POINT_SHAFT_UNKNOWN": "Shaft \"{name}\" is not in laboratory \"{lab}\".",
    "E_PROJECT_INVALID": "{name} is not a Molonaviz project.",
    "E_PROJECT_LEGACY_DATABASE": "{name} is a database from the previous Molonaviz. Use File › Import Molonaviz database to convert it.",
    "E_PROJECT_TOO_NEW": "{name} was created by a newer version of Molonaviz. Please update the application.",
    "E_STUDY_EXISTS": "A study named \"{name}\" already exists.",
    "E_STUDY_NAME_REQUIRED": "The study needs a name.",
    "E_STUDY_NOT_FOUND": "Study \"{name}\" does not exist.",
    # io
    "E_DATES_EMPTY": "No dates to read.",
    "E_DATE_FORMAT_UNKNOWN": "Unrecognised date format (first value: \"{value}\").",
    "E_FOLDER_NOT_FOUND": "Folder not found: {path}",
    "E_LAB_FILE_EMPTY_VALUE": "\"{key}\" is empty.",
    "E_LAB_FILE_LINE_COUNT": "Expected {expected} lines, found {found}.",
    "E_LAB_FILE_NOT_A_NUMBER": "\"{key}\" is not a number: \"{value}\".",
    "E_LAB_FOLDER_INVALID": "This folder does not look like a laboratory: it should contain temperature_sensors, pressure_sensors and/or shafts.",
    "E_LEGACY_DATABASE_INVALID": "{name} is not a database from the previous Molonaviz.",
    "E_MEASURES_FILE_EMPTY": "The file {file} is empty.",
    "E_MEASURES_NOT_CHRONOLOGICAL": "Dates in {file} are not in chronological order: line {line} ({value}) comes after line {previous_line} ({previous}). The file was probably altered by a spreadsheet; export it again from the logger software.",
    "E_MEASURES_NO_DATE": "No dated measurement found at the start of {file}.",
    "E_MEASURES_TOO_FEW": "Fewer than 2 valid measurements in {file}.",
    "E_SHAFT_DEPTHS_FORMAT": "Sensor depths must be a list of 4 numbers, e.g. [0.1, 0.2, 0.3, 0.4].",
    "E_SHAFT_DEPTHS_ORDER": "Sensor depths must be 4 strictly increasing positive values (m).",
    "W_LAB_DUPLICATE_NAME": "Duplicate name \"{name}\": file ignored.",
    "W_LAB_UNKNOWN_THERMOMETER": "Refers to thermometer \"{name}\", which is not in this laboratory.",
    "W_LEGACY_LAB_EXISTS": "Laboratory \"{name}\" already exists: kept as is.",
    "W_LEGACY_NOTICE_NOT_FOUND": "{point}: notice not found ({path}).",
    "W_LEGACY_POINT_SKIPPED": "{point} : {error}",
    "W_LEGACY_SCHEME_NOT_FOUND": "{point}: scheme not found ({path}).",
    "W_LEGACY_STUDY_EXISTS": "Study \"{name}\" already exists: skipped.",
    "W_POINT_INFO_UNREADABLE": "Could not read \"{value}\" for {field}.",
    "W_ROW_DUPLICATE_DATE": "duplicate date {date}",
    "W_ROW_EMPTY_VALUES": "empty values (logger event)",
    "W_ROW_MISSING_COLUMNS": "missing columns",
    "W_ROW_NOT_A_NUMBER": "not a number: \"{value}\"",
    "W_ROW_NO_DATE": "no date",
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
    erreurs qui regroupent plusieurs problèmes. `code` et `field` sont positionnels : un paramètre de
    message peut donc s'appeler `field`.
    """

    def __init__(self, code: str, field: str | None = None, /, **params):
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
    `.params`. Attention : `==` compare le texte, pas le code. `field` et `code` sont positionnels.
    """

    code: str
    field: str | None
    params: dict
    message: str
    prefixed: bool

    def __new__(cls, field: str | None, code: str, /, prefixed: bool = False, **params):
        message = format_message(code, params)
        issue = super().__new__(cls, f"{field}: {message}" if prefixed and field else message)
        issue.code, issue.field, issue.params, issue.message, issue.prefixed = code, field, params, message, prefixed
        return issue

    @classmethod
    def from_error(cls, field: str | None, error: UserError, prefixed: bool = False) -> Issue:
        """Problème qui reprend le code et les paramètres d'une erreur attendue."""
        return cls(field, error.code, prefixed=prefixed, **error.params)

    def __getnewargs_ex__(self):
        return (self.field, self.code), {"prefixed": self.prefixed, **self.params}

    def __repr__(self) -> str:
        return f"Issue({self.field!r}, {self.code!r}, {str(self)!r})"
