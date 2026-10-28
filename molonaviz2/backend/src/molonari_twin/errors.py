"""Exceptions et messages partagés par toutes les couches (domaine, stockage, moteur, interface).

Chaque erreur attendue (`UserError`) et chaque problème de saisie (`Issue`) porte un code stable
(`E_…` pour une erreur, `W_…` pour un avertissement), les paramètres de son message et, s'il y a
lieu, le champ concerné. Le texte est le modèle anglais de `MESSAGES`, formaté avec ces paramètres :
le cœur ne traduit rien. Le bord (interface, ligne de commande) traduit à partir du code, le modèle
anglais servant de clé dans le catalogue (`i18n.render`).
"""

from __future__ import annotations

import re

CODE_PATTERN = re.compile(r"^[EW]_[A-Z0-9_]+$")

# Code → modèle anglais. Le modèle est aussi la clé de sa traduction dans i18n/fr.json.
MESSAGES: dict[str, str] = {
    # Repli de l'API Python : `UserError(texte)` sans code (texte affiché tel quel). Interdit dans le code
    # de molonari_twin et de molonaviz2 (tests/test_messages.py), admis pour les appelants externes et les tests.
    "E_TEXT": "{text}",
    # Erreurs qui regroupent des problèmes (paramètre issues)
    "E_SETTINGS_INVALID": "{issues}",
    "E_TWIN_CONFIG_INVALID": "{issues}",
    # Texte d'une erreur système (OSError, ValueError…), non traduit
    "E_SYSTEM_ERROR": "{detail}",
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
    "E_MEASURES_UNIT_UNSUPPORTED": "The column \"{column}\" of {file} is not a voltage in V: only the raw voltage of the pressure sensor is supported.",
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
    # processing
    "E_CLEANED_FILE_COLUMNS": "Missing columns in the cleaned file: {cols}.",
    "E_CLEANED_FILE_NOT_CHRONOLOGICAL": "The dates of the cleaned file are not in chronological order.",
    "E_CLEANUP_MISSING_MEASURES": "Both temperature and pressure measurements are needed.",
    "E_CLEANUP_NO_COMMON_DATE": "The temperature and pressure files have no date in common.",
    "E_CLEANUP_PRESSURE_CALIBRATION": "Pressure sensor \"{name}\" has dU/dH = 0: its calibration is unusable.",
    # twin/pipeline : du brut au résultat
    "E_PIPELINE_FORMAT_UNSUPPORTED": "Unsupported measurement format in {folder}: {files}. Only the .csv exports of the logger software are read; export the measurements again as CSV.",
    "E_PIPELINE_NO_INFO": "No info file (…_info.csv or …_info.txt) in {folder}.",
    "E_PIPELINE_NO_PRESSURE_FILE": "No pressure measurement file found in {folder}.",
    "E_PIPELINE_NO_PRESSURE_SENSOR": "The info file of point \"{point}\" does not name its pressure sensor.",
    "E_PIPELINE_NO_SHAFT": "The info file of point \"{point}\" does not name its shaft.",
    "E_PIPELINE_NO_TEMPERATURE_FILE": "No temperature measurement file found in {folder}.",
    "E_PIPELINE_SIGMA_POSITIVE": "The measurement uncertainty σ of \"{name}\" must be positive (got {sigma:g}).",
    "E_PIPELINE_THERMOMETER_UNKNOWN": "Thermometer \"{name}\" of shaft \"{shaft}\" is not in laboratory \"{lab}\".",
    "E_PIPELINE_TIMEZONE": "Time zones are not supported yet: the dates are read as written in the files, without a time zone.",
    # domain
    "E_LAYERS_COLUMN_DEPTH": "The last layer must end at the column depth ({depth:g} m).",
    "E_LAYERS_ORDER": "Layer bottoms must be strictly increasing and positive.",
    "E_LAYERS_REQUIRED": "At least one layer is required.",
    "E_MCMC_BURN_IN": "The burn-in needs at least 2 iterations.",
    "E_MCMC_C": "c must be between 0 and 1.",
    "E_MCMC_C_STAR": "c* must be positive.",
    "E_MCMC_DELTA": "At least one pair of chains is required.",
    "E_MCMC_FIELD_SAMPLES": "At least 2 field samples are required.",
    "E_MCMC_FIELDS_BOTH": "give either n_field_samples or fields_per_chain, not both.",
    "E_MCMC_FIELDS_PER_CHAIN": "Fields per chain must be between 1 and the iterations kept per chain ({n}).",
    "E_MCMC_GR_THRESHOLD": "The Gelman-Rubin threshold must be greater than 1.",
    "E_MCMC_NB_CHAINS": "DREAM needs at least 2·δ + 1 = {n} chains.",
    "E_MCMC_NB_ITER": "At least one iteration is required.",
    "E_MCMC_N_CR": "At least one crossover value is required.",
    "E_MCMC_QUANTILES": "Quantiles must be strictly between 0 and 1.",
    "E_MCMC_STORE_FIELDS": "Fields stored must be \"none\", \"sensors\" or \"full\".",
    "E_MCMC_SUBSAMPLING": "Sub-sampling steps must be at least 1.",
    "E_MCMC_THIN": "The thinning step must be between 1 and the number of iterations ({n}).",
    "E_NB_CELLS_RANGE": "The number of cells must be between 10 and 2000.",
    "E_PRIORS_PER_LAYER": "Each layer needs its own prior ranges.",
    "E_PRIOR_MISSING": "Missing prior range.",
    "E_PRIOR_ORDER": "Minimum must be lower than maximum.",
    "E_PRIOR_OUT_OF_BOUNDS": "Range outside the physical bounds [{low:g}, {high:g}].",
    "E_SIGMA_T_POSITIVE": "The measurement uncertainty must be positive.",
    "E_VALUE_OUT_OF_RANGE": "Value outside the physical range [{low:g}, {high:g}].",
    # interface
    "E_POINT_NOT_CLEANED": "Clean the measurements first (Cleaning tab).",
    "E_POINT_SENSORS_MISSING": "The sensors of this point are no longer in laboratory \"{lab}\".",
    # engine
    "E_COLUMN_DUPLICATE_TIMES": "Measurement times must be strictly increasing (duplicate dates found).",
    "E_COLUMN_MISSING_COLUMNS": "Cleaned measures are missing columns: {cols}.",
    "E_COLUMN_SENSOR_DEPTHS": "The shaft must define 4 strictly increasing, positive sensor depths.",
    "E_COLUMN_TOO_FEW_TIMES": "At least 3 cleaned measurement times are required to run a model.",
    "E_MODEL_NON_FINITE": "The model produced non-finite values. Check the layer parameters and the measurements.",
    # questions
    "E_QUESTION_CHAINS": "Unknown chain {chain}: this computation has {n} chains, numbered from 0.",
    "E_QUESTION_LEVEL": "The level must be strictly between 0 and 1 (got {level}).",
    "E_THRESHOLDS_INVALID": "Invalid reliability thresholds: {detail}",
    "W_RELIABILITY_CONVERGENCE": "Convergence: largest R̂ {rhat:.3g}, smallest ESS {ess:.0f} (green: R̂ ≤ {rhat_green:g} and ESS ≥ {ess_green:g}; orange: R̂ ≤ {rhat_orange:g} and ESS ≥ {ess_orange:g}).",
    "W_RELIABILITY_COVERAGE": "Coverage: the predictive band covers {coverage:.1%} of the measurements for a level of {level:.1%} (green: within {green:.1%}; orange: within {orange:.1%}).",
    "W_RELIABILITY_DATA": "Data: {gaps:.1%} of the expected measurements are missing (green: at most {green:.1%}; orange: at most {orange:.1%}).",
    "W_RELIABILITY_FIT": "Fit: largest RMSE/σT {ratio:.3g} (green: at most {green:g}; orange: at most {orange:g}).",
    "W_RELIABILITY_IDENTIFIABILITY": "Identifiability: largest posterior over prior standard-deviation ratio {ratio:.3g}, for {parameter} in layer {layer} (green: at most {green:g}; orange: at most {orange:g}).",
    "W_RELIABILITY_PROVENANCE": "Provenance: {count} problem(s) recorded (green: none; otherwise orange).",
    "E_QUESTION_VARIABLE": "Unknown field \"{variable}\" (available: {known}).",
    "E_QUESTION_NO_FIELDS": "No temperature fields were stored with this computation (store_fields: none).",
    "E_QUESTION_NO_TWIN": "Question \"{question}\" is not about a twin: ask it without one.",
    "E_QUESTION_TWIN_REQUIRED": "Question \"{question}\" is about a twin: ask it of one.",
    "E_QUESTION_NOT_AVAILABLE": "Question \"{question}\" does not apply to {ref} ({mode} mode).",
    "E_QUESTION_OUT_OF_RANGE": "{name} = {value} is outside the results ({low} – {high}).",
    "E_QUESTION_PARAMS": "Invalid parameters for question \"{question}\":\n{issues}",
    "E_QUESTION_PERIOD_EMPTY": "No result between {start} and {end}.",
    "E_QUESTION_SIGMA_REQUIRED": "These results do not record the measurement uncertainty σ: give it as a parameter.",
    "E_QUESTION_UNKNOWN": "Unknown question \"{question}\".",
    "E_QUESTION_Z_OR_T": "Give a depth z, a date t, or both.",
    "E_QUESTION_NO_POSTERIOR": "These results have no posterior samples (direct mode, or results written before format 2.0.0).",
    "E_QUESTION_SHAPES": "Incompatible shapes: expected {expected}, found {found}.",
    "E_QUESTION_SIGMA": "The measurement uncertainty σ must be positive or zero (got {sigma}).",
    "E_QUESTION_THIN": "The display thinning must be a whole number of at least 1 (got {thin}).",
    "E_QUESTION_TIMES": "Times must be strictly increasing.",
    "W_LATERAL_NOT_MODELLED": "Lateral heat exchange is not modelled by this version: it is taken as zero.",
    "E_QUESTION_MODELS_INCOMPARABLE": "The two models cannot be compared: they were simulated at different dates ({forward} and {inversion} dates).",
    "W_FIT_NO_MEDIAN": "The posterior median is not available (no stored quantile 0.5): no RMSE of the median.",
    "W_FLUX_NO_SAMPLES": "The flux of the posterior samples was not stored with this computation (results written by an older version): the mean flux has no interval.",
    "W_COVERAGE_NO_POINTS": "No measurement can be compared with the band: coverage is undefined.",
    # rapport public : visibilité des éléments exposés (P2c.2)
    "E_REPORT_DEV_ONLY": "Question \"{question}\" is reserved for the twin developer: a public report never contains it.",
    "E_REPORT_NO_PROJECTION": "Question \"{question}\" has no public projection yet: it cannot enter a public report.",
    "E_REPORT_PRIVATE": "Question \"{question}\" exposes elements the developer has not made public: {elements}.",
    # twin
    "E_NO_CLEANED_MEASURES": "No cleaned measurements yet: apply a cleaning first.",
    "E_STRICT_DIRTY": "The {target} repository has uncommitted changes.",
    "E_STRICT_LOCAL_BUILD": "The code ({version}) is not exactly a published version.",
    "E_STRICT_NO_PROVENANCE": "The provenance of the {target} cannot be read (no git repository).",
    "E_STRICT_PRERELEASE": "Version {version} is a pre-release.",
    "E_STRICT_REFUSED": "Strict mode: the twin was not run.\n{issues}",
    "E_STRICT_UNTAGGED": "The twin repository has no tag on its commit.",
    "E_TWIN_CODE_VERSION": "this twin was defined for \"{declared}\" but the code is \"{current}\". Create a new twin to use this code.",
    "E_TWIN_COLUMN_DEPTH": "must equal the deepest sensor depth + offset ({depth:g} m).",
    "E_TWIN_CODE_VERSION_REMOVED": "code_version is no longer written in twin.yaml since schema 2.0.0: the versions are in twin.cgs.",
    "E_TWIN_DATA_NOT_FOUND": "file not found: {path}",
    "E_TWIN_DATA_PATH": "must be a relative path inside the data repository.",
    "E_TWIN_DATA_ROOT": "the data repository is unknown: set {env} or put the twin in a cgs workspace (molonari.cgs).",
    "E_TWIN_EXPECTED_LIST": "expected a list.",
    "E_TWIN_KEY_FROM_DATA": "since schema 2.0.0, the depths and the offset are read from the laboratory and point files.",
    "E_TWIN_KEY_MOVED": "moved to {new} since schema 2.0.0.",
    "E_TWIN_LEGACY_ID": "expected a twin_id of {length} hexadecimal characters.",
    "E_TWIN_METHOD": "must be one of {methods}.",
    "E_TWIN_DATE_UNREADABLE": "unreadable date \"{value}\".",
    "E_TWIN_DIRECT_KEYS": "direct mode needs {keys}.",
    "E_TWIN_EMPTY": "must not be empty.",
    "E_TWIN_END_BEFORE_START": "must be after {start}.",
    "E_TWIN_EXPECTED_INTEGER": "expected a whole number.",
    "E_TWIN_EXPECTED_MAPPING": "expected a mapping (key: value).",
    "E_TWIN_EXPECTED_NUMBER": "expected a number.",
    "E_TWIN_EXPECTED_BOOLEAN": "expected true or false.",
    "E_TWIN_EXPECTED_TEXT": "expected a text.",
    "E_TWIN_ONE_OF": "must be one of {values}.",
    "E_TWIN_REPORT_DUPLICATE": "same question and same parameters as entry {first}.",
    "E_TWIN_REPORT_EMPTY": "a published twin needs a report: give at least one entry, or leave report out to use the default.",
    "E_TWIN_REPORT_LEVEL": "an mcmc twin must report the water flux with an interval at {level} (water_flux, level {level}).",
    "E_TWIN_REPORT_MINIMUM": "the report must contain the question \"{question}\".",
    "E_TWIN_REPORT_MODEL": "expected the name of a twin folder (e.g. twin_01).",
    "E_TWIN_REPORT_MODEL_SAME": "forward and inversion must be two different twins.",
    "E_TWIN_REPORT_PARAMETERS": "invalid parameters for \"{question}\": {message}",
    "E_TWIN_REPORT_QUANTILES": "must contain {quantiles}: the report asks for an interval at 90 % ({question}).",
    "E_TWIN_REPORT_QUESTION": "unknown question \"{question}\" (known: {known}).",
    "E_TWIN_REPORT_QUESTION_KIND": "\"{question}\" does not answer about a twin: a report only holds questions that read a configuration or results.",
    "E_TWIN_SITE_LABEL_DISTINCT": "must not contain the {other} (the label is public, the {other} is not).",
    "E_TWIN_SITE_LABEL_FORMAT": "expected one line of at most {n} characters.",
    "E_TWIN_VISIBILITY_REVEALED": "the public samples reveal these settings: they stay public while posterior is public.",
    "E_TWIN_VISIBILITY_SEED": "the seed is not a visibility element: it is reserved to the ORX level and cannot be declared public.",
    "E_TWIN_FOLDER_NOT_FOUND": "Twin folder not found: {path}",
    "E_TWIN_FROZEN": "{name} was already run with another configuration (id {previous}). A configuration is frozen: create a new twin to change a parameter.",
    "E_TWIN_INITIAL_CONDITION": "only \"linear\" is supported (linear interpolation of the measurements at t0).",
    "E_TWIN_INVALID_YAML": "Invalid YAML: {error}",
    "E_TWIN_LAYERS": "expected a non-empty list of layers.",
    "E_TWIN_MCMC_ONLY": "only allowed in mcmc mode.",
    "E_TWIN_MINIMUM": "must be at least {v}.",
    "E_TWIN_MODE": "must be \"direct\" or \"mcmc\".",
    "E_TWIN_PER_LAYER": "expected one entry per layer ({n}).",
    "E_TWIN_POSITIVE": "must be positive.",
    "E_TWIN_QUANTILES": "expected a list of numbers between 0 and 1.",
    "E_TWIN_RANGE": "expected [min, max].",
    "E_TWIN_REQUIRED": "required.",
    "E_TWIN_SCHEMA": "unsupported schema \"{value}\" (expected {expected}).",
    "E_TWIN_SENSOR_DEPTHS": "expected 4 strictly increasing positive depths (m).",
    "E_TWIN_TOO_FEW_MEASUREMENTS": "fewer than 3 measurements in the selected period.",
    "E_TWIN_UNKNOWN_KEY": "unknown key (allowed: {keys}).",
    "W_FIELD_UNAVAILABLE": "Results written before format 2.0.0: flux fields, quantiles and chains are unavailable. Create a new twin with this version to get them.",
    "W_PROVENANCE_NOT_A_REPOSITORY": "The {target} is not in a git repository: its provenance is not recorded.",
    "W_PROVENANCE_NO_GIT": "git was not found: the provenance of the code and of the twin is not recorded.",
    "W_PROVENANCE_NO_REMOTE": "The git repository of the {target} has no remote: its URL is not recorded.",
    # tasks, server, api
    "E_PARAM_INVALID": "{message}",
    "E_REQUEST_INVALID": "Invalid request:\n{issues}",
    "E_DIAG_FAILURE": "The diagnostic task failed on purpose at block {block}.",
    "E_SERVER_EXTRA_MISSING": "The server needs optional dependencies: install molonari-twin[server].",
    "E_TASK_FINISHED": "Task {task_id} is already finished ({state}).",
    "E_TASK_INTERRUPTED": "The task was interrupted: its worker stopped before the end.",
    "E_TASK_KIND_UNKNOWN": "Unknown task type \"{kind}\" (available: {kinds}).",
    "E_TASK_NOT_FINISHED": "Task {task_id} is not finished yet ({state}).",
    "E_TASK_NOT_FOUND": "Task {task_id} does not exist.",
    "E_TASK_TIMEOUT": "The task exceeded its time limit ({timeout_s:g} s) and was stopped.",
    "E_TWIN_NOT_COMPUTED": "Twin \"{ref}\" has no results yet: run it first.",
    "E_TWIN_NOT_FOUND": "Twin \"{ref}\" not found in {folder}.",
}


def message_template(code: str) -> str:
    """Modèle anglais du code (KeyError si le code est inconnu)."""
    return MESSAGES[code]


def format_message(code: str, params: dict) -> str:
    """Modèle anglais du code formaté avec ses paramètres ; une liste de problèmes est jointe ligne par ligne."""
    values = {key: "\n".join(map(str, value)) if key == "issues" else value for key, value in params.items()}
    return message_template(code).format(**values)


class UserError(Exception):
    """Erreur attendue, causée par une donnée ou une action de l'utilisateur.

    `UserError(code, field=None, **params)`. `str(exc)` est le message anglais formaté ; le bord le
    traduit à partir de `.code` et `.params`. Le paramètre `issues` (liste d'`Issue`) est réservé aux
    erreurs qui regroupent plusieurs problèmes. `code` et `field` sont positionnels : un paramètre de
    message peut donc s'appeler `field`. Un premier argument qui n'est pas un code connu est pris pour un
    texte libre (code `E_TEXT`).
    """

    def __init__(self, code: str, field: str | None = None, /, **params):
        if code not in MESSAGES and not CODE_PATTERN.match(code):
            code, params = "E_TEXT", {"text": code}
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


class Cancelled(Exception):
    """L'utilisateur a annulé le calcul (ce n'est pas une erreur : rien n'est affiché)."""


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
    def from_error(cls, field: str | None, error: UserError | Issue, prefixed: bool = False) -> Issue:
        """Problème qui reprend le code et les paramètres d'une erreur attendue (ou d'un autre problème)."""
        return cls(field, error.code, prefixed=prefixed, **error.params)

    def __getnewargs_ex__(self):
        return (self.field, self.code), {"prefixed": self.prefixed, **self.params}

    def __repr__(self) -> str:
        return f"Issue({self.field!r}, {self.code!r}, {str(self)!r})"
