"""Configuration d'un twin (`twin.yaml`) : lecture, validation, forme canonique, identifiant.

Un twin est la copie numérique d'un point de mesure : une colonne 1D de sédiments
entre le fond de la rivière et le thermomètre le plus profond. Il est défini
entièrement par son `twin.yaml` (données d'entrée, paramètres du sol, réglages
numériques, graine). Même config + mêmes données + même code donnent le même résultat.

Deux familles de schémas coexistent, lues par le même `load_twin` :
- 1.0.0 (`molonaviz-twin/1`, `molonari-twin/1.0.0`) et 1.1.0 : un fichier de mesures déjà nettoyé
  (`data.measurements`), la graine au niveau supérieur (`seed`, 0 par défaut), `code_version` facultatif.
  Ils sont lus sans changement, et leur `twin_id` ne bouge pas.
- 2.0.0 (`molonari-twin/2.0.0`) : le brut. `data: {lab, point}` désigne le laboratoire et le point dans le dépôt de
  données (chemins relatifs à sa racine), `processing: {start, end, methods, excluded}` le nettoyage. La graine du MCMC
  n'est plus dans `twin.yaml` (refusée, E_TWIN_SEED_ORX) mais dans `orx.yaml` (couche `orx`, `twin/orx.py`) : sans elle le
  twin reste lisible, mais `run_twin` le refuse (E_TWIN_REQUIRED, `orx.mcmc.seed`). Plus de `code_version` (les versions sont dans `twin.cgs`), plus de
  `data.measurements`, de `sensors.depths` ni de `sensors.offset` : profondeurs, décalage et σ viennent des fichiers du
  laboratoire et du point (`twin/pipeline.py`), `sensors.sigma_*` les remplace. `legacy_twin_id` garde l'identifiant
  d'un twin 1.x migré. Quatre clés de publication, absentes des schémas 1.x (P2.3) : `site_label` (libellé public du
  site), `monitoring: {status}`, `visibility` (ce que le développeur accepte de publier) et `report` (ce que le rapport
  public contient). Aucune n'entre dans le `twin_id` : elles ne changent pas le résultat.

Règles :
- `twin_id` = empreinte (SHA-256) de la config canonique. Pour un twin 1.x, elle contient l'empreinte du fichier de
  mesures et la version du code ; pour un twin 2.0.0, l'empreinte des mesures nettoyées (ce qui entre vraiment dans le
  calcul, quel que soit le chemin des données) et la version du code. Les libellés (nom du twin, description, nom
  du point, noms des couches, `legacy_twin_id`) n'y entrent pas : deux twins qui ne diffèrent que par leur nom sont des
  doublons. Un schéma ne change pas l'identifiant d'un twin 1.x : la forme canonique 1.x est figée.
- Les clés inconnues sont refusées : une faute de frappe ne doit pas être ignorée en
  silence, sinon le twin ne serait pas celui que l'on croit. Les clés 1.x retirées du 2.0.0 reçoivent un message qui
  dit où elles sont passées.

Exemple minimal (schéma 1.1.0, mode direct) :

    schema: molonari-twin/1.1.0
    name: twin_01
    point: Point034
    mode: direct
    data:
      measurements: measurements.csv   # date, temp_bed, t1, t2, t3, t4, pressure (°C, m)
    sensors:
      depths: [0.1, 0.2, 0.3, 0.4]
      offset: 0.0
    layers:
      - {name: couche 1, z_low: 0.4, neg_log10_k: 12, porosity: 0.15, lambda_s: 3.4, rhos_cs: 5.0e6}
    numerics:
      nb_cells: 100

Le même point, en schéma 2.0.0 depuis le brut (mode mcmc) :

    schema: molonari-twin/2.0.0
    name: twin_05
    mode: mcmc
    data: {lab: Laboratory, point: SamplingPoints/Point034}
    processing:
      start: 2016-07-12 10:00
      methods: {t1: iqr}
      excluded: [2016-07-13 12:00]
    layers:
      - {name: couche 1, z_low: 0.4}
    mcmc:
      nb_iter: 300

et, dans `orx.yaml` à côté (couche `orx`) :

    mcmc:
      seed: 42
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from itertools import pairwise
from pathlib import Path

import pandas as pd
import yaml

from ..domain.compute import (
    DEFAULT_PRIORS,
    INVERTED_PARAMETERS,
    STORE_FIELDS,
    ForwardConfig,
    InversionConfig,
    LayerSpec,
    PriorRange,
)
from ..domain.models import CLEANED_COLUMNS
from ..engine.column import ColumnInputs
from ..errors import Issue, UserError
from ..processing.cleanup import METHODS, VARIABLES, CleanupResult, CleanupSettings
from . import pipeline
from .orx import read_orx

# Schémas lus, avec leur version. molonaviz-twin/1 est l'ancien nom de 1.0.0.
SCHEMA_2_0_0 = "molonari-twin/2.0.0"
SCHEMAS = {"molonaviz-twin/1": (1, 0, 0), "molonari-twin/1.0.0": (1, 0, 0), "molonari-twin/1.1.0": (1, 1, 0), SCHEMA_2_0_0: (2, 0, 0)}
SCHEMA_WRITTEN = "molonari-twin/1.1.0"  # schéma des twins créés depuis des mesures nettoyées (`create_twin`)
CANONICAL_SCHEMA = "molonaviz-twin/1"  # schéma de la forme canonique 1.x, quel que soit le schéma déclaré : le twin_id ne change pas
CANONICAL_SCHEMA_2 = SCHEMA_2_0_0  # forme canonique d'un twin 2.0.0
CONFIG_NAME = "twin.yaml"
MODES = ("direct", "mcmc")
ID_LENGTH = 16
DATA_ROOT_ENV = "MOLONARI_DATA"  # racine du dépôt de données d'un twin 2.0.0 (sinon : `data/` de l'espace de travail cgs)
WORKSPACE_MARKER = "molonari.cgs"  # fichier à la racine de l'espace de travail cgs, dont `data/` est le dépôt de données

TOP_KEYS = {"schema", "name", "description", "point", "mode", "data", "sensors", "layers", "numerics", "mcmc", "seed", "code_version"}
REQUIRED_TOP = ("schema", "name", "mode", "data", "sensors", "layers")
DATA_KEYS = {"measurements", "start", "end"}
SENSOR_KEYS = {"depths", "offset", "sigma_temperature", "sigma_pressure"}
LAYER_KEYS = {"name", "z_low", "neg_log10_k", "porosity", "lambda_s", "rhos_cs"}
NUMERIC_KEYS = {"nb_cells", "initial_condition", "column_depth"}
MCMC_KEYS_1_1 = {"thin", "fields_per_chain", "store_fields"}  # clés admises à partir du schéma 1.1.0
MCMC_KEYS = {"priors", "nb_chains", "nb_iter", "burn_in_max", "delta", "n_cr", "c", "c_star", "gr_threshold", "quantiles",
             "n_field_samples", "space_step", "time_step"} | MCMC_KEYS_1_1
# Valeurs par défaut des clés 1.1.0 : à ces valeurs, elles n'entrent ni dans la forme canonique ni dans un twin créé.
MCMC_DEFAULTS_1_1 = {"thin": 1, "fields_per_chain": None, "store_fields": "sensors"}

# Schéma 2.0.0 : le brut (`data`), son nettoyage (`processing`) et la graine (`mcmc.seed`).
TOP_KEYS_2 = {"schema", "name", "description", "mode", "data", "processing", "sensors", "layers", "numerics", "mcmc", "legacy_twin_id",
              "site_label", "monitoring", "visibility", "report"}
# Schéma 2.0.0 : le brut (`data`) et son nettoyage (`processing`). La graine est dans `orx.yaml`, pas dans `twin.yaml`.
TOP_KEYS_2 = {"schema", "name", "description", "mode", "data", "processing", "sensors", "layers", "numerics", "mcmc", "legacy_twin_id"}
REQUIRED_TOP_2 = ("schema", "name", "mode", "data", "layers")
DATA_KEYS_2 = {"lab", "point"}
PROCESSING_KEYS = {"start", "end", "methods", "excluded"}
SENSOR_KEYS_2 = {"sigma_temperature", "sigma_pressure"}  # remplacent les σ du laboratoire
MCMC_KEYS_2 = MCMC_KEYS
# Clés des schémas 1.x retirées du 2.0.0, avec ce qui les remplace : ("moved", nouvelle clé), ("from_data", None),
# ("versions", None) ou ("orx", None) pour la graine, réservée à la couche `orx` (`mcmc.seed` l'est aussi : P0.7).
REMOVED_IN_2_0_0 = {
    "seed": ("orx", None),
    "mcmc.seed": ("orx", None),
    "code_version": ("versions", None),
    "point": ("moved", "data.point"),
    "data.measurements": ("moved", "data.lab, data.point"),
    "data.start": ("moved", "processing.start"),
    "data.end": ("moved", "processing.end"),
    "sensors.depths": ("from_data", None),
    "sensors.offset": ("from_data", None),
}
INITIAL_CONDITIONS = ("linear",)

# Clés de publication du schéma 2.0.0 (P2.3) -----------------------------------------------------------------
# Éléments de visibilité (SPEC_RAPPORT_PUBLIC, section 5) et leur défaut : tout privé, sauf la PDF a posteriori et les
# réglages que ses échantillons révèlent de toute façon. La graine n'en est PAS un : elle est réservée au niveau ORX,
# et aucune clé ne peut la déclarer publique. `questions/visibility.py` (`ELEMENTS`, `Visibility.from_config`) en est construit.
VISIBILITY_DEFAULTS = {
    "inversion_settings": False,  # réglages du MCMC qui restent privés : priors, delta, n_cr, c, c_star, seuil R̂, pas, σ
    "inversion_settings_public": True,  # nb_chains, nb_iter, thin, burn-in effectif : les échantillons publics les révèlent
    "forward_parameters": False,  # paramètres du sol d'un twin direct (k, n, λs, ρs·cs par couche)
    "posterior": True,  # PDF a posteriori : résumé, échantillons et données par chaîne
    "states": False,  # champs T, q(z, t), flux de chaleur
    "raw_data": False,  # données brutes
    "processed_data": False,  # données traitées
}
VISIBILITY_ELEMENTS = tuple(VISIBILITY_DEFAULTS)
VISIBILITY_REVEALED = "inversion_settings_public"  # public de fait tant que `posterior` est public (échantillons publics)
MONITORING_STATUS = ("actif", "inactif")
MONITORING_KEYS = {"status"}
SITE_LABEL_MAX = 120
REPORT_KEYS = {"models", "entries"}
REPORT_MODEL_KEYS = ("forward", "inversion")
REPORT_ENTRY_KEYS = {"question", "parameters"}
# Rapport par défaut (twin sans clé `report`) : le flux d'eau à 90 % et la qualité de l'ajustement (HN-10, HN-40).
DEFAULT_REPORT_ENTRIES = (("water_flux", {"level": 0.9}), ("fit_quality", {}))
REPORT_REQUIRED = ("water_flux", "fit_quality")  # informations obligatoires du rapport (SPEC_RAPPORT_PUBLIC, section 4)
REPORT_INTERVAL_LEVEL = 0.9  # un intervalle à 90 % demande les quantiles 0,05 et 0,95 de mcmc.quantiles


@dataclass(frozen=True)
class ReportEntry:
    """Une entrée de `report.entries` : une question du catalogue et ses paramètres (tels que déclarés)."""

    question: str
    parameters: dict = field(default_factory=dict)


@dataclass(frozen=True)
class ReportSpec:
    """Clé `report` : les deux instances comparées (noms de dossiers de twins) et la grille fixe des entrées.

    `declared` vaut False quand le twin n'a pas de clé `report` : c'est alors le rapport par défaut."""

    entries: tuple[ReportEntry, ...]
    forward: str | None = None
    inversion: str | None = None
    declared: bool = False


def default_report() -> ReportSpec:
    return ReportSpec(entries=tuple(ReportEntry(question, dict(parameters)) for question, parameters in DEFAULT_REPORT_ENTRIES))


class TwinConfigError(UserError):
    """Config invalide. `issues` liste les problèmes (`Issue`), chacun préfixé par le chemin de la clé."""

    def __init__(self, issues: list[Issue]):
        self.issues = issues
        super().__init__("E_TWIN_CONFIG_INVALID", issues=issues)

    def __reduce__(self):
        return type(self), (self.issues,)


@dataclass
class TwinConfig:
    path: Path  # chemin du twin.yaml
    name: str
    mode: str
    measurements: Path | None  # fichier de mesures nettoyées (1.x) ; None en 2.0.0 (brut + nettoyage)
    depths: tuple[float, float, float, float]  # en 2.0.0 : profondeurs de la tige du point, lues dans le laboratoire
    offset: float = 0.0  # en 2.0.0 : décalage du fichier info du point
    sigma_temperature: float = 0.1  # en 2.0.0 : Sigma_Meas_T du thermomètre de la tige, ou la valeur de `sensors`
    sigma_pressure: float = 0.01  # en 2.0.0 : Sigma_Meas_P du capteur de pression, ou la valeur de `sensors`
    layers: list[LayerSpec] = field(default_factory=list)
    nb_cells: int = 100
    initial_condition: str = "linear"
    start: datetime | None = None  # en 2.0.0 : processing.start
    end: datetime | None = None  # en 2.0.0 : processing.end
    mcmc: InversionConfig | None = None
    # 1.x : `seed` de twin.yaml (0 par défaut). 2.0.0 : `mcmc.seed` de orx.yaml, obligatoire pour calculer un twin mcmc ; None
    # sans elle (le twin reste lisible), 0 en mode direct qui n'a pas de graine.
    seed: int | None = 0
    description: str = ""
    point: str = ""  # libellé ; en 2.0.0 : nom du point lu dans son fichier info (ou son dossier)
    declared_code_version: str | None = None
    # Schéma 2.0.0 -----------------------------------------------------------------------
    schema_version: tuple[int, int, int] = (1, 0, 0)
    data_root: Path | None = None  # racine du dépôt de données
    data_lab: str = ""  # dossier du laboratoire, relatif à data_root
    data_point: str = ""  # dossier du point, relatif à data_root
    processing: CleanupSettings | None = None
    legacy_twin_id: str | None = None  # identifiant 1.x d'un twin migré (étiquette : il n'entre pas dans le twin_id)
    # Clés de publication (P2.3) : 2.0.0 seulement ; les twins 1.x ont les valeurs par défaut. Hors du twin_id.
    site_label: str = ""  # libellé public du site, distinct du nom du point (tension T7)
    monitoring_status: str | None = None  # "actif" | "inactif" ; None : non renseigné
    visibility: dict[str, bool] = field(default_factory=lambda: dict(VISIBILITY_DEFAULTS))  # tous les éléments, défauts compris
    report: ReportSpec = field(default_factory=default_report)
    orx_path: Path | None = None  # orx.yaml de la couche `orx`, s'il existe (2.0.0)
    reference_platform: dict[str, str] | None = None  # plateforme de référence de orx.yaml (étiquette, hors du twin_id)
    # Lecture du brut, une fois par objet (chargée par load_twin).
    _point: pipeline.LoadedPoint | None = field(default=None, init=False, repr=False, compare=False)
    _cleanup: CleanupResult | None = field(default=None, init=False, repr=False, compare=False)

    @property
    def folder(self) -> Path:
        return self.path.parent

    @property
    def raw_data(self) -> bool:
        """Vrai pour un twin 2.0.0, qui part du brut : `data: {lab, point}` et `processing`."""
        return self.schema_version >= (2, 0, 0)

    @property
    def column_depth(self) -> float:
        return float(self.depths[-1] + self.offset)

    @property
    def experimental(self) -> bool:
        return self.mode == "mcmc"

    @property
    def has_seed(self) -> bool:
        """Faux pour un twin 2.0.0 en mode mcmc dont la couche `orx` ne donne pas de graine : il se lit, il ne se calcule pas."""
        return self.mode != "mcmc" or self.seed is not None

    def require_seed(self) -> int:
        """La graine, ou TwinConfigError (E_TWIN_REQUIRED, `orx.mcmc.seed`) : c'est le refus de `run_twin` (P2.2c)."""
        if not self.has_seed:
            raise TwinConfigError([Issue("orx.mcmc.seed", "E_TWIN_REQUIRED", prefixed=True)])
        return int(self.seed or 0)

    # Données ---------------------------------------------------------------------------
    def cleanup(self) -> CleanupResult:
        """Nettoyage du brut d'un twin 2.0.0 (une fois par objet)."""
        assert self._point is not None and self.processing is not None
        if self._cleanup is None:
            self._cleanup = pipeline.clean_point(self._point, self.processing)
        return self._cleanup

    def load_measurements(self) -> pd.DataFrame:
        if self.raw_data:
            return self.cleanup().cleaned
        from ..processing.cleanup import read_cleaned_csv

        frame = read_cleaned_csv(self.measurements)
        if self.start is not None:
            frame = frame[frame["date"] >= pd.Timestamp(self.start)]
        if self.end is not None:
            frame = frame[frame["date"] <= pd.Timestamp(self.end)]
        return frame.reset_index(drop=True)

    def column_inputs(self) -> ColumnInputs:
        if self.raw_data:
            return pipeline.column_inputs(self._point, self.cleanup())
        return ColumnInputs.from_cleaned(self.load_measurements(), self.depths, self.offset, self.sigma_pressure, self.sigma_temperature)

    def forward_config(self) -> ForwardConfig:
        return ForwardConfig(self.layers, self.nb_cells)

    def inversion_config(self) -> InversionConfig:
        assert self.mcmc is not None
        return self.mcmc

    # Identité --------------------------------------------------------------------------
    def canonical(self, code_version: str, with_seed: bool = True) -> dict:
        """Forme canonique : tout ce qui détermine le résultat, et rien d'autre.

        1.x : figée (le `twin_id` d'un twin 1.x ne change pas). 2.0.0 : mêmes sections, avec l'empreinte des mesures
        nettoyées à la place de celle du fichier, la graine dans `mcmc` et toujours la version du code.
        `with_seed=False` : la recette sans la graine, celle que le dev lit (c'est `without_seed(canonical)`, et la
        seule forme possible d'un twin 2.0.0 sans `orx.yaml`). Avec la graine, un twin mcmc 2.0.0 qui n'en a pas est
        refusé (`require_seed`)."""
        if self.raw_data:
            data = {"cleaned_sha256": frame_digest(self.load_measurements())}
        else:
            data = {
                "measurements_sha256": file_digest(self.measurements),
                "start": self.start.isoformat() if self.start else None,
                "end": self.end.isoformat() if self.end else None,
            }
        sensors = {"depths": [float(d) for d in self.depths], "offset": float(self.offset), "sigma_pressure": float(self.sigma_pressure)}
        canonical = {
            "schema": CANONICAL_SCHEMA_2 if self.raw_data else CANONICAL_SCHEMA,
            "mode": self.mode,
            "data": data,
            "sensors": sensors,
            "numerics": {"nb_cells": int(self.nb_cells), "initial_condition": self.initial_condition},
            "code_version": code_version if self.raw_data else (self.declared_code_version or code_version),
        }
        if self.mode == "direct":
            canonical["layers"] = [
                {k: float(getattr(layer, k)) for k in ("z_low", "neg_log10_k", "porosity", "lambda_s", "rhos_cs")}
                for layer in self.layers
            ]
        else:
            sensors["sigma_temperature"] = float(self.sigma_temperature)
            mcmc = self.mcmc
            canonical["layers"] = [{"z_low": float(layer.z_low)} for layer in self.layers]
            canonical["mcmc"] = {
                "priors": [{name: [float(p[name].low), float(p[name].high)] for name in INVERTED_PARAMETERS} for p in mcmc.priors],
                **{key: getattr(mcmc, key) for key in ("nb_chains", "nb_iter", "burn_in_max", "delta", "n_cr", "n_field_samples",
                                                         "space_step", "time_step")},
                **{key: float(getattr(mcmc, key)) for key in ("c", "c_star", "gr_threshold")},
                "quantiles": [float(q) for q in mcmc.quantiles],
                **{key: getattr(mcmc, key) for key, default in MCMC_DEFAULTS_1_1.items() if getattr(mcmc, key) != default},
            }
            if with_seed and self.raw_data:
                canonical["mcmc"]["seed"] = self.require_seed()
            elif with_seed:
                canonical["seed"] = int(self.seed)
        return canonical

    def twin_id(self, code_version: str) -> str:
        text = json.dumps(self.canonical(code_version), sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        return hashlib.sha256(text.encode("ascii")).hexdigest()[:ID_LENGTH]


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def frame_digest(frame: pd.DataFrame) -> str:
    """Empreinte des mesures nettoyées : dates à la seconde, flottants à 17 chiffres (sans perte), sans index."""
    out = frame[CLEANED_COLUMNS].copy()
    out["date"] = pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d %H:%M:%S")
    return hashlib.sha256(out.to_csv(index=False, float_format="%.17g").encode("utf-8")).hexdigest()


def data_root(twin_folder: Path) -> Path | None:
    """Racine du dépôt de données d'un twin 2.0.0 : la variable MOLONARI_DATA, sinon le dossier `data/` de l'espace de
    travail cgs qui contient le twin (celui où se trouve `molonari.cgs`). None si aucun des deux. Le dépôt et la version
    des données sont ceux du manifeste cgs : le twin.yaml n'en dit pas plus que des chemins relatifs à cette racine."""
    declared = os.environ.get(DATA_ROOT_ENV)
    if declared:
        return Path(declared).expanduser().resolve()
    twin_folder = Path(twin_folder).resolve()
    for folder in (twin_folder, *twin_folder.parents):
        if (folder / WORKSPACE_MARKER).is_file():
            return folder / "data"
    return None


def without_seed(canonical: dict) -> dict:
    """Forme canonique sans la graine : celle qui est écrite dans metadata.json (la graine est hors visibilité, P0.7)."""
    out = {key: value for key, value in canonical.items() if key != "seed"}
    if "mcmc" in out:
        out["mcmc"] = {key: value for key, value in out["mcmc"].items() if key != "seed"}
    return out


def config_path(path: str | Path) -> Path:
    """Accepte le dossier twin_xx ou directement son twin.yaml."""
    path = Path(path)
    return path / CONFIG_NAME if path.is_dir() else path


# Validation ---------------------------------------------------------------------------


class _Checker:
    def __init__(self):
        self.issues: list[Issue] = []

    def add(self, where: str, code: str, **params) -> None:
        self.issues.append(Issue(where or None, code, prefixed=True, **params))

    def removed(self, where: str) -> None:
        """Clé d'un schéma 1.x retirée du 2.0.0 : dit où elle est passée."""
        kind, new = REMOVED_IN_2_0_0[where]
        if kind == "moved":
            self.add(where, "E_TWIN_KEY_MOVED", new=new)
        elif kind == "orx":
            self.add(where, "E_TWIN_SEED_ORX")
        elif kind == "from_data":
            self.add(where, "E_TWIN_KEY_FROM_DATA")
        else:
            self.add(where, "E_TWIN_CODE_VERSION_REMOVED")

    def mapping(self, value, where: str, allowed: set[str], required=(), v2: bool = False) -> dict:
        """`v2` : les clés retirées du schéma 2.0.0 ont leur message plutôt que « clé inconnue »."""
        if not isinstance(value, dict):
            self.add(where, "E_TWIN_EXPECTED_MAPPING")
            return {}
        for key in sorted(set(value) - allowed, key=str):
            path = f"{where}.{key}" if where else str(key)
            if v2 and path in REMOVED_IN_2_0_0:
                self.removed(path)
            else:
                self.add(path, "E_TWIN_UNKNOWN_KEY", keys=", ".join(sorted(allowed)))
        for key in required:
            if key not in value:
                self.add(f"{where}.{key}" if where else key, "E_TWIN_REQUIRED")
        return value

    def number(self, value, where: str, integer: bool = False, positive: bool = False, minimum: float | None = None):
        if isinstance(value, bool) or not isinstance(value, int | float):
            if integer:
                self.add(where, "E_TWIN_EXPECTED_INTEGER")
            else:
                self.add(where, "E_TWIN_EXPECTED_NUMBER")
            return None
        if integer and value != int(value):
            self.add(where, "E_TWIN_EXPECTED_INTEGER")
            return None
        if positive and value <= 0:
            self.add(where, "E_TWIN_POSITIVE")
            return None
        if minimum is not None and value < minimum:
            self.add(where, "E_TWIN_MINIMUM", v=minimum)
            return None
        return int(value) if integer else float(value)

    def date(self, value, where: str):
        if value is None:
            return None
        if isinstance(value, datetime):
            return value
        from ..io.dates import parse_date

        try:
            return parse_date(str(value))
        except UserError:
            self.add(where, "E_TWIN_DATE_UNREADABLE", value=value)
            return None


def _data_folder(check: _Checker, root: Path, value, where: str) -> Path | None:
    """Dossier du laboratoire ou du point : un chemin relatif à la racine du dépôt de données, qui n'en sort pas."""
    if not isinstance(value, str) or not value.strip() or Path(value).is_absolute():
        check.add(where, "E_TWIN_DATA_PATH")
        return None
    folder = (root / value).resolve()
    if not folder.is_relative_to(root):
        check.add(where, "E_TWIN_DATA_PATH")
        return None
    if not folder.is_dir():
        check.add(where, "E_FOLDER_NOT_FOUND", path=folder)
        return None
    return folder


def _processing(check: _Checker, raw) -> CleanupSettings:
    """`processing: {start, end, methods, excluded}` d'un twin 2.0.0, qui devient les réglages de `clean`."""
    section = check.mapping(raw if raw is not None else {}, "processing", PROCESSING_KEYS)
    start, end = check.date(section.get("start"), "processing.start"), check.date(section.get("end"), "processing.end")
    if start and end and start >= end:
        check.add("processing.end", "E_TWIN_END_BEFORE_START", start="processing.start")
    methods = {variable: "none" for variable in VARIABLES}
    declared = check.mapping(section.get("methods") if section.get("methods") is not None else {}, "processing.methods", set(VARIABLES))
    for variable in VARIABLES:
        if variable in declared:
            if declared[variable] in METHODS:
                methods[variable] = declared[variable]
            else:
                check.add(f"processing.methods.{variable}", "E_TWIN_METHOD", methods=", ".join(METHODS))
    excluded = []
    raw_excluded = section.get("excluded")
    if raw_excluded is not None and not isinstance(raw_excluded, list):
        check.add("processing.excluded", "E_TWIN_EXPECTED_LIST")
    else:
        for i, value in enumerate(raw_excluded or []):
            date = check.date(value, f"processing.excluded[{i}]")
            if date is not None:
                excluded.append(date)
    return CleanupSettings(start=start, end=end, methods=methods, excluded=sorted(set(excluded)))


def _visibility(check: _Checker, raw) -> dict[str, bool]:
    """Clé `visibility` : un booléen par élément, les défauts pour les autres. La graine n'est pas un élément."""
    visibility = dict(VISIBILITY_DEFAULTS)
    if raw is None:
        return visibility
    if not isinstance(raw, dict):
        check.add("visibility", "E_TWIN_EXPECTED_MAPPING")
        return visibility
    for key in sorted(set(raw) - set(VISIBILITY_ELEMENTS), key=str):
        if key == "seed":
            check.add("visibility.seed", "E_TWIN_VISIBILITY_SEED")
        else:
            check.add(f"visibility.{key}", "E_TWIN_UNKNOWN_KEY", keys=", ".join(VISIBILITY_ELEMENTS))
    for element in VISIBILITY_ELEMENTS:
        if element in raw:
            if isinstance(raw[element], bool):
                visibility[element] = raw[element]
            else:
                check.add(f"visibility.{element}", "E_TWIN_EXPECTED_BOOLEAN")
    # Les échantillons de la PDF publique révèlent ces réglages : ils ne peuvent pas rester privés (SPEC, tension T11).
    if visibility["posterior"] and not visibility[VISIBILITY_REVEALED]:
        check.add(f"visibility.{VISIBILITY_REVEALED}", "E_TWIN_VISIBILITY_REVEALED")
    return visibility


def _monitoring(check: _Checker, raw) -> str | None:
    """Clé `monitoring: {status: actif | inactif}`."""
    if raw is None:
        return None
    section = check.mapping(raw, "monitoring", MONITORING_KEYS, ("status",))
    status = section.get("status")
    if "status" in section and status not in MONITORING_STATUS:
        check.add("monitoring.status", "E_TWIN_ONE_OF", values=", ".join(MONITORING_STATUS))
        return None
    return status


def _site_label(check: _Checker, raw, names: dict[str, str]) -> str:
    """Clé `site_label` : un libellé public d'une ligne, qui ne reprend ni le nom du twin ni celui du point.

    `names` : ce que le libellé ne doit pas contenir (clé du message → valeur). Le nom du point reste privé."""
    if raw is None:
        return ""
    if not isinstance(raw, str):
        check.add("site_label", "E_TWIN_EXPECTED_TEXT")
        return ""
    label = raw.strip()
    if not label:
        check.add("site_label", "E_TWIN_EMPTY")
    elif len(label) > SITE_LABEL_MAX or "\n" in label:
        check.add("site_label", "E_TWIN_SITE_LABEL_FORMAT", n=SITE_LABEL_MAX)
    else:
        for other, value in names.items():
            if value and re.search(rf"(?<!\w){re.escape(value.casefold())}(?!\w)", label.casefold()):
                check.add("site_label", "E_TWIN_SITE_LABEL_DISTINCT", other=other)
                break
    return label


def _report_models(check: _Checker, raw) -> tuple[str | None, str | None]:
    section = check.mapping(raw, "report.models", set(REPORT_MODEL_KEYS))
    names = []
    for key in REPORT_MODEL_KEYS:
        value = section.get(key)
        if key in section and (not isinstance(value, str) or not value.strip() or Path(value).name != value or value.startswith(".")):
            check.add(f"report.models.{key}", "E_TWIN_REPORT_MODEL")
            value = None
        names.append(value)
    if names[0] is not None and names[0] == names[1]:
        check.add("report.models", "E_TWIN_REPORT_MODEL_SAME")
    return names[0], names[1]


def _report_entry(check: _Checker, raw, where: str, catalog: dict):
    """Une entrée `{question, parameters}`. Rend (ReportEntry, niveau de l'intervalle ou None), ou None si elle est invalide."""
    from pydantic import ValidationError

    entry = check.mapping(raw, where, REPORT_ENTRY_KEYS, ("question",))
    name = entry.get("question")
    if "question" not in entry:
        return None
    if not isinstance(name, str) or name not in catalog:
        check.add(f"{where}.question", "E_TWIN_REPORT_QUESTION", question=name, known=", ".join(sorted(catalog)))
        return None
    question = catalog[name]
    if question.requires not in ("config", "results"):
        check.add(f"{where}.question", "E_TWIN_REPORT_QUESTION_KIND", question=name)
        return None
    parameters = entry.get("parameters")
    if parameters is None:
        parameters = {}
    elif not isinstance(parameters, dict):
        check.add(f"{where}.parameters", "E_TWIN_EXPECTED_MAPPING")
        return None
    try:
        validated = question.params.model_validate(parameters)
    except ValidationError as error:
        for problem in error.errors():
            path = ".".join(str(part) for part in problem["loc"])
            check.add(f"{where}.parameters.{path}" if path else f"{where}.parameters", "E_TWIN_REPORT_PARAMETERS", question=name, message=problem["msg"])
        return None
    level = getattr(validated, "level", None)
    return ReportEntry(name, dict(parameters)), (float(level) if level is not None else None)


def _report_quantiles(check: _Checker, asked: list[str], quantiles: tuple[float, ...] | None) -> None:
    """P2.3e : `mcmc.quantiles` contient 0,05 et 0,95 si une entrée (`asked` : ses questions) demande un intervalle à 90 %."""
    if quantiles is None or not asked:
        return
    lower, upper = (1 - REPORT_INTERVAL_LEVEL) / 2, (1 + REPORT_INTERVAL_LEVEL) / 2
    missing = [q for q in (lower, upper) if not any(abs(q - stored) < 1e-9 for stored in quantiles)]
    if missing:
        check.add("mcmc.quantiles", "E_TWIN_REPORT_QUANTILES", quantiles=", ".join(f"{q:g}" for q in missing), question=asked[0])


def _report(check: _Checker, raw, mode, quantiles: tuple[float, ...] | None) -> ReportSpec:
    """Clé `report: {models, entries}`.

    Sans clé : le rapport par défaut. Avec la clé : les questions existent, leurs paramètres sont valides, aucune entrée
    n'est en double, le flux d'eau et la qualité de l'ajustement y sont (en mode mcmc, le flux avec un intervalle à 90 %),
    et `mcmc.quantiles` contient 0,05 et 0,95 dès qu'une entrée demande un intervalle à 90 %. `quantiles` : None hors du
    mode mcmc, ou si `mcmc` est invalide."""
    if raw is None:
        _report_quantiles(check, ["water_flux"], quantiles)  # le rapport par défaut demande le flux d'eau à 90 %
        return default_report()
    from ..api.catalog import CATALOG  # à l'appel : le catalogue importe ce module

    catalog = {question.id: question for question in CATALOG}
    before = len(check.issues)
    section = check.mapping(raw, "report", REPORT_KEYS)
    forward, inversion = _report_models(check, section["models"]) if section.get("models") is not None else (None, None)
    raw_entries = section.get("entries")
    entries: list[ReportEntry] = []
    levels: list[tuple[ReportEntry, float | None]] = []
    if "entries" not in section or raw_entries is None:
        check.add("report.entries", "E_TWIN_REQUIRED")
    elif not isinstance(raw_entries, list):
        check.add("report.entries", "E_TWIN_EXPECTED_LIST")
    elif not raw_entries:
        check.add("report.entries", "E_TWIN_REPORT_EMPTY")
    else:
        seen: dict[tuple[str, str], int] = {}
        for i, raw_entry in enumerate(raw_entries):
            parsed = _report_entry(check, raw_entry, f"report.entries[{i}]", catalog)
            if parsed is None:
                continue
            entry, level = parsed
            key = (entry.question, json.dumps(entry.parameters, sort_keys=True, default=str))
            if key in seen:
                check.add(f"report.entries[{i}]", "E_TWIN_REPORT_DUPLICATE", first=seen[key])
                continue
            seen[key] = i
            entries.append(entry)
            levels.append((entry, level))
        if len(check.issues) == before:
            names = {entry.question for entry in entries}
            for question in REPORT_REQUIRED:
                if question not in names:
                    check.add("report.entries", "E_TWIN_REPORT_MINIMUM", question=question)
            if mode == "mcmc" and not any(e.question == "water_flux" and abs(lv - REPORT_INTERVAL_LEVEL) < 1e-9 for e, lv in levels):
                check.add("report.entries", "E_TWIN_REPORT_LEVEL", level=f"{REPORT_INTERVAL_LEVEL:g}")
            asked = [e.question for e, lv in levels if lv is not None and abs(lv - REPORT_INTERVAL_LEVEL) < 1e-9]
            _report_quantiles(check, asked, quantiles)
    return ReportSpec(entries=tuple(entries), forward=forward, inversion=inversion, declared=True)


def load_twin(path: str | Path, check_data: bool = True, data_folder: str | Path | None = None) -> TwinConfig:
    """Lit et valide un twin. Lève TwinConfigError avec la liste complète des problèmes.

    `data_folder` : racine du dépôt de données d'un twin 2.0.0, à défaut de la variable MOLONARI_DATA et de l'espace de
    travail cgs (voir `data_root`). Sans effet sur un twin 1.x."""
    path = config_path(path)
    if not path.is_file():
        raise TwinConfigError([Issue(None, "E_FILE_NOT_FOUND", path=path)])
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise TwinConfigError([Issue(None, "E_TWIN_INVALID_YAML", error=exc)]) from None
    check = _Checker()
    # Le schéma décide des clés admises : il est lu avant le reste.
    declared_schema = raw.get("schema", "molonaviz-twin/1") if isinstance(raw, dict) else None
    schema_version = SCHEMAS.get(declared_schema) if isinstance(declared_schema, str) else None
    v2 = schema_version is not None and schema_version >= (2, 0, 0)
    doc = check.mapping(raw, "", TOP_KEYS_2 if v2 else TOP_KEYS, REQUIRED_TOP_2 if v2 else REQUIRED_TOP, v2)
    if check.issues and not doc:
        raise TwinConfigError(check.issues)

    if schema_version is None:
        check.add("schema", "E_TWIN_SCHEMA", value=doc.get("schema"), expected=", ".join(SCHEMAS))
    name = str(doc.get("name", "")).strip()
    if "name" in doc and not name:
        check.add("name", "E_TWIN_EMPTY")
    mode = doc.get("mode")
    if "mode" in doc and mode not in MODES:
        check.add("mode", "E_TWIN_MODE")

    measurements = start = end = None
    lab_name = point_name = ""
    root = lab_dir = point_dir = None
    if v2:
        data = check.mapping(doc.get("data", {}), "data", DATA_KEYS_2, ("lab", "point"), True)
        root = Path(data_folder).resolve() if data_folder is not None else data_root(path.parent)
        if root is None and ("lab" in data or "point" in data):
            check.add("data", "E_TWIN_DATA_ROOT", env=DATA_ROOT_ENV)
        elif root is not None:
            lab_dir = _data_folder(check, root, data["lab"], "data.lab") if "lab" in data else None
            point_dir = _data_folder(check, root, data["point"], "data.point") if "point" in data else None
            lab_name = Path(data["lab"]).as_posix() if lab_dir else ""
            point_name = Path(data["point"]).as_posix() if point_dir else ""
    else:
        data = check.mapping(doc.get("data", {}), "data", DATA_KEYS, ("measurements",))
        if "measurements" in data:
            measurements = (path.parent / str(data["measurements"])).resolve()
            if not measurements.is_file():
                check.add("data.measurements", "E_TWIN_DATA_NOT_FOUND", path=measurements)
                measurements = None
        start, end = check.date(data.get("start"), "data.start"), check.date(data.get("end"), "data.end")
        if start and end and start >= end:
            check.add("data.end", "E_TWIN_END_BEFORE_START", start="data.start")
    processing = _processing(check, doc.get("processing")) if v2 else None
    if processing is not None:
        start, end = processing.start, processing.end

    if v2:
        sensors = check.mapping(doc.get("sensors", {}) or {}, "sensors", SENSOR_KEYS_2, (), True)
        sigma_t = check.number(sensors["sigma_temperature"], "sensors.sigma_temperature", positive=True) if "sigma_temperature" in sensors else None
        sigma_p = check.number(sensors["sigma_pressure"], "sensors.sigma_pressure", positive=True) if "sigma_pressure" in sensors else None
        # Profondeurs, décalage et σ viennent du laboratoire et du point (pipeline du brut).
        depths, offset, loaded = None, None, None
        if lab_dir is not None and point_dir is not None and not check.issues:
            try:
                loaded = pipeline.load_point(point_dir, lab_dir, sigma_temperature=sigma_t, sigma_pressure=sigma_p)
            except UserError as exc:
                check.issues.append(Issue.from_error("data", exc, prefixed=True))
        if loaded is not None:
            depths, offset = tuple(float(d) for d in loaded.instruments.shaft.depths), loaded.offset
            sigma_t, sigma_p = loaded.instruments.sigma_temperature, loaded.instruments.sigma_pressure
            if len(depths) != 4 or depths[0] <= 0 or any(b <= a for a, b in pairwise(depths)):
                check.add("data.point", "E_TWIN_SENSOR_DEPTHS")
                depths = None
    else:
        sensors = check.mapping(doc.get("sensors", {}), "sensors", SENSOR_KEYS, ("depths",))
        depths = sensors.get("depths")
        if "depths" in sensors:
            values = [check.number(d, "sensors.depths", positive=True) for d in depths] if isinstance(depths, list) else []
            if len(values) != 4 or None in values or any(b <= a for a, b in pairwise(values)):
                check.add("sensors.depths", "E_TWIN_SENSOR_DEPTHS")
                depths = None
            else:
                depths = tuple(values)
        offset = check.number(sensors.get("offset", 0.0), "sensors.offset")
        sigma_t = check.number(sensors.get("sigma_temperature", 0.1), "sensors.sigma_temperature", positive=True)
        sigma_p = check.number(sensors.get("sigma_pressure", 0.01), "sensors.sigma_pressure", positive=True)

    numerics = check.mapping(doc.get("numerics", {}) or {}, "numerics", NUMERIC_KEYS)
    nb_cells = check.number(numerics.get("nb_cells", 100), "numerics.nb_cells", integer=True, positive=True)
    initial = numerics.get("initial_condition", "linear")
    if initial not in INITIAL_CONDITIONS:
        check.add("numerics.initial_condition", "E_TWIN_INITIAL_CONDITION")
    column_depth = None
    if depths is not None and offset is not None:
        column_depth = depths[-1] + offset
        if "column_depth" in numerics:
            declared = check.number(numerics["column_depth"], "numerics.column_depth", positive=True)
            if declared is not None and abs(declared - column_depth) > 1e-9:
                check.add("numerics.column_depth", "E_TWIN_COLUMN_DEPTH", depth=column_depth)

    layers: list[LayerSpec] = []
    raw_layers = doc.get("layers")
    if "layers" in doc:
        if not isinstance(raw_layers, list) or not raw_layers:
            check.add("layers", "E_TWIN_LAYERS")
            raw_layers = []
        for i, raw_layer in enumerate(raw_layers):
            where = f"layers[{i}]"
            layer = check.mapping(raw_layer, where, LAYER_KEYS, ("z_low",))
            values = {"name": str(layer.get("name", f"layer {i + 1}"))}
            for key in ("z_low", "neg_log10_k", "porosity", "lambda_s", "rhos_cs"):
                if key in layer:
                    number = check.number(layer[key], f"{where}.{key}")
                    if number is not None:
                        values[key] = number
            if mode == "direct":
                missing = [k for k in INVERTED_PARAMETERS if k not in layer]
                if missing:
                    check.add(where, "E_TWIN_DIRECT_KEYS", keys=", ".join(missing))
            if "z_low" in values:
                layers.append(LayerSpec(**values))

    # Graine : au niveau supérieur et facultative (0) avant le 2.0.0. Ensuite elle vient de `orx.yaml` (couche `orx`) : son
    # absence ne gêne pas la lecture (`seed` est None), elle est refusée par `run_twin` (`require_seed`).
    orx = None
    if v2:
        orx, orx_issues = read_orx(path.parent)
        check.issues.extend(orx_issues)
        seed = orx.seed if orx is not None else None
        if mode != "mcmc":
            seed = 0
    else:
        seed = check.number(doc.get("seed", 0), "seed", integer=True, minimum=0)
    mcmc = None
    if mode == "mcmc":
        if v2:
            allowed = MCMC_KEYS_2
        else:
            allowed = MCMC_KEYS if schema_version is not None and schema_version >= (1, 1, 0) else MCMC_KEYS - MCMC_KEYS_1_1
        raw_mcmc = check.mapping(doc.get("mcmc", {}) or {}, "mcmc", allowed, v2=v2)
        priors = []
        raw_priors = raw_mcmc.get("priors")
        if raw_priors is None:
            priors = [{k: PriorRange(v.low, v.high) for k, v in DEFAULT_PRIORS.items()} for _ in layers]
        elif not isinstance(raw_priors, list) or len(raw_priors) != len(layers):
            check.add("mcmc.priors", "E_TWIN_PER_LAYER", n=len(layers))
        else:
            for i, raw_prior in enumerate(raw_priors):
                where = f"mcmc.priors[{i}]"
                entry = check.mapping(raw_prior, where, set(INVERTED_PARAMETERS))
                prior = {}
                for param in INVERTED_PARAMETERS:
                    bounds = entry.get(param, [DEFAULT_PRIORS[param].low, DEFAULT_PRIORS[param].high])
                    if not isinstance(bounds, list) or len(bounds) != 2:
                        check.add(f"{where}.{param}", "E_TWIN_RANGE")
                        bounds = [DEFAULT_PRIORS[param].low, DEFAULT_PRIORS[param].high]
                    low, high = (check.number(b, f"{where}.{param}") for b in bounds)
                    prior[param] = PriorRange(low if low is not None else 0.0, high if high is not None else 0.0)
                priors.append(prior)
        settings = {}
        if "n_field_samples" in raw_mcmc and "fields_per_chain" in raw_mcmc:
            check.add("mcmc.fields_per_chain", "E_MCMC_FIELDS_BOTH")
        for key in ("nb_chains", "nb_iter", "burn_in_max", "delta", "n_cr", "n_field_samples", "space_step", "time_step", "thin",
                    "fields_per_chain"):
            if key in raw_mcmc:
                value = check.number(raw_mcmc[key], f"mcmc.{key}", integer=True, positive=True)
                if value is not None:
                    settings[key] = value
        for key in ("c", "c_star", "gr_threshold"):
            if key in raw_mcmc:
                value = check.number(raw_mcmc[key], f"mcmc.{key}", positive=True)
                if value is not None:
                    settings[key] = value
        if "store_fields" in raw_mcmc:
            if raw_mcmc["store_fields"] in STORE_FIELDS:
                settings["store_fields"] = raw_mcmc["store_fields"]
            else:
                check.add("mcmc.store_fields", "E_MCMC_STORE_FIELDS")
        if "quantiles" in raw_mcmc:
            quantiles = raw_mcmc["quantiles"]
            if not isinstance(quantiles, list) or not quantiles:
                check.add("mcmc.quantiles", "E_TWIN_QUANTILES")
            else:
                settings["quantiles"] = tuple(check.number(q, "mcmc.quantiles") or 0.0 for q in quantiles)
        if layers and len(priors) == len(layers):
            mcmc = InversionConfig(layers=layers, priors=priors, nb_cells=nb_cells or 100, sigma_t=sigma_t or 0.1, seed=seed or 0, **settings)
    elif "mcmc" in doc:
        check.add("mcmc", "E_TWIN_MCMC_ONLY")

    # Validation métier commune avec l'application (bornes physiques, couches, DREAM).
    if layers and column_depth is not None and nb_cells:
        engine_config = mcmc if mcmc is not None else ForwardConfig(layers, nb_cells)
        for issue in engine_config.validate(column_depth):
            check.issues.append(Issue.from_error(issue.field, issue, prefixed=True))

    declared_version = None if v2 else doc.get("code_version")
    if declared_version is not None:
        from ..engine import api

        if str(declared_version) != api.engine_version():
            check.add("code_version", "E_TWIN_CODE_VERSION", declared=declared_version, current=api.engine_version())

    legacy_twin_id = doc.get("legacy_twin_id")
    if legacy_twin_id is not None:
        legacy_twin_id = str(legacy_twin_id)
        if len(legacy_twin_id) != ID_LENGTH or any(c not in "0123456789abcdef" for c in legacy_twin_id):
            check.add("legacy_twin_id", "E_TWIN_LEGACY_ID", length=ID_LENGTH)

    # Clés de publication (2.0.0). Le libellé du site ne doit contenir ni le nom du twin ni celui du point (privé).
    site_label, monitoring_status, visibility, report = "", None, dict(VISIBILITY_DEFAULTS), default_report()
    if v2:
        point_label = loaded.raw.name if loaded is not None else Path(str(data.get("point", ""))).name
        site_label = _site_label(check, doc.get("site_label"), {"name": name, "point": point_label})
        monitoring_status = _monitoring(check, doc.get("monitoring"))
        visibility = _visibility(check, doc.get("visibility"))
        report = _report(check, doc.get("report"), mode, mcmc.quantiles if mcmc is not None else None)

    config = None
    if not check.issues:
        config = TwinConfig(
            path=path, name=name, mode=mode, measurements=measurements, depths=depths, offset=offset,
            sigma_temperature=sigma_t, sigma_pressure=sigma_p, layers=layers, nb_cells=nb_cells, initial_condition=initial,
            start=start, end=end, mcmc=mcmc, seed=seed, description=str(doc.get("description", "")),
            point=loaded.raw.name if v2 else str(doc.get("point", "")),
            declared_code_version=str(declared_version) if declared_version is not None else None,
            schema_version=schema_version, data_root=root, data_lab=lab_name, data_point=point_name,
            processing=processing, legacy_twin_id=legacy_twin_id,
            site_label=site_label, monitoring_status=monitoring_status, visibility=visibility, report=report,
            orx_path=orx.path if orx is not None else None, reference_platform=orx.platform if orx is not None else None,
        )
        if v2:
            config._point = loaded
        if check_data:
            try:
                frame = config.load_measurements()
                if len(frame) < 3:
                    check.add("data", "E_TWIN_TOO_FEW_MEASUREMENTS")
            except UserError as exc:
                check.issues.append(Issue.from_error("data" if v2 else "data.measurements", exc, prefixed=True))
            except (ValueError, OSError) as exc:
                check.add("data" if v2 else "data.measurements", "E_SYSTEM_ERROR", detail=str(exc))
    if check.issues:
        raise TwinConfigError(check.issues)
    return config


def write_twin_yaml(path: Path, document: dict) -> None:
    """Écrit un twin.yaml lisible (ordre des clés conservé)."""
    header = "# Config figée d'un twin Molonaviz. Pour changer un paramètre, créez un nouveau twin.\n"
    path.write_text(header + yaml.safe_dump(document, sort_keys=False, allow_unicode=True, default_flow_style=None, width=120), encoding="utf-8")
