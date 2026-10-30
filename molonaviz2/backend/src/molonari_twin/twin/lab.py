"""Virtual lab (`lab.yaml`) : le laboratoire, les points qu'il couvre, leur nettoyage et la version du code.

Un virtual lab dit, une fois, comment les bruts d'un ou plusieurs points deviennent des mesures nettoyées.

    schema: molonari-lab/1.0.0
    name: lab_point034
    laboratory: Laboratory         # dossier du laboratoire, relatif à la racine des données
    code_version: 0.8.0-alpha.1    # facultatif : une différence avec le code est un avertissement, pas un refus
    points:
      Point034:
        folder: SamplingPoints/Point034
        processing: {start: 2016-06-27 13:00, end: 2016-07-12 10:15}   # mêmes clés que `processing` d'un twin 2.0.0

Mêmes règles que `twin.yaml` : une clé inconnue est refusée (une faute de frappe ne doit pas être ignorée en silence), les
chemins sont relatifs à la racine des données (`config.data_root` : `MOLONARI_DATA`, sinon le `data/` de l'espace cgs qui
contient le lab) et n'en sortent pas, et tous les problèmes sont rapportés ensemble (`LabConfigError`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .. import __version__
from ..errors import Issue, UserError
from ..processing.cleanup import CleanupSettings
from .config import (
    DATA_ROOT_ENV,
    _Checker,
    _data_folder,
    _processing,
    data_root,
)

LAB_SCHEMA = "molonari-lab/1.0.0"
LAB_NAME = "lab.yaml"
LAB_KEYS = {"schema", "name", "laboratory", "code_version", "points"}
LAB_REQUIRED = ("schema", "name", "laboratory", "points")
POINT_KEYS = {"folder", "processing"}


class LabConfigError(UserError):
    """`lab.yaml` invalide. `issues` liste les problèmes (`Issue`), chacun préfixé par le chemin de la clé."""

    def __init__(self, issues: list[Issue]):
        self.issues = issues
        super().__init__("E_LAB_CONFIG_INVALID", issues=issues)

    def __reduce__(self):
        return type(self), (self.issues,)


@dataclass
class PointSpec:
    """Un point couvert par le lab : son dossier (relatif à la racine des données) et son nettoyage."""

    folder: str
    processing: CleanupSettings


@dataclass
class LabSpec:
    path: Path  # chemin du lab.yaml
    name: str
    laboratory: str  # dossier du laboratoire, relatif à `root`
    code_version: str | None  # version du code déclarée, None si le lab n'en déclare pas
    points: dict[str, PointSpec]
    root: Path  # racine des données
    warnings: list[Issue] = field(default_factory=list)

    @property
    def laboratory_folder(self) -> Path:
        return self.root / self.laboratory


def lab_path(path: str | Path) -> Path:
    """Accepte le dossier du lab ou directement son lab.yaml."""
    path = Path(path)
    return path / LAB_NAME if path.is_dir() else path


def read_lab(path: str | Path, data_folder: str | Path | None = None) -> LabSpec:
    """Lit et valide un `lab.yaml`. Lève LabConfigError avec la liste complète des problèmes.

    `data_folder` : racine des données, à défaut de MOLONARI_DATA et de l'espace de travail cgs (voir `data_root`)."""
    path = lab_path(path)
    if not path.is_file():
        raise LabConfigError([Issue(None, "E_FILE_NOT_FOUND", path=path)])
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise LabConfigError([Issue(None, "E_TWIN_INVALID_YAML", error=exc)]) from None
    check = _Checker()
    doc = check.mapping(raw, "", LAB_KEYS, LAB_REQUIRED)
    if check.issues and not doc:
        raise LabConfigError(check.issues)

    if "schema" in doc and doc["schema"] != LAB_SCHEMA:
        check.add("schema", "E_TWIN_SCHEMA", value=doc["schema"], expected=LAB_SCHEMA)
    name = str(doc.get("name", "")).strip()
    if "name" in doc and not name:
        check.add("name", "E_TWIN_EMPTY")

    root = Path(data_folder).resolve() if data_folder is not None else data_root(path.parent)
    laboratory = ""
    if root is None:
        check.add("laboratory", "E_LAB_DATA_ROOT", env=DATA_ROOT_ENV)
    elif "laboratory" in doc and _data_folder(check, root, doc["laboratory"], "laboratory") is not None:
        laboratory = Path(doc["laboratory"]).as_posix()

    declared = doc.get("code_version")
    declared = str(declared) if declared is not None else None

    points: dict[str, PointSpec] = {}
    raw_points = doc.get("points", {})
    if "points" in doc and not isinstance(raw_points, dict):
        check.add("points", "E_TWIN_EXPECTED_MAPPING")
        raw_points = {}
    elif "points" in doc and not raw_points:
        check.add("points", "E_TWIN_REQUIRED")  # un lab sans point ne couvre rien
    for key, entry in raw_points.items():
        key = str(key)
        where = f"points.{key}"
        if not key.strip() or key in (".", "..") or "/" in key or "\\" in key:
            check.add(where, "E_TWIN_DATA_PATH")  # le nom du point est un nom de dossier de processed_data
            continue
        section = check.mapping(entry, where, POINT_KEYS, ("folder",))
        folder = ""
        if root is not None and "folder" in section and _data_folder(check, root, section["folder"], f"{where}.folder") is not None:
            folder = Path(section["folder"]).as_posix()
        processing = _processing(check, section.get("processing"), f"{where}.processing")
        if folder:
            points[key] = PointSpec(folder, processing)

    if check.issues:
        raise LabConfigError(check.issues)

    warnings = []
    if declared is not None and declared != __version__:
        warnings.append(Issue(None, "W_LAB_CODE_VERSION", file=path.name, declared=declared, current=__version__))
    return LabSpec(path, name, laboratory, declared, points, root, warnings)
