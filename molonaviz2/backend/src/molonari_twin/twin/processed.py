"""Données traitées d'un point (`processed_data/<point>/`) : les mesures nettoyées et leur manifeste.

Un virtual lab (`twin/lab.py`) construit ce dossier une fois (`molonari-twin lab build`) ; un twin 2.0.0 le lit ensuite
(`data: {processed: processed_data/Point034}`) sans refaire le nettoyage.

    processed_data/Point034/
        measurements.csv     les mesures nettoyées (CLEANED_COLUMNS), écrites sans perte
        manifest.json        d'où elles viennent et ce qui les a produites

Les mesures sont écrites comme `frame_digest` les compte (dates à la seconde, flottants à 17 chiffres significatifs) et
relues avec un analyseur exact : l'aller-retour redonne les mêmes flottants, donc la même empreinte (`cleaned_sha256`),
donc le même `twin_id` que depuis le brut. Le manifeste ne porte aucun horodatage : deux constructions des mêmes
entrées avec le même code donnent les mêmes octets.

Manifeste (`schema: molonari-processed/1.0.0`) : `lab` (nom du lab), `point` (libellé), `code_version` (celle qui a
construit), `raw_files` (empreintes des fichiers bruts du point, noms relatifs au dossier du point), `lab_files`
(empreintes des fichiers du laboratoire utilisés : capteur de pression, tige, thermomètre, noms relatifs au dossier du
laboratoire), `depths`, `offset`, `sigma_pressure`, `sigma_temperature`, `utc_offset`, `processing` (le nettoyage
appliqué), `rows`, `start`, `end` et `cleaned_sha256`.

Ce module ne dépend pas de `config.py` (qui l'importe) : l'empreinte est vérifiée par `config.py`, qui a `frame_digest`.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd

from ..domain.models import CLEANED_COLUMNS
from ..errors import UserError
from ..processing.cleanup import read_cleaned_csv

PROCESSED_SCHEMA = "molonari-processed/1.0.0"
PROCESSED_FOLDER = "processed_data"  # dossier, à la racine des données, qui contient un sous-dossier par point
MEASUREMENTS_NAME = "measurements.csv"
MANIFEST_NAME = "manifest.json"


def cleaned_csv(frame: pd.DataFrame) -> bytes:
    """Mesures nettoyées en CSV sans perte : le texte qu'empreint `config.frame_digest` (dates à la seconde, flottants à
    17 chiffres, sans index), avec « \\n » pour fin de ligne quelle que soit la plate-forme."""
    out = frame[CLEANED_COLUMNS].copy()
    out["date"] = pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d %H:%M:%S")
    return out.to_csv(index=False, float_format="%.17g", lineterminator="\n").encode("utf-8")


def dump_manifest(manifest: dict) -> bytes:
    """JSON déterministe : clés triées, UTF-8, indentation de 2, pas de NaN, saut de ligne final."""
    return (json.dumps(manifest, sort_keys=True, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")


def write_processed(target: Path, manifest: dict, cleaned: pd.DataFrame) -> None:
    """Écrit `measurements.csv` et `manifest.json` dans `target`, qui remplace l'ancien dossier d'un coup (sur le modèle de
    `report.writer.write_folder`) : un échec avant le remplacement laisse l'ancien dossier intact et aucun dossier partiel."""
    target = Path(target)
    partial = target.with_name(target.name + ".partial")
    old = target.with_name(target.name + ".old")
    for leftover in (partial, old):
        if leftover.exists():
            shutil.rmtree(leftover)
    try:
        partial.mkdir(parents=True)
        (partial / MEASUREMENTS_NAME).write_bytes(cleaned_csv(cleaned))
        (partial / MANIFEST_NAME).write_bytes(dump_manifest(manifest))
        if target.exists():
            target.rename(old)
        partial.rename(target)
    except BaseException:
        shutil.rmtree(partial, ignore_errors=True)
        if old.exists() and not target.exists():
            old.rename(target)
        raise
    if old.exists():
        shutil.rmtree(old)


@dataclass(frozen=True)
class Processed:
    """Données traitées lues : le manifeste et les mesures nettoyées."""

    folder: Path
    manifest: dict
    cleaned: pd.DataFrame

    @property
    def lab(self) -> str:
        return self.manifest["lab"]

    @property
    def point(self) -> str:
        return self.manifest["point"]

    @property
    def code_version(self) -> str:
        return self.manifest["code_version"]

    @property
    def depths(self) -> tuple[float, ...]:
        return tuple(float(d) for d in self.manifest["depths"])

    @property
    def offset(self) -> float:
        return float(self.manifest["offset"])

    @property
    def sigma_pressure(self) -> float:
        return float(self.manifest["sigma_pressure"])

    @property
    def sigma_temperature(self) -> float:
        return float(self.manifest["sigma_temperature"])

    @property
    def cleaned_sha256(self) -> str:
        return self.manifest["cleaned_sha256"]

    @property
    def processing(self) -> dict:
        return self.manifest["processing"]

    @property
    def window(self) -> tuple[datetime | None, datetime | None]:
        """Début et fin du nettoyage (`processing.start`, `processing.end`), None quand il n'y en avait pas."""
        return tuple(datetime.fromisoformat(v) if v else None for v in (self.processing.get("start"), self.processing.get("end")))


def _number(value) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def _iso(value) -> bool:
    """Une date ISO 8601, ou None (pas de borne)."""
    if value is None:
        return True
    try:
        datetime.fromisoformat(value)
    except (ValueError, TypeError):
        return False
    return True


def _check_manifest(manifest, path: Path) -> None:
    """Le manifeste est écrit par `lab build` : une clé absente ou d'un autre type est refusée, jamais devinée."""
    if not isinstance(manifest, dict):
        raise UserError("E_PROCESSED_MANIFEST_UNREADABLE", path=path)
    schema = manifest.get("schema")
    if schema != PROCESSED_SCHEMA:
        raise UserError("E_PROCESSED_SCHEMA", path=path, value=schema, expected=PROCESSED_SCHEMA)
    texts = ("lab", "point", "code_version")
    numbers = ("offset", "sigma_pressure", "sigma_temperature")
    problems = [key for key in texts if not isinstance(manifest.get(key), str)]
    problems += [key for key in numbers if not _number(manifest.get(key))]
    problems += [key for key in ("sigma_pressure", "sigma_temperature") if _number(manifest.get(key)) and manifest[key] <= 0]
    depths = manifest.get("depths")
    if not (isinstance(depths, list) and len(depths) == 4 and all(_number(d) for d in depths)):
        problems.append("depths")
    digest = manifest.get("cleaned_sha256")
    if not (isinstance(digest, str) and len(digest) == 64):
        problems.append("cleaned_sha256")
    processing = manifest.get("processing")
    if not isinstance(processing, dict) or not all(_iso(processing.get(key)) for key in ("start", "end")):
        problems.append("processing")
    if problems:
        raise UserError("E_PROCESSED_MANIFEST_KEY", path=path, keys=", ".join(dict.fromkeys(problems)))


def read_processed(folder: Path) -> Processed:
    """Lit un dossier `processed_data/<point>/`. Les mesures sont relues sans perte. L'empreinte n'est pas vérifiée ici."""
    folder = Path(folder)
    manifest_path, csv_path = folder / MANIFEST_NAME, folder / MEASUREMENTS_NAME
    for path in (manifest_path, csv_path):
        if not path.is_file():
            raise UserError("E_FILE_NOT_FOUND", path=path)
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except ValueError:
        raise UserError("E_PROCESSED_MANIFEST_UNREADABLE", path=manifest_path) from None
    _check_manifest(manifest, manifest_path)
    return Processed(folder, manifest, read_cleaned_csv(csv_path, exact=True))
