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
from pathlib import Path

import pandas as pd

from ..domain.models import CLEANED_COLUMNS

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
