"""Registre des twins (`twins/registry.csv`) et emplacement des résultats (`results/<twin_id>/`).

Disposition sur disque (le dossier « twins » est celui choisi dans l'application) :

    <racine>/
      twins/
        registry.csv
        twin_01/twin.yaml (+ fichier de mesures)
        twin_02/twin.yaml
      results/
        <twin_id>/  temperature.csv, water_flux.csv, residuals.csv, metadata.json, …

Le registre garde une ligne par twin exécuté (ou reconnu comme doublon). Il permet :
- de renvoyer au résultat existant quand une config identique est relancée ;
- de refuser un twin dont la config a été modifiée après son exécution (config figée).
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, fields
from pathlib import Path

REGISTRY_NAME = "registry.csv"
RESULTS_FOLDER = "results"


@dataclass
class RegistryEntry:
    twin_id: str
    name: str
    mode: str
    status: str  # « computed » ou « duplicate »
    created_at: str
    duration_s: str
    code_version: str
    results: str  # chemin relatif à la racine (parent du dossier twins)
    duplicate_of: str = ""


def twins_root(twins_folder: Path) -> Path:
    return Path(twins_folder).parent


def results_folder(twins_folder: Path) -> Path:
    return twins_root(twins_folder) / RESULTS_FOLDER


def results_dir(twins_folder: Path, twin_id: str) -> Path:
    return results_folder(twins_folder) / twin_id


def read_registry(twins_folder: Path) -> list[RegistryEntry]:
    path = Path(twins_folder) / REGISTRY_NAME
    if not path.is_file():
        return []
    names = [f.name for f in fields(RegistryEntry)]
    with open(path, newline="", encoding="utf-8") as handle:
        return [RegistryEntry(**{k: row.get(k, "") or "" for k in names}) for row in csv.DictReader(handle)]


def append_entry(twins_folder: Path, entry: RegistryEntry) -> None:
    path = Path(twins_folder) / REGISTRY_NAME
    new = not path.is_file()
    with open(path, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[f.name for f in fields(RegistryEntry)])
        if new:
            writer.writeheader()
        writer.writerow(asdict(entry))


def computed_entry(entries: list[RegistryEntry], twin_id: str) -> RegistryEntry | None:
    """Première exécution réelle de cet identifiant (la source des résultats)."""
    return next((e for e in entries if e.twin_id == twin_id and e.status == "computed"), None)


def entry_for_name(entries: list[RegistryEntry], name: str) -> RegistryEntry | None:
    return next((e for e in reversed(entries) if e.name == name), None)
