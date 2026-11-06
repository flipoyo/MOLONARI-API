"""Registre d'un twin (`registry.csv`) et emplacement de ses résultats (`results/<twin_id>/`).

Disposition sur disque (T7, spec 4 niveaux §9) : le registre et les résultats sont **dans le dossier du twin**, donc dans
le sous-dépôt de ce twin, et le chemin d'un sous-dépôt est le même à tous les niveaux :

    <point>/
      twins/
        twin_mcmc/
          twin.yaml
          registry.csv
          results/<twin_id>/   temperature.csv, water_flux.csv, residuals.csv, metadata.json, chain_state.npz, …

Ancienne disposition, toujours lue avec un avertissement (twins de `examples/`, démo) : un registre partagé
`<point>/twins/registry.csv` et des résultats partagés `<point>/results/<twin_id>/`. On n'y écrit plus.

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
    results: str  # chemin relatif au dossier du twin (ancienne disposition : à la racine du point)
    duplicate_of: str = ""


def registry_path(twin_folder: Path) -> Path:
    return Path(twin_folder) / REGISTRY_NAME


def results_folder(twin_folder: Path) -> Path:
    return Path(twin_folder) / RESULTS_FOLDER


def results_dir(twin_folder: Path, twin_id: str) -> Path:
    return results_folder(twin_folder) / twin_id


def legacy_root(twin_folder: Path) -> Path:
    """Racine de l'ancienne disposition : le parent du dossier `twins`, qui portait `results/`."""
    return Path(twin_folder).parent.parent


def legacy_registry_path(twin_folder: Path) -> Path:
    return Path(twin_folder).parent / REGISTRY_NAME


def legacy_results_dir(twin_folder: Path, twin_id: str) -> Path:
    return legacy_root(twin_folder) / RESULTS_FOLDER / twin_id


def _read_csv(path: Path) -> list[RegistryEntry]:
    if not path.is_file():
        return []
    names = [f.name for f in fields(RegistryEntry)]
    with open(path, newline="", encoding="utf-8") as handle:
        return [RegistryEntry(**{k: row.get(k, "") or "" for k in names}) for row in csv.DictReader(handle)]


def read_registry(twin_folder: Path) -> list[RegistryEntry]:
    """Les lignes du registre du twin `twin_folder`, anciennes d'abord : celles du registre partagé de l'ancienne disposition
    (pour les twins qui ont encore des résultats là), puis celles du registre du twin, qui l'emportent (`entry_for_name`
    prend la dernière)."""
    own = _read_csv(registry_path(twin_folder))
    seen = {(e.twin_id, e.name, e.status) for e in own}
    old = [e for e in _read_csv(legacy_registry_path(twin_folder)) if (e.twin_id, e.name, e.status) not in seen]
    return old + own


def locate_results(twin_folder: Path, entry: RegistryEntry) -> tuple[Path, bool] | None:
    """Où sont les résultats de `entry` : (dossier, ancien emplacement ?) ou None s'ils ont disparu. Le dossier du twin
    d'abord ; sinon l'ancien `<point>/results/`, que la ligne cite relativement à sa racine."""
    twin_folder = Path(twin_folder)
    for candidate, legacy in ((twin_folder / entry.results, False), (results_dir(twin_folder, entry.twin_id), False),
                              (legacy_root(twin_folder) / entry.results, True), (legacy_results_dir(twin_folder, entry.twin_id), True)):
        if (candidate / "metadata.json").is_file():
            return candidate, legacy
    return None


def append_entry(twin_folder: Path, entry: RegistryEntry) -> None:
    path = registry_path(twin_folder)
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
