"""Reproduire une chaîne calculée et vérifier le déterminisme.

`reproduce <twin>` relance `run --strict` sans cache (les données traitées ne sont pas reconstruites : `lab build`
reste à lancer avant), calcule les SHA-256 de tous les fichiers produits, les compare à la référence et écrit un
compte rendu sans la graine : machine, versions, empreintes, résultat, écart maximal par fichier. Le mode strict
refuse une pré-version, un code local ou un dépôt modifié ou sans tag : sur un code non publié, utiliser `--export`
et `--compare`.

`reproduce <twin> --export DOSSIER` n'exécute rien : il copie les résultats existants dans DOSSIER/results/ et écrit
`fingerprint.json` (empreintes, versions, machine) et `resultats.txt` (valeurs clés en repr et en hex), sans la graine.
`reproduce --compare A B` compare deux exports (fichier par fichier, tableau par tableau pour les .npz) : code de sortie
0 si identiques, 1 sinon.

La commande est réservée à `opx`.
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
import shutil
import socket
import zipfile
from dataclasses import dataclass, field
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np

from ..errors import UserError
from .config import load_twin
from .run import run_twin, twin_status


@dataclass(frozen=True)
class FileComparison:
    """Comparaison d'un fichier avec sa référence."""

    path: str  # chemin relatif au dossier des résultats
    reference_sha256: str
    reproduced_sha256: str
    match: bool
    size_bytes: int


@dataclass(frozen=True)
class ReproductionReport:
    """Compte rendu de reproduction."""

    machine: str
    platform_system: str
    platform_release: str
    python_version: str
    code_version: str
    device: str
    timestamp: str
    twin_id: str
    all_files_match: bool
    files_compared: list[FileComparison] = field(default_factory=list)
    comparable_metadata_match: bool = True
    chain_state_match: bool = True
    message: str = "IDENTIQUE"

    def to_dict(self) -> dict:
        """Représentation JSON du rapport, sans la graine."""
        return {
            "machine": self.machine,
            "platform": {"system": self.platform_system, "release": self.platform_release},
            "python_version": self.python_version,
            "code_version": self.code_version,
            "device": self.device,
            "timestamp": self.timestamp,
            "twin_id": self.twin_id,
            "all_files_match": self.all_files_match,
            "files_compared": [
                {"path": f.path, "reference_sha256": f.reference_sha256, "reproduced_sha256": f.reproduced_sha256,
                 "match": f.match, "size_bytes": f.size_bytes}
                for f in self.files_compared
            ],
            "comparable_metadata_match": self.comparable_metadata_match,
            "chain_state_match": self.chain_state_match,
            "message": self.message,
        }

    def to_text(self) -> str:
        """Représentation textuelle lisible du rapport."""
        lines = [
            "=" * 70, "Rapport de reproduction", "=" * 70, "",
            f"Machine: {self.machine}",
            f"Plateforme: {self.platform_system} {self.platform_release}",
            f"Python: {self.python_version}",
            f"Code: {self.code_version}",
            f"Device: {self.device}",
            f"Date: {self.timestamp}", "",
            f"Twin: {self.twin_id}", "",
        ]
        if self.all_files_match:
            lines.append("IDENTIQUE")
        else:
            lines.append("DIFFERENCES DETECTEES")
            lines.append("")
            for f in self.files_compared:
                status = "ok" if f.match else "DIFFERENT"
                lines.append(f"  {f.path}: {status} ({f.size_bytes} octets)")
                if not f.match:
                    lines.append(f"    reference:  {f.reference_sha256}")
                    lines.append(f"    reproduit:  {f.reproduced_sha256}")
            if not self.comparable_metadata_match:
                lines.append("  metadata: DIFFERENT (hors date, duree, provenance)")
            if not self.chain_state_match:
                lines.append("  chain_state.npz: DIFFERENT (hors burn_in_duration_s)")
        lines += ["", self.message, "=" * 70]
        return "\n".join(lines)


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _comparable_metadata(results_dir: Path) -> dict:
    """Metadonnees comparables (comme dans test_determinism.py)."""
    metadata = json.loads((results_dir / "metadata.json").read_text(encoding="utf-8"))
    for key in ("created_at", "duration_s", "provenance"):
        metadata.pop(key, None)
    files = metadata.get("files")
    if files and "chain_state.npz" in files:
        files.pop("chain_state.npz")
    return metadata


def _compare_chain_state(ref_dir: Path, repr_dir: Path) -> bool:
    """chain_state.npz : tableau par tableau, sans burn_in_duration_s (T3)."""
    ref_file = ref_dir / "chain_state.npz"
    repr_file = repr_dir / "chain_state.npz"
    if not ref_file.is_file():
        return not repr_file.is_file()
    if not repr_file.is_file():
        return False
    try:
        with np.load(ref_file, allow_pickle=False) as ref, np.load(repr_file, allow_pickle=False) as rep:
            if sorted(ref.keys()) != sorted(rep.keys()):
                return False
            return all(ref[k].tobytes() == rep[k].tobytes() for k in ref if k != "burn_in_duration_s")
    except (OSError, ValueError, KeyError, zipfile.BadZipFile):
        return False


def reproduce(twin_path: str | Path, strict: bool = True) -> ReproductionReport:
    """Reproduit un twin calcule et compare les resultats a la reference.

    Supprime les resultats existants, relance `run_twin` (avec ``strict`` par defaut),
    compare fichier par fichier et ecrit un compte rendu sans la graine.
    """
    from ..engine import api

    twin_path = Path(twin_path).resolve()
    config = load_twin(twin_path)

    status = twin_status(config)
    if not status.has_results:
        raise UserError("E_REPRODUCE_NO_REFERENCE", name=config.name)

    reference_dir = status.results_dir

    # Sauvegarder les resultats de reference
    backup = reference_dir.with_name(reference_dir.name + ".ref_backup")
    if backup.exists():
        shutil.rmtree(backup)
    shutil.copytree(reference_dir, backup)

    # Supprimer les resultats pour forcer le recalcul
    shutil.rmtree(reference_dir)

    # Relancer
    try:
        result = run_twin(config, strict=strict)
    except Exception as exc:
        # Restaurer la reference en cas d'echec
        if not reference_dir.exists():
            shutil.copytree(backup, reference_dir)
        shutil.rmtree(backup, ignore_errors=True)
        raise UserError("E_REPRODUCE_RUN_FAILED", error=str(exc)) from exc

    if not result.computed:
        shutil.rmtree(backup, ignore_errors=True)
        raise UserError("E_REPRODUCE_NOT_RECOMPUTED", name=config.name)

    reproduced_dir = result.status.results_dir

    # Comparer les fichiers (sauf metadata.json et chain_state.npz, traites a part)
    skip = {"metadata.json", "chain_state.npz"}
    ref_names = {p.name for p in backup.iterdir() if p.is_file()}
    rep_names = {p.name for p in reproduced_dir.iterdir() if p.is_file()}

    comparisons: list[FileComparison] = []
    for name in sorted(ref_names | rep_names):
        if name in skip:
            continue
        ref_file, rep_file = backup / name, reproduced_dir / name
        if ref_file.is_file() and rep_file.is_file():
            ref_sha = _sha256_file(ref_file)
            rep_sha = _sha256_file(rep_file)
            comparisons.append(FileComparison(name, ref_sha, rep_sha, ref_sha == rep_sha, ref_file.stat().st_size))
        elif ref_file.is_file():
            comparisons.append(FileComparison(name, _sha256_file(ref_file), "", False, ref_file.stat().st_size))
        else:
            comparisons.append(FileComparison(name, "", _sha256_file(rep_file), False, rep_file.stat().st_size))

    metadata_match = _comparable_metadata(backup) == _comparable_metadata(reproduced_dir)
    cs_match = _compare_chain_state(backup, reproduced_dir)
    all_match = all(c.match for c in comparisons) and metadata_match and cs_match

    # Nettoyer le backup
    shutil.rmtree(backup, ignore_errors=True)

    return ReproductionReport(
        machine=socket.gethostname(),
        platform_system=platform.system(),
        platform_release=platform.release(),
        python_version=platform.python_version(),
        code_version=api.engine_version(),
        device=api.device(),
        timestamp=datetime.now().isoformat(timespec="seconds"),
        twin_id=status.twin_id,
        all_files_match=all_match,
        files_compared=comparisons,
        comparable_metadata_match=metadata_match,
        chain_state_match=cs_match,
        message="IDENTIQUE" if all_match else "DIFFERENCES DETECTEES",
    )


# --- export et comparaison entre deux machines (T10) ---

FINGERPRINT_MACHINE_FIELDS = ("platform", "device")  # propres à la machine : jamais une « recette différente »
FINGERPRINT_RECIPE_FIELDS = ("twin_id", "engine_version", "jax", "jaxlib", "numpy", "cleaned_sha256")


def _library_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _metadata_sha256(results_dir: Path) -> str:
    canonical = json.dumps(_comparable_metadata(results_dir), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _key_values(results_dir: Path) -> list[tuple[str, float | None]]:
    """Flux d'eau moyen, flux de chaleur (z = 0) et RMSE, par les fonctions des questions ; None si indisponible."""
    # imports dans la fonction : twin/ n'importe pas questions/ au chargement (cycle)
    from ..questions.flux import water_flux
    from ..questions.heat import heat_budget
    from .results import load_results

    results = load_results(results_dir)
    values: list[tuple[str, float | None]] = []
    try:
        values.append(("water_flux_mean", float(water_flux(results).data["mean"].value)))
    except UserError:
        values.append(("water_flux_mean", None))
    try:
        heat = heat_budget(results, z=0.0).data
        for key in ("total", "advective", "conductive"):
            values.append((f"heat_{key}", float(heat[key].value)))
    except UserError:
        values += [(f"heat_{key}", None) for key in ("total", "advective", "conductive")]
    rmse = results.metadata.get("rmse_celsius") or {}
    for key in ("t1", "t2", "t3", "total"):
        value = rmse.get(key)
        values.append((f"rmse_{key}", None if value is None else float(value)))
    return values


def export_twin(twin_path: str | Path, folder: str | Path) -> Path:
    """Exporte les résultats existants d'un twin et leurs empreintes dans `folder`, sans relancer le calcul ni écrire la graine."""
    from ..engine import api
    from .provenance import cgs_state_digest, code_path

    twin_path = Path(twin_path).resolve()
    config = load_twin(twin_path)
    status = twin_status(config)
    if not status.has_results:
        raise UserError("E_REPRODUCE_NO_REFERENCE", name=config.name)
    results_dir = status.results_dir
    folder = Path(folder).resolve()
    exported = folder / "results"
    if exported.exists():
        shutil.rmtree(exported)
    exported.mkdir(parents=True)
    names = sorted(p.name for p in results_dir.iterdir() if p.is_file())
    for name in names:
        shutil.copy2(results_dir / name, exported / name)

    fingerprint = {
        "files": {name: _sha256_file(exported / name) for name in names},
        "comparable_metadata_sha256": _metadata_sha256(exported),
        "cleaned_sha256": (config.processed_manifest or {}).get("cleaned_sha256"),
        "twin_id": status.twin_id,
        "engine_version": api.engine_version(),
        "device": api.device(),
        "platform": {"system": platform.system(), "release": platform.release(),
                     "machine": platform.machine(), "processor": platform.processor()},
        "jax": _library_version("jax"),
        "jaxlib": _library_version("jaxlib"),
        "numpy": _library_version("numpy"),
        "cgs_state_sha256": cgs_state_digest(Path(config.folder), code_path()),
    }
    (folder / "fingerprint.json").write_text(json.dumps(fingerprint, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [f"# mode = {config.mode} (en mcmc, le flux d'eau est la moyenne a posteriori)"]
    for name, value in _key_values(exported):
        lines.append(f"{name} = indisponible" if value is None else f"{name} = {value!r} | {value.hex()}")
    (folder / "resultats.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return folder


@dataclass(frozen=True)
class ExportFileComparison:
    """Comparaison d'un fichier des résultats entre deux exports."""

    path: str
    match: bool
    detail: str = ""  # explication d'une différence (présence, empreinte)
    arrays: list[dict] = field(default_factory=list)  # tableaux différents d'un .npz : nom, motif, écart


@dataclass(frozen=True)
class ExportComparison:
    """Comparaison de deux exports de twin."""

    identical: bool
    files: list[ExportFileComparison] = field(default_factory=list)
    recipe_differences: list[str] = field(default_factory=list)  # « recette différente » : champs de fingerprint.json

    def to_dict(self) -> dict:
        return {
            "identical": self.identical,
            "files": [{"path": f.path, "match": f.match, "detail": f.detail, "arrays": f.arrays} for f in self.files],
            "recipe_differences": self.recipe_differences,
        }

    def to_text(self) -> str:
        lines = ["=" * 70, "Comparaison de deux exports", "=" * 70, ""]
        lines.append("IDENTIQUE" if self.identical else "DIFFERENCES DETECTEES")
        for f in self.files:
            lines.append(f"  {f.path}: {'ok' if f.match else 'DIFFERENT'}" + (f" ({f.detail})" if f.detail else ""))
            for array in f.arrays:
                lines.append(f"    {array['name']}: {array['reason']}")
        for name in self.recipe_differences:
            lines.append(f"  recette différente : {name}")
        lines.append("=" * 70)
        return "\n".join(lines)


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _compare_arrays(a: np.ndarray, b: np.ndarray) -> dict | None:
    """Différence entre deux tableaux (forme, type, éléments, écart absolu maximal), ou None s'ils sont égaux."""
    if a.shape != b.shape:
        return {"reason": f"forme {a.shape} contre {b.shape}", "shape": [list(a.shape), list(b.shape)]}
    if a.dtype != b.dtype:
        return {"reason": f"type {a.dtype} contre {b.dtype}", "dtype": [str(a.dtype), str(b.dtype)]}
    if np.issubdtype(a.dtype, np.number):
        different = ~((a == b) | (np.isnan(a) & np.isnan(b))) if np.issubdtype(a.dtype, np.inexact) else (a != b)
        count = int(different.sum())
        if count == 0:
            return None
        gap = float(np.nanmax(np.abs(a[different].astype(float) - b[different].astype(float))))
        if math.isnan(gap):
            gap = float("inf")
        return {"reason": f"{count} éléments différents sur {a.size}, écart absolu maximal {gap!r}", "different": count, "max_abs_diff": gap}
    count = int((a != b).sum())
    return None if count == 0 else {"reason": f"{count} éléments différents sur {a.size}", "different": count}


def _compare_npz(path_a: Path, path_b: Path, name: str) -> list[dict]:
    try:
        with np.load(path_a, allow_pickle=False) as a, np.load(path_b, allow_pickle=False) as b:
            keys_a, keys_b = set(a.keys()), set(b.keys())
            out = [{"name": k, "reason": "absent de B"} for k in sorted(keys_a - keys_b)]
            out += [{"name": k, "reason": "absent de A"} for k in sorted(keys_b - keys_a)]
            for key in sorted(keys_a & keys_b):
                if name == "chain_state.npz" and key == "burn_in_duration_s":
                    continue
                diff = _compare_arrays(a[key], b[key])
                if diff is not None:
                    out.append({"name": key, **diff})
            return out
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        return [{"name": "(fichier)", "reason": f"illisible : {exc}"}]


def compare_exports(a: str | Path, b: str | Path) -> ExportComparison:
    """Compare deux dossiers produits par `export_twin`, fichier par fichier, puis les empreintes hors machine."""
    a, b = Path(a), Path(b)
    results_a, results_b = a / "results", b / "results"
    names_a = {p.name for p in results_a.iterdir() if p.is_file()} if results_a.is_dir() else set()
    names_b = {p.name for p in results_b.iterdir() if p.is_file()} if results_b.is_dir() else set()
    files: list[ExportFileComparison] = []
    for name in sorted(names_a | names_b):
        if name not in names_b:
            files.append(ExportFileComparison(name, False, "absent de B"))
        elif name not in names_a:
            files.append(ExportFileComparison(name, False, "absent de A"))
        elif name == "metadata.json":
            same = _metadata_sha256(results_a) == _metadata_sha256(results_b)
            files.append(ExportFileComparison(name, same, "" if same else "métadonnées comparables différentes (hors date, durée, provenance)"))
        elif _sha256_file(results_a / name) == _sha256_file(results_b / name):
            files.append(ExportFileComparison(name, True))
        elif name.endswith(".npz"):
            arrays = _compare_npz(results_a / name, results_b / name, name)
            # chain_state.npz : burn_in_duration_s est exclu (T3), donc des octets différents sans tableau différent y sont admis
            admitted = name == "chain_state.npz" and not arrays
            detail = "" if admitted else "empreintes SHA-256 différentes" + ("" if arrays else " (tableaux identiques)")
            files.append(ExportFileComparison(name, admitted, detail, arrays))
        else:
            files.append(ExportFileComparison(name, False, "empreintes SHA-256 différentes"))

    recipe: list[str] = []
    fp_a, fp_b = a / "fingerprint.json", b / "fingerprint.json"
    if fp_a.is_file() and fp_b.is_file():
        fa, fb = _read_json(fp_a), _read_json(fp_b)
        recipe = [k for k in FINGERPRINT_RECIPE_FIELDS if fa.get(k) != fb.get(k)]
    else:
        recipe = ["fingerprint.json"]
    identical = all(f.match for f in files) and not recipe
    return ExportComparison(identical, files, recipe)
