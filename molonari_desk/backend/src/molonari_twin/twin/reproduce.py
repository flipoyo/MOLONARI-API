"""Reproduire une chaîne calculée et vérifier le déterminisme.

`reproduce <twin>` relance `lab build` puis `run --strict` sans cache, calcule les SHA-256 de tous
les fichiers produits, les compare à la référence et écrit un compte rendu sans la graine : machine,
versions, empreintes, résultat, écart maximal par fichier.

La commande est réservée à `opx`.
"""

from __future__ import annotations

import hashlib
import json
import platform
import socket
import zipfile
from dataclasses import dataclass, field
from datetime import datetime
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
    import shutil

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
