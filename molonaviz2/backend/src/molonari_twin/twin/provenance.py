"""Provenance d'un twin, écrite dans metadata.json["provenance"], et motifs du mode strict.

- code : dépôt du code (qui contient molonaviz2/backend/) ; twin : dépôt du dossier du twin. Pour chacun :
  remote sans identifiants, commit, tags exacts, branche ou HEAD détaché, état propre limité au chemin.
- environment : Python, plateforme, processeur, appareil, versions de jax, jaxlib et numpy.
  pyheatmy n'est pas enregistré : le moteur est un portage JAX qui ne l'importe pas.
- cgs_state_sha256 : empreinte du fichier d'état .gts cgs le plus récent de l'espace de travail.
- warnings : ce qui manque (W_PROVENANCE_*), sous la forme {code, params}. Rien n'échoue en silence.
"""

from __future__ import annotations

import hashlib
import platform
import re
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from ..errors import Issue
from ..gitinfo import GitUnavailable, repository_state

LIBRARIES = ("jax", "jaxlib", "numpy")
EMPTY_REPOSITORY = {"path": None, "remote": None, "commit": None, "tags": [], "branch": None, "detached_head": None, "clean": None}


def code_path() -> Path:
    """Dossier du code : le projet backend/ en installation éditable, sinon le paquet installé."""
    package = Path(__file__).resolve().parents[1]
    project = package.parents[1]
    return project if (project / "pyproject.toml").is_file() else package


def _repository(path: Path, target: str, warnings: list[dict]) -> dict:
    try:
        state = repository_state(path)
    except GitUnavailable:
        if not any(w["code"] == "W_PROVENANCE_NO_GIT" for w in warnings):
            warnings.append({"code": "W_PROVENANCE_NO_GIT", "params": {}})
        return dict(EMPTY_REPOSITORY)
    if state is None:
        warnings.append({"code": "W_PROVENANCE_NOT_A_REPOSITORY", "params": {"target": target}})
        return dict(EMPTY_REPOSITORY)
    if state.remote is None:
        warnings.append({"code": "W_PROVENANCE_NO_REMOTE", "params": {"target": target}})
    return state.to_dict()


def _library_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def cgs_state_digest(*starts: Path) -> str | None:
    """SHA-256 du .gts le plus récent de `.cgitsync/state/`, cherché en remontant depuis chaque point de départ."""
    for start in starts:
        for folder in (start.resolve(), *start.resolve().parents):
            files = sorted((folder / ".cgitsync" / "state").glob("*.gts"), key=lambda p: (p.stat().st_mtime, p.name))
            if files:
                return hashlib.sha256(files[-1].read_bytes()).hexdigest()
    return None


def twin_provenance(twin_folder: Path, device: str) -> dict:
    warnings: list[dict] = []
    code = code_path()
    return {
        "code": _repository(code, "code", warnings),
        "twin": _repository(Path(twin_folder), "twin", warnings),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "processor": platform.processor() or platform.machine(),
            "device": device,
            **{name: _library_version(name) for name in LIBRARIES},
        },
        "cgs_state_sha256": cgs_state_digest(Path(twin_folder), code),
        "warnings": warnings,
    }


def provenance_warnings(provenance: dict | None) -> list[Issue]:
    """Avertissements enregistrés dans une provenance, sous forme d'Issue (pour l'affichage)."""
    return [Issue(None, w["code"], **w.get("params", {})) for w in (provenance or {}).get("warnings", [])]

PRERELEASE = re.compile(r"-(alpha|beta|rc)\.\d+")


def strict_problems(code_version: str, package_version: str, provenance: dict) -> list[Issue]:
    """Motifs de refus du mode strict : version pré-publiée ou locale, dépôts modifiés, twin sans tag, provenance illisible."""
    problems = []
    if PRERELEASE.search(package_version):
        problems.append(Issue(None, "E_STRICT_PRERELEASE", version=package_version))
    if "+g" in code_version:
        problems.append(Issue(None, "E_STRICT_LOCAL_BUILD", version=code_version))
    for target in ("code", "twin"):
        entry = provenance[target]
        if entry["commit"] is None:
            problems.append(Issue(None, "E_STRICT_NO_PROVENANCE", target=target))
        elif entry["clean"] is not True:
            problems.append(Issue(None, "E_STRICT_DIRTY", target=target))
    if provenance["twin"]["commit"] is not None and not provenance["twin"]["tags"]:
        problems.append(Issue(None, "E_STRICT_UNTAGGED"))
    return problems
