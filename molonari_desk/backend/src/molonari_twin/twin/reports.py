"""Lire les rapports publics d'un point avec l'outil (niveau user, facultatif : le public lit les fichiers).

    molonari-twin reports list [<dossier>]    les points qui ont un `public_reports/` (par défaut le dossier courant)
    molonari-twin reports show <point>        l'index du rapport : twins, plateforme, entrées, figures
    molonari-twin reports chains <point>      résumé de `chains.npz` : chaînes, échantillons, publication_thin, SHA-256 cité

`<point>` est le dossier du sampling point dans MOLONARI-TWIN (`<point>/public_reports/report.json`, `chains.npz`).
Lecture seule, sans JAX ni calcul : `chains.npz` se lit sans pickle (`generate.read_chains`). Rien ici ne donne de graine ni
de chemin machine : le rapport public n'en contient pas, et les chemins rendus sont relatifs au point.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from ..errors import UserError
from .generate import CHAINS_FILE, PUBLIC_REPORTS, read_chains

INDEX_FILE = "report.json"  # report.writer.INDEX_NAME ; non importé pour ne pas charger tout le générateur de rapport


def _folder(point: str | Path) -> Path:
    root = Path(point).resolve()
    if not root.is_dir():
        raise UserError("E_FOLDER_NOT_FOUND", path=str(point))
    return root


def list_points(folder: str | Path = ".") -> list[dict]:
    """Les points de `folder` qui ont un `public_reports/` (ou `folder` lui-même, s'il en a un), triés par nom."""
    root = _folder(folder)
    candidates = [root] if (root / PUBLIC_REPORTS).is_dir() else sorted(p for p in root.iterdir() if (p / PUBLIC_REPORTS).is_dir())
    points = []
    for candidate in candidates:
        reports = candidate / PUBLIC_REPORTS
        points.append({"point": candidate.name, "report": (reports / INDEX_FILE).is_file(), "chains": (reports / CHAINS_FILE).is_file()})
    return points


def read_index(point: str | Path) -> dict:
    """Index `report.json` du point ; E_REPORTS_NO_REPORT s'il n'y a pas de rapport publié, E_REPORTS_INDEX_INVALID s'il est illisible."""
    root = _folder(point)
    path = root / PUBLIC_REPORTS / INDEX_FILE
    if not path.is_file():
        raise UserError("E_REPORTS_NO_REPORT", point=root.name, file=f"{PUBLIC_REPORTS}/{INDEX_FILE}")
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        document = None
    if not isinstance(document, dict) or not isinstance(document.get("entries"), list):
        raise UserError("E_REPORTS_INDEX_INVALID", point=root.name, file=f"{PUBLIC_REPORTS}/{INDEX_FILE}")
    return document


def show_report(point: str | Path) -> dict:
    """Ce que `reports show` affiche : l'index sans les empreintes de recette, les entrées et les figures résumées."""
    index = read_index(point)
    return {
        "point": index.get("point") or _folder(point).name,
        "report_format": index.get("report_format"),
        "generated_at": index.get("generated_at"),
        "generator_version": index.get("generator_version"),
        "space": {role: (None if entry is None else {"twin": entry.get("twin"), "commit": entry.get("commit")})
                  for role, entry in (index.get("space") or {}).items()},
        "platform": index.get("platform") or {},
        "entries": [{key: entry.get(key) for key in ("file", "question", "model", "uncertainty", "sha256")} for entry in index["entries"]],
        "skipped": [{"question": s.get("question"), "model": s.get("model"), "code": s.get("code")} for s in index.get("skipped") or []],
        "figures": [figure.get("file") for figure in index.get("figures") or []],
        "chains": (index.get("chains") or {}).get("file"),
    }


def summarize_chains(point: str | Path) -> dict:
    """Résumé de `chains.npz` : chaînes, échantillons par chaîne, couches, itérations, `publication_thin` et SHA-256 du
    `posterior.npz` complet cités par l'index (None sans index). `sha256_matches` dit si l'empreinte de `chains.npz` est celle de l'index."""
    root = _folder(point)
    path = root / PUBLIC_REPORTS / CHAINS_FILE
    if not path.is_file():
        raise UserError("E_REPORTS_NO_CHAINS", point=root.name, file=f"{PUBLIC_REPORTS}/{CHAINS_FILE}")
    try:
        chains = read_chains(path)
        iterations, nb_chains = np.asarray(chains["iteration"]), len(chains["chain"])
        layers = int(np.asarray(chains["samples.porosity"]).shape[-1])
    except (OSError, ValueError, KeyError, IndexError):
        raise UserError("E_REPORTS_CHAINS_INVALID", point=root.name, file=f"{PUBLIC_REPORTS}/{CHAINS_FILE}") from None
    try:
        cited = read_index(root).get("chains") or {}
    except UserError:  # sans index lisible, on résume tout de même le fichier
        cited = {}
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {
        "point": root.name,
        "chains": nb_chains,
        "samples_per_chain": int(iterations.size),
        "layers": layers,
        "first_iteration": int(iterations[0]) if iterations.size else None,
        "last_iteration": int(iterations[-1]) if iterations.size else None,
        "publication_thin": cited.get("publication_thin"),
        "posterior_sha256": cited.get("posterior_sha256"),
        "sha256": digest,
        "sha256_matches": (digest == cited["sha256"]) if cited.get("sha256") else None,
    }
