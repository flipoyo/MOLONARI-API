"""Écrire une entrée du rapport et le dossier `Public Report/` (P2c.3d, P2c.3e).

Une entrée est la réponse `Answer` du contrat : en JSON (codec existant, clés triées, saut de ligne final : mêmes octets
d'une génération à l'autre), ou en npz, relisible sans pickle, au-delà de `NPZ_THRESHOLD` octets de JSON ou pour les
questions de la PDF publique (échantillons, données par chaîne : toujours en npz, P2c.3e).

Le dossier est écrit en entier dans `Public Report.partial/`, puis il remplace l'ancien : un échec avant le remplacement
laisse l'ancien rapport intact et ne laisse aucun dossier partiel.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

from ..contract.codec import answer_to_json, answer_to_npz
from ..questions.answer import Answer

NPZ_THRESHOLD = 1 << 20
"""Taille de JSON (octets) au-delà de laquelle une entrée passe en npz. Valeur provisoire : décision P0.5f (1 Mo proposé)."""

ALWAYS_NPZ = frozenset({"posterior_samples", "chain_data"})
"""Questions de la PDF publique dont l'entrée est toujours en npz (échantillons et données par chaîne, P2c.3e)."""

FOLDER_NAME = "Public Report"
INDEX_NAME = "report.json"


def dump_json(document: dict) -> bytes:
    """JSON déterministe : clés triées, UTF-8, indentation de 2, pas de NaN (le codec écrit `null`), saut de ligne final."""
    return (json.dumps(document, sort_keys=True, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")


def sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def encode_entry(question: str, answer: Answer, threshold: int = NPZ_THRESHOLD) -> tuple[str, bytes]:
    """(extension, octets) de l'entrée : `json`, ou `npz` pour la PDF publique ou au-delà de `threshold`."""
    if question not in ALWAYS_NPZ:
        content = dump_json(answer_to_json(answer))
        if len(content) <= threshold:
            return "json", content
    return "npz", answer_to_npz(answer)


def write_folder(target: Path, files: dict[str, bytes]) -> None:
    """Écrit `files` (chemin relatif → octets) dans `target`, qui remplace l'ancien dossier d'un coup."""
    target = Path(target)
    partial = target.with_name(target.name + ".partial")
    old = target.with_name(target.name + ".old")
    for leftover in (partial, old):
        if leftover.exists():
            shutil.rmtree(leftover)
    try:
        for name, content in files.items():
            path = partial / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
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
