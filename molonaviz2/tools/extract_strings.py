"""Liste les chaînes à traduire du code source : appels t("..."), tr("...") (ligne de commande) et
N_("...") avec un littéral, et modèles du catalogue des codes d'erreur (MESSAGES de errors.py).

Le catalogue fr.json est encore commun au backend et au frontend : les deux sources sont parcourues.

Usage : python tools/extract_strings.py            → affiche les chaînes sans traduction française
        python tools/extract_strings.py --all      → affiche toutes les chaînes
"""

from __future__ import annotations

import ast
import json
import re
import sys
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[1]
BACKEND = WORKSPACE / "backend" / "src" / "molonari_twin"
FRONTEND = WORKSPACE / "frontend" / "src" / "molonaviz2"
SOURCES = (BACKEND, FRONTEND)
CATALOG = BACKEND / "i18n" / "fr.json"


def extract(sources: tuple[Path, ...] = SOURCES) -> dict[str, list[str]]:
    """{chaîne: [fichier:ligne, ...]}, chemins relatifs à l'espace."""
    found: dict[str, list[str]] = {}
    paths = sorted(path for source in sources for path in source.rglob("*.py"))
    for path in paths:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id in ("t", "tr", "N_")
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)
            ):
                found.setdefault(node.args[0].value, []).append(f"{path.relative_to(WORKSPACE)}:{node.lineno}")
    for text, where in messages(BACKEND):
        found.setdefault(text, []).append(where)
    return found


def messages(source: Path = BACKEND) -> list[tuple[str, str]]:
    """Modèles anglais du catalogue des codes (MESSAGES de errors.py), sauf ceux sans lettre hors de leurs champs."""
    path = source / "errors.py"
    for node in ast.parse(path.read_text(encoding="utf-8"), filename=str(path)).body:
        if isinstance(node, ast.AnnAssign | ast.Assign) and "MESSAGES" in ast.unparse(node.targets[0] if isinstance(node, ast.Assign) else node.target):
            return [
                (value.value, f"errors.py:{value.lineno}")
                for value in node.value.values
                if isinstance(value, ast.Constant) and re.search(r"[^\W\d_]", re.sub(r"\{[^}]*\}", "", value.value))
            ]
    return []


if __name__ == "__main__":
    strings = extract()
    catalog = json.loads(CATALOG.read_text(encoding="utf-8"))
    for text in sorted(strings):
        if "--all" in sys.argv or text not in catalog:
            print(json.dumps(text, ensure_ascii=False))
