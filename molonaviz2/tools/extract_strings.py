"""Liste les chaînes à traduire du code source, paquet par paquet : appels t("..."), tr("...") (ligne de
commande) et N_("...") avec un littéral et, pour le backend, modèles du catalogue des codes d'erreur
(MESSAGES de errors.py).

Chaque paquet a son catalogue (i18n/fr.json). L'application lit aussi celui du backend : une chaîne
de l'application est traduite si elle est dans l'un des deux.

Usage : python tools/extract_strings.py [backend|frontend]          → chaînes sans traduction française
        python tools/extract_strings.py --all [backend|frontend]    → toutes les chaînes
Sans nom de paquet, les deux paquets sont traités.
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
# Paquet → (source, avec les modèles de MESSAGES, paquets dont il lit aussi le catalogue).
PACKAGES = {
    "backend": (BACKEND, True, ()),
    "frontend": (FRONTEND, False, ("backend",)),
}


def extract(package: str | None = None) -> dict[str, list[str]]:
    """{chaîne: [fichier:ligne, ...]} d'un paquet (des deux si `package` est None), chemins relatifs à l'espace."""
    found: dict[str, list[str]] = {}
    for name in [package] if package else list(PACKAGES):
        source, with_messages, _ = PACKAGES[name]
        for path in sorted(source.rglob("*.py")):
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
        if with_messages:
            for text, where in messages(source):
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


def catalog(package: str, language: str = "fr") -> dict[str, str]:
    """Catalogue propre d'un paquet ({} s'il n'existe pas)."""
    path = PACKAGES[package][0] / "i18n" / f"{language}.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def available(package: str, language: str = "fr") -> dict[str, str]:
    """Traductions dont dispose un paquet : celles des paquets qu'il lit, puis les siennes."""
    merged: dict[str, str] = {}
    for name in (*PACKAGES[package][2], package):
        merged.update(catalog(name, language))
    return merged


if __name__ == "__main__":
    arguments = [a for a in sys.argv[1:] if a != "--all"]
    for name in arguments or list(PACKAGES):
        translations = available(name)
        for text in sorted(extract(name)):
            if "--all" in sys.argv or text not in translations:
                print(f"{name}\t{json.dumps(text, ensure_ascii=False)}")
