"""Liste les chaînes à traduire du code source : appels t("...") et N_("...") avec un littéral.

Usage : python tools/extract_strings.py            → affiche les chaînes sans traduction française
        python tools/extract_strings.py --all      → affiche toutes les chaînes
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[1] / "src" / "molonaviz2"


def extract(source: Path = SOURCE) -> dict[str, list[str]]:
    """{chaîne: [fichier:ligne, ...]}"""
    found: dict[str, list[str]] = {}
    for path in sorted(source.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id in ("t", "N_")
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)
            ):
                found.setdefault(node.args[0].value, []).append(f"{path.relative_to(source)}:{node.lineno}")
    return found


if __name__ == "__main__":
    strings = extract()
    catalog = json.loads((SOURCE / "i18n" / "fr.json").read_text(encoding="utf-8"))
    for text in sorted(strings):
        if "--all" in sys.argv or text not in catalog:
            print(json.dumps(text, ensure_ascii=False))
