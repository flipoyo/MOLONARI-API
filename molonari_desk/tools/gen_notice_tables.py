"""Génère la table des paramètres de la notice scientifique (docs/NOTICE_SCIENTIFIQUE.md) à partir de la notice
intégrée à l'application (molonari_desk.notice, Aide › Notice des paramètres).

    python tools/gen_notice_tables.py           # réécrit la table entre ses deux marqueurs
    python tools/gen_notice_tables.py --check   # code 1 si la table n'est pas à jour (aussi vérifié par les tests)

Les unités, valeurs par défaut, bornes et clés de twin.yaml ne sont pas recopiées à la main : elles viennent du
domaine (molonari_twin.domain.compute) par la notice de l'application, traduites en français sans Qt.
"""

from __future__ import annotations

import sys
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[1]
DOC = WORKSPACE / "docs" / "NOTICE_SCIENTIFIQUE.md"
BEGIN = "<!-- début de la table des paramètres : générée par tools/gen_notice_tables.py, ne pas modifier à la main -->"
END = "<!-- fin de la table des paramètres -->"
LANGUAGE = "fr"


def _cell(text: str) -> str:
    return " ".join(text.split()).replace("|", "\\|") or "—"


def table() -> str:
    """Tables Markdown des réglages de calcul (couches, modèle direct, inversion, réglages avancés), en français."""
    from molonari_desk import notice
    from molonari_desk.i18n import translate

    original = notice.t
    notice.t = lambda text: translate(text, LANGUAGE)  # pas de Qt : la notice est traduite avec le catalogue réuni
    try:
        lines = []
        for section in (notice.LAYERS, notice.DIRECT, notice.INVERSION, notice.ADVANCED):
            entries = [entry for entry in notice.ENTRIES if entry.section == section]
            lines += [f"**{notice.t(section)}**", "",
                      "| Paramètre | Unité | Par défaut | Valeurs admises | Clé de `twin.yaml` | Définition |",
                      "|---|---|---|---|---|---|"]
            for entry in entries:
                twin_key = f"`{entry.twin_key}`" if entry.twin_key else ""
                lines.append("| " + " | ".join(_cell(value) for value in (
                    entry.title(),
                    notice.t(entry.unit) if entry.unit else "",
                    entry.default() if entry.default else "",
                    entry.limits() if entry.limits else "",
                    twin_key,
                    entry.text("summary"),
                )) + " |")
            lines.append("")
        return "\n".join(lines).rstrip("\n")
    finally:
        notice.t = original


def render(text: str) -> str:
    """`text` avec la table régénérée entre ses marqueurs."""
    start, end = text.index(BEGIN) + len(BEGIN), text.index(END)
    return text[:start] + "\n" + table() + "\n" + text[end:]


def main(argv: list[str]) -> int:
    current = DOC.read_text(encoding="utf-8")
    expected = render(current)
    if "--check" in argv:
        if current != expected:
            print(f"{DOC.relative_to(WORKSPACE)} n'est pas à jour : python tools/gen_notice_tables.py", file=sys.stderr)
            return 1
        return 0
    DOC.write_text(expected, encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
