"""Génère la référence des questions de l'API /v1 (backend/docs/API_V1.md) à partir du catalogue.

    python tools/gen_api_doc.py           # écrit le fichier
    python tools/gen_api_doc.py --check   # code 1 si le fichier n'est pas à jour (aussi vérifié par les tests)

La documentation suit le code : paramètres (schémas pydantic du contrat), sorties déclarées (vérifiées sur de
vraies réponses par tests/test_api.py), modes, versions et codes d'erreur viennent du backend lui-même.
"""

from __future__ import annotations

import sys
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[1]
OUTPUT = WORKSPACE / "backend" / "docs" / "API_V1.md"
CATEGORIES = {
    "discovery": "Découverte",
    "water": "Échanges d'eau",
    "temperature": "Températures",
    "reliability": "Fiabilité",
    "raw_results": "Accès bruts aux résultats",
}
MODES = {"D": "direct", "M": "mcmc"}


def _type(schema: dict) -> str:
    if "anyOf" in schema:
        return " ou ".join(_type(option) for option in schema["anyOf"])
    kind = schema.get("type", "any")
    if kind == "array":
        if "prefixItems" in schema:
            return "[" + ", ".join(_type(item) for item in schema["prefixItems"]) + "]"
        return f"list[{_type(schema.get('items', {}))}]"
    if schema.get("format") == "date-time":
        return "date"
    return {"number": "nombre", "integer": "entier", "boolean": "booléen", "string": "texte", "null": "null"}.get(kind, kind)


def _bounds(schema: dict) -> str:
    signs = (("exclusiveMinimum", ">"), ("minimum", "≥"), ("exclusiveMaximum", "<"), ("maximum", "≤"))
    found = [f"{sign} {schema[key]:g}" for key, sign in signs if key in schema]
    for option in schema.get("anyOf", []):
        found += [f"{sign} {option[key]:g}" for key, sign in signs if key in option]
    return ", ".join(found)


def _parameters(question) -> list[str]:
    schema = question.params.model_json_schema()
    properties = schema.get("properties", {})
    if not properties:
        return ["Aucun paramètre : le corps de la requête est vide (ou `{}`)."]
    lines = ["| Paramètre | Type | Défaut | Bornes | Description |", "|---|---|---|---|---|"]
    for name, field in properties.items():
        default = field.get("default", "")
        default = "—" if default is None else f"`{default}`" if default != "" else "requis"
        lines.append(f"| `{name}` | {_type(field)} | {default} | {_bounds(field)} | {field.get('description', '')} |")
    return lines


def render() -> str:
    sys.path.insert(0, str(WORKSPACE / "backend" / "src"))
    from molonari_twin.api import CATALOG
    from molonari_twin.contract import API_VERSION
    from molonari_twin.errors import MESSAGES
    from molonari_twin.i18n import load_catalog
    from molonari_twin.server.app import HTTP_STATUS, http_status

    french = load_catalog("fr")
    lines = [
        "# API /v1 de molonari-twin : référence des questions",
        "",
        "<!-- Fichier généré par tools/gen_api_doc.py à partir du catalogue (molonari_twin.api). Ne pas modifier à la main. -->",
        "",
        f"Version de l'API : **{API_VERSION}**. Schéma complet : `GET /openapi.json` ; catalogue avec les schémas de",
        "paramètres : `GET /v1/questions` (`?twin=<ref>` : les questions de son mode).",
        "",
        "## Poser une question",
        "",
        "```bash",
        "molonari-twin serve --local --twins-folder examples/twins",
        "curl -X POST http://127.0.0.1:8765/v1/twins/twin_03/questions/exchanged_volume \\",
        "     -H 'Content-Type: application/json' -d '{\"start\": \"2016-06-28T00:00:00\", \"area\": 2.0}'",
        "curl -X POST http://127.0.0.1:8765/v1/twins/twin_03/questions/HN-12 -H 'Accept: application/x-npz' -o answer.npz",
        "```",
        "",
        "Une question se désigne par son identifiant (`exchanged_volume`) ou par sa référence du plan (`HN-12`). Le",
        "corps de la requête contient ses paramètres ; un paramètre inconnu est refusé. Les questions lisent la",
        "configuration ou les résultats d'un twin : elles répondent tout de suite, sans tâche et sans charger le moteur.",
        "Calculer un twin passe par `POST /v1/twins/{ref}/run`, puis par le suivi de la tâche (`/v1/tasks/…`).",
        "",
        "## Réponse",
        "",
        "`{data: {nom: {value, unit, interval}}, provenance, warnings}`. `interval` vaut `[bas, haut]`, de la forme de",
        "`value`, ou `null`. La provenance donne la question, ses paramètres, le twin, son `twin_id` et ses résultats.",
        "",
        "- **JSON** (par défaut) : un tableau s'écrit `{dtype, shape, data}`, valeurs à plat dans l'ordre C. NaN et",
        "  dates absentes valent `null`, ±∞ s'écrivent `\"inf\"` et `\"-inf\"`. Un flottant scalaire non fini s'écrit",
        "  comme un tableau de forme `[]`.",
        "- **npz** (`Accept: application/x-npz`) : chaque tableau va dans l'archive (`data/<nom>/value`, `…/low`,",
        "  `…/high`) ; la réponse elle-même est l'entrée `meta` (JSON). L'archive se lit sans pickle.",
        "- **Dates** : ISO 8601 à la seconde, **sans fuseau**, dans le repère des données du twin. Un fuseau donné en",
        "  paramètre est refusé tant que la convention de dates n'est pas fixée (avec le pipeline brut).",
        "- **Unités** : SI (m/s, m³/m², °C, m, s). Les conversions d'affichage (m/jour, L/m²) sont faites par le client.",
        "- **Tableaux aux capteurs** : capteur d'abord, `(3, T)`.",
        "",
        "## Erreurs",
        "",
        "Toute erreur attendue a la forme `{code, field, params, message_en}` ; le client traduit à partir de `code`",
        "et de `params`.",
        "",
        "| Code | Statut HTTP |",
        "|---|---|",
    ]
    codes = sorted(set(HTTP_STATUS) | {c for c in MESSAGES if c.startswith("E_QUESTION_")} | {"E_REQUEST_INVALID"})
    lines += [f"| `{code}` | {422 if code == 'E_REQUEST_INVALID' else http_status(code)} |" for code in codes]
    lines += ["", "Les autres codes `E_…` répondent 400.", ""]
    for category, title in CATEGORIES.items():
        questions = [q for q in CATALOG if q.category == category]
        if not questions:
            continue
        lines += [f"## {title}", ""]
        for question in questions:
            lines += [
                f"### `{question.id}` ({question.ref})",
                "",
                f"**{question.label}** — *{french.get(question.label, question.label)}*",
                "",
                question.description,
                "",
                f"- Modes : {', '.join(MODES[m] for m in question.modes)} ; depuis l'API {question.since} ; exige "
                + ("la configuration du twin." if question.requires == "config" else "ses résultats (twin calculé)."),
                f"- Requête : `POST /v1/twins/{{ref}}/questions/{question.id}`",
                "",
                *_parameters(question),
                "",
                "Sorties : " + ", ".join(f"`{name}`" for name in question.outputs) + ".",
                "",
            ]
    return "\n".join(lines).rstrip("\n") + "\n"


def main(argv: list[str]) -> int:
    text = render()
    if "--check" in argv:
        current = OUTPUT.read_text(encoding="utf-8") if OUTPUT.is_file() else ""
        if current != text:
            print(f"{OUTPUT.relative_to(WORKSPACE)} n'est pas à jour : python tools/gen_api_doc.py", file=sys.stderr)
            return 1
        return 0
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(text, encoding="utf-8")
    print(f"écrit : {OUTPUT.relative_to(WORKSPACE)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
