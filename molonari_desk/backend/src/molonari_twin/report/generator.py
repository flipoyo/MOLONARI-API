"""Générateur du rapport public d'un space (P2c.3) : écrit `Public Report/` à partir de twins calculés.

    result = generate_report("twins/twin_03")        # écrit twins/twin_03/Public Report/
    report = build_report(resolve_space("twins/twin_03"), TwinAPI("twins"))   # en mémoire, sans rien écrire

Le **space** (provisoire, jusqu'aux operating spaces, P2b.1) est la ou les instances que désigne la clé `report` du twin
donné : `report.models.forward` (un twin direct) et `report.models.inversion` (un twin mcmc), des dossiers frères du twin.
Sans `report.models`, le space est le twin donné seul. La clé `report.entries` (la grille fixe des entrées, ou le rapport
par défaut d'un twin qui n'a pas la clé) est celle du twin donné.

Pour un space de deux instances, `generate_report` écrit aussi `generation.yaml` dans le dossier du twin direct (couches et
discrétisation de l'inversion, bornes des paramètres : twin/generate.py). Il n'entre pas dans `Public Report/`.

Le générateur tourne au niveau **dev** (P2c.3i) : il ne lit ni `orx.yaml` ni la graine, et pose ses questions par `TwinAPI`,
qui lit les résultats sans JAX. Il ne lit directement que `metadata.json`, pour l'index.

Entrées, dans l'ordre fixe (qui fixe les noms de fichiers, l'ordre de l'index, donc les octets) :
- obligatoires : pour chaque modèle `describe`, `provenance` et les `report.entries` (le flux d'eau et la qualité de
  l'ajustement au minimum) ; la convergence publique pour l'inversion ; `compare` (différence forward − inversion) si les
  deux modèles sont là. Le refus d'une entrée obligatoire fait échouer le rapport (rien n'est écrit) ;
- par défaut : les propriétés du sol (HN-42) pour chaque modèle dont l'élément est public, et, pour l'inversion, les
  échantillons (BN-32) et les données par chaîne (BN-34) si `posterior` est public. Une entrée refusée est notée dans
  `skipped`, avec son code.

La visibilité appliquée est la plus stricte des twins du space : un élément n'est public que s'il l'est dans chaque twin
auquel il s'applique. Un rapport sans aucune entrée est refusé (E_REPORT_EMPTY).
"""

from __future__ import annotations

import dataclasses
import io
import json
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..api import TwinAPI, check_publishable
from ..errors import Issue, UserError
from ..questions import ELEMENTS, Answer, Visibility, public_answer
from ..questions.chains import public_chains_npz
from ..questions.visibility import ELEMENT_MODES
from ..twin.config import ReportEntry, TwinConfig, config_path, load_twin, without_seed
from .figures import FIGURES
from .writer import (
    FOLDER_NAME,
    INDEX_NAME,
    NPZ_THRESHOLD,
    dump_json,
    encode_entry,
    sha256,
    write_folder,
)

REPORT_FORMAT = "1.0.0"
PUBLIC_REPORTS = "public_reports"
ROLES = ("forward", "inversion")
MODE_OF_ROLE = {"forward": "direct", "inversion": "mcmc"}
ROLE_OF_MODE = {mode: role for role, mode in MODE_OF_ROLE.items()}
LETTER = {"direct": "D", "mcmc": "M"}
UNCERTAINTY = {"forward": "none", "inversion": "posterior"}
BOTH = "both"
COMPARE = "compare"

AGGREGATES = frozenset({"fit_quality"})
"""Questions obligatoires dont la projection ne garde que des agrégats (RMSE, biais) : elles sont publiées même si
`raw_data` et `processed_data` sont privés, bien que le catalogue les déclare « rapport si public » (SPEC_RAPPORT_PUBLIC,
section 4 : la RMSE est une information obligatoire, et un agrégat n'est pas une fuite)."""

OPTIONAL = {"forward": ("soil_properties",), "inversion": ("soil_properties", "posterior_samples", "chain_data")}
"""Entrées par défaut de chaque modèle, publiées quand la visibilité le permet."""


@dataclass(frozen=True)
class Space:
    """Les instances du rapport : rôle (forward, inversion) → dossier de twin, et leur configuration."""

    carrier: str  # le twin dont on génère le rapport, et dont le dossier reçoit `Public Report/`
    refs: dict[str, str]
    configs: dict[str, TwinConfig]
    entries: tuple[ReportEntry, ...]
    declared: bool  # False : le twin n'a pas de clé `report`, c'est le rapport par défaut
    publication_thin: int = 10  # facteur de sous-échantillonnage pour les chaînes publiques (T2)
    local: bool = False  # rapport local d'un dev (`--forward`) : l'index le dit, ce n'est pas une publication (T7)


@dataclass(frozen=True)
class Item:
    question: str  # identifiant du catalogue, ou « compare »
    role: str  # forward, inversion, ou « both »
    parameters: dict
    required: bool


@dataclass
class Report:
    index: dict
    files: dict[str, bytes]  # chemin relatif dans `Public Report/` → octets (index compris)
    reasons: list[UserError]  # le motif de chaque entrée de `index["skipped"]`, dans le même ordre (pour l'afficher traduit)


@dataclass(frozen=True)
class ReportResult:
    folder: Path
    entries: int
    figures: int
    skipped: list[dict]
    reasons: list[UserError]


# Space -------------------------------------------------------------------------------------------------------------


def resolve_space(twin: str | Path, forward: str | Path | None = None) -> Space:
    """Le space du rapport du twin `twin` (dossier ou twin.yaml) : E_REPORT_MODEL_MODE si un twin désigné n'a pas le bon
    mode, E_REPORT_POINT_MISMATCH si les twins ne portent pas sur le même point, E_TWIN_NOT_FOUND s'il n'existe pas.
    `forward` (T7) : dossier du twin direct, donné explicitement à la place du frère nommé par `report.models` ; `twin` est alors
    l'inversion, par exemple un twin de σ hors du point. Le space est marqué local."""
    folder = config_path(twin).parent
    config = load_twin(folder)
    spec = config.report
    other: dict[str, Path] = {}
    if forward is not None:
        other["forward"] = config_path(forward).parent
        if other["forward"].name == folder.name:
            raise UserError("E_REPORT_FORWARD_SAME", ref=folder.name)
        refs = {"forward": other["forward"].name, "inversion": folder.name}
    elif spec.forward or spec.inversion:
        refs = {role: name for role, name in (("forward", spec.forward), ("inversion", spec.inversion)) if name}
    else:
        refs = {ROLE_OF_MODE[config.mode]: folder.name}
    api = TwinAPI(folder.parent)
    configs = {}
    for role, ref in refs.items():
        configs[role] = config if ref == folder.name else load_twin(other.get(role) or api.twin_folder(ref))
        if configs[role].mode != MODE_OF_ROLE[role]:
            raise UserError("E_REPORT_MODEL_MODE", ref=ref, mode=configs[role].mode, role=role, expected=MODE_OF_ROLE[role])
    points = sorted({c.point for c in configs.values()})
    if len(points) > 1:  # un rapport public appartient à un seul sampling point
        raise UserError("E_REPORT_POINT_MISMATCH", points=", ".join(points))
    return Space(folder.name, refs, configs, spec.entries, spec.declared, spec.publication_thin, local=forward is not None)


def effective_visibility(configs) -> Visibility:
    """La visibilité la plus stricte : un élément est public s'il l'est dans chaque twin auquel il s'applique (la PDF
    n'existe que pour un twin mcmc, les paramètres écrits que pour un twin direct)."""
    configs = list(configs)
    values = {}
    for element in ELEMENTS:
        mode = ELEMENT_MODES.get(element)
        concerned = [c for c in configs if mode is None or LETTER[c.mode] == mode] or configs
        values[element] = all(bool(c.visibility[element]) for c in concerned)
    return Visibility(**values)


# Plan --------------------------------------------------------------------------------------------------------------


def plan(space: Space) -> list[Item]:
    """La liste ordonnée des entrées à produire. Une entrée déclarée qui est aussi une entrée fixe n'est produite qu'une fois."""
    items: list[Item] = []
    for role in space.refs:
        items += [Item("describe", role, {}, True), Item("provenance", role, {}, True)]
        items += [Item(entry.question, role, dict(entry.parameters), True) for entry in space.entries]
        if role == "inversion":
            items.append(Item("public_convergence", role, {}, True))
    if len(space.refs) == len(ROLES):
        items.append(Item(COMPARE, BOTH, {}, True))
    items += [Item(question, role, {}, False) for role in space.refs for question in OPTIONAL[role]]
    seen: set[tuple[str, str, str]] = set()
    unique = []
    for item in items:
        key = (item.role, item.question, json.dumps(item.parameters, sort_keys=True))
        if key not in seen:
            seen.add(key)
            unique.append(item)
    return unique


# Génération --------------------------------------------------------------------------------------------------------


def _metadata(api: TwinAPI, ref: str, config: TwinConfig) -> dict:
    from ..twin.run import twin_status

    status = twin_status(config)
    if not status.has_results:
        raise UserError("E_TWIN_NOT_COMPUTED", ref=ref)
    return json.loads((status.results_dir / "metadata.json").read_text(encoding="utf-8"))


def _space_entry(ref: str, metadata: dict) -> dict:
    """Le twin du space dans l'index : son dossier, l'empreinte de sa recette dev (sans graine, sans sel : T12) et son commit."""
    recipe = json.dumps(without_seed(metadata.get("config", {})), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return {"twin": ref, "recipe_sha256": sha256(recipe.encode("utf-8")), "commit": ((metadata.get("provenance") or {}).get("twin") or {}).get("commit")}


def _platform(metadata: dict) -> dict:
    from ..questions.public import PUBLIC_ENVIRONMENT

    environment = ((metadata.get("provenance") or {}).get("environment")) or {}
    return {name: str(environment[name]) for name in PUBLIC_ENVIRONMENT if environment.get(name) is not None}


def _skipped(item: Item, error: UserError) -> dict:
    """Une entrée sautée et son motif (code et paramètres du message), tels que l'index les recopie."""
    return {"question": item.question, "model": item.role, "code": error.code, "params": {k: str(v) for k, v in error.params.items()}}


def _label(answer: Answer, role: str) -> Answer:
    """Étiquette de modèle : un résultat forward (ou la comparaison) n'a jamais d'intervalle, et le dit (W_NO_UNCERTAINTY)."""
    if role == "inversion":
        return answer
    if any(quantity.interval is not None for quantity in answer.data.values()):
        raise RuntimeError(f"a forward answer carries an interval: {answer.provenance.get('question')}")
    warnings = list(answer.warnings)
    if not any(w.code == "W_NO_UNCERTAINTY" for w in warnings):
        warnings.append(Issue(None, "W_NO_UNCERTAINTY"))
    return dataclasses.replace(answer, warnings=warnings)


def _answer(api: TwinAPI, space: Space, visibility: Visibility, item: Item) -> tuple[Answer, str, str | None]:
    """(réponse publiable, identifiant de la question, référence du plan) ; UserError si la question est refusée."""
    if item.role == BOTH:
        forward, inversion = space.refs["forward"], space.refs["inversion"]
        answer = public_answer("model_difference", api.model_difference(forward, inversion), f"{forward}+{inversion}")
        return dataclasses.replace(answer, provenance={**answer.provenance, "question": COMPARE, "parameters": {}}), COMPARE, None
    question = api.question(item.question)
    arguments = api.parameters(question, item.parameters)
    mode = space.configs[item.role].mode
    if question.id not in AGGREGATES:
        check_publishable(question, visibility, mode)
    answer = api.ask(space.refs[item.role], question.id, item.parameters)
    answer = public_answer(question.id, answer, space.refs[item.role])
    # Les paramètres publiés sont ceux de la demande : ceux de la réponse peuvent contenir un réglage privé (σ).
    requested = arguments.model_dump(mode="json", exclude_unset=True)
    return dataclasses.replace(answer, provenance={**answer.provenance, "parameters": requested}), question.id, question.ref


def build_report(space: Space, api: TwinAPI, npz_threshold: int = NPZ_THRESHOLD) -> Report:
    """Calcule le rapport en mémoire (aucun fichier écrit). Échoue sans rien produire si une entrée obligatoire est refusée.
    `api` est le `TwinAPI` du dossier qui contient les twins du space."""
    metadata = {role: _metadata(api, ref, space.configs[role]) for role, ref in space.refs.items()}
    visibility = effective_visibility(space.configs.values())

    entries: list[dict] = []
    figures: list[dict] = []
    skipped: list[dict] = []
    reasons: list[UserError] = []
    files: dict[str, bytes] = {}
    used: dict[tuple[str, str], int] = {}
    for item in plan(space):
        if item.role != BOTH and item.question != COMPARE:
            question = api.question(item.question)
            if LETTER[MODE_OF_ROLE[item.role]] not in question.modes:
                error = UserError("E_QUESTION_NOT_AVAILABLE", question=item.question, ref=space.refs[item.role], mode=MODE_OF_ROLE[item.role])
                skipped.append(_skipped(item, error))
                reasons.append(error)
                continue
        try:
            answer, question_id, ref = _answer(api, space, visibility, item)
        except UserError as error:
            if item.required:
                raise
            skipped.append(_skipped(item, error))
            reasons.append(error)
            continue
        answer = _label(answer, item.role)
        extension, content = encode_entry(question_id, answer, npz_threshold)
        count = used[(item.role, question_id)] = used.get((item.role, question_id), 0) + 1
        stem = COMPARE if item.role == BOTH else f"{item.role}-{question_id}"
        stem = stem if count == 1 else f"{stem}-{count}"
        name = f"{stem}.{extension}"
        files[name] = content
        entries.append({
            "file": name, "format": extension, "question": question_id, "ref": ref, "model": item.role,
            "uncertainty": "none" if item.role == BOTH else UNCERTAINTY[item.role],
            "parameters": answer.provenance["parameters"], "sha256": sha256(content),
        })
        if question_id in FIGURES and item.role != BOTH:
            figure = f"figures/{stem}.svg"
            files[figure] = FIGURES[question_id](answer)
            figures.append({"file": figure, "from_entry": name, "sha256": sha256(files[figure])})
    if not entries:
        raise UserError("E_REPORT_EMPTY")

    # chains.npz public (T2) : un sous-échantillon des chaînes, sans champ T ni q, ni graine
    chains_info = None
    if "inversion" in space.refs and visibility.posterior:
        from ..twin.run import twin_status

        inv_config = space.configs["inversion"]
        inv_status = twin_status(inv_config)
        if inv_status.has_results and inv_config.mcmc is not None:
            posterior_path = inv_status.results_dir / "posterior.npz"
            posterior_sha = sha256(posterior_path.read_bytes()) if posterior_path.is_file() else None
            results = api._results(inv_status.results_dir)
            chains_dict = public_chains_npz(results, space.publication_thin, inv_config.mcmc.thin)
            buf = io.BytesIO()
            np.savez(buf, **chains_dict)
            chains_bytes = buf.getvalue()
            files["chains.npz"] = chains_bytes
            chains_info = {
                "file": "chains.npz",
                "publication_thin": space.publication_thin,
                "posterior_sha256": posterior_sha,
                "sha256": sha256(chains_bytes),
            }

    space_index = {role: (_space_entry(space.refs[role], metadata[role]) if role in space.refs else None) for role in ROLES}
    if space.local:  # le dev cite son twin d'inversion par son twin_id (il n'est dans aucun registre public)
        space_index["inversion"]["twin_id"] = metadata["inversion"]["twin_id"]
    index = {
        "report_format": REPORT_FORMAT,
        "point": next(iter(space.configs.values())).point,  # le sampling point du rapport, commun aux twins du space
        "space": space_index,
        "twin_semver": None,  # le SemVer composé du Twin attend P0.4
        "generated_at": max(str(m["created_at"]) for m in metadata.values()),  # date du calcul, jamais l'horloge : mêmes octets
        "generator_version": _generator_version(),
        "platform": {role: _platform(metadata[role]) for role in space.refs},
        "visibility": visibility.to_dict(),
        "entries": entries,
        "skipped": skipped,
        "figures": figures,
        "chains": chains_info,
        "signature": None,
        "license": None,
    }
    if space.local:
        index["local"] = True  # absente d'un rapport de l'OPX : ses octets ne changent pas
    files[INDEX_NAME] = dump_json(index)
    return Report(index, files, reasons)


def _generator_version() -> str:
    from .. import __version__

    return f"molonari-twin {__version__}"


def generate_report(twin: str | Path, output: str | Path | None = None, npz_threshold: int = NPZ_THRESHOLD,
                    forward: str | Path | None = None) -> ReportResult:
    """Génère le rapport du twin `twin` (dossier ou twin.yaml) dans `output` (par défaut `<twin>/Public Report`).

    Avec `forward` (dossier du twin direct), le rapport est **local** (T7) : celui d'un dev pour un twin de σ hors du point.
    Il n'écrit rien dans le twin direct (`generation.yaml` reste à l'OPX) et refuse une sortie dans `public_reports/`."""
    folder = config_path(twin).parent
    target = Path(output) if output is not None else folder / FOLDER_NAME
    if forward is not None and PUBLIC_REPORTS in target.resolve().parts:
        raise UserError("E_REPORT_LOCAL_OUTPUT_FORBIDDEN", path=str(target), folder=PUBLIC_REPORTS)
    space = resolve_space(folder, forward)
    if forward is None:
        report = build_report(space, TwinAPI(folder.parent), npz_threshold)
    else:
        # Le twin direct n'est pas un frère : un dossier d'appoint les nomme tous deux (liens), le temps du calcul.
        with tempfile.TemporaryDirectory() as staging:
            for role in ROLES:
                (Path(staging) / space.refs[role]).symlink_to(space.configs[role].folder.resolve(), target_is_directory=True)
            report = build_report(space, TwinAPI(staging), npz_threshold)
    write_folder(target, report.files)
    if forward is None and len(space.refs) == len(ROLES):
        # Réglages de génération (T4) : dans le twin direct (niveau user), jamais dans le rapport public
        from ..twin.generate import write_generation

        write_generation(space.configs["inversion"], space.configs["forward"].folder)
    return ReportResult(target, len(report.index["entries"]), len(report.index["figures"]), report.index["skipped"], report.reasons)
