"""Générer un état par le modèle direct (niveau user) et écrire `generation.yaml` (OPX).

    molonari-twin generate <point> [--sample <chaîne>:<itération> | --params <fichier>] [--output <dossier>]

`<point>` est le dossier du sampling point dans MOLONARI-TWIN :

    <point>/public_reports/chains.npz      le sous-échantillon publié des chaînes (T2)
    <point>/twins/<twin direct>/           twin.yaml (les données) et generation.yaml (réglages de génération)

`generation.yaml` est écrit par l'OPX à la publication (`write_generation`), dans le twin direct du rapport. Il donne
la discrétisation de l'**inversion** (couches, `nb_cells`, `initial_condition`) et les bornes des paramètres : un
échantillon n'est cohérent qu'avec elle, alors que le twin direct a sa propre discrétisation. Les bornes sont au
niveau user et absentes du rapport public.

L'état généré a le format des résultats d'un twin direct (`twin/results.py`), sans `twin.yaml` : `ask` le lit comme tel
(`TwinAPI.ask_state`). Il reste local : il n'entre ni dans `public_reports/` ni dans `results/`, et n'est pas inscrit au
registre. Par défaut il part de l'échantillon publié de plus basse énergie (déterministe) ; `--sample` en désigne un
autre ; `--params` explore ses propres paramètres, dans les bornes, et marque l'état « paramètres libres ».

Le calcul ne charge JAX qu'après avoir lu et validé tout le reste (échantillon, bornes, sortie).
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from ..domain.compute import INVERTED_PARAMETERS, ForwardConfig, LayerSpec
from ..errors import Issue, UserError
from .config import TwinConfig, file_digest, load_twin
from .registry import RESULTS_FOLDER
from .results import RESULTS_FORMAT, record_file_digests, write_results

GENERATION_FILE = "generation.yaml"
GENERATION_SCHEMA = "molonari-generation/1.0.0"
CHAINS_FILE = "chains.npz"
PUBLIC_REPORTS = "public_reports"
TWINS = "twins"
STATE_KEY = "generated"  # clé de metadata.json qui désigne un état généré
STATE_FORMAT = "1.0.0"


# generation.yaml ---------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Generation:
    """Réglages de génération : la discrétisation de l'inversion et les bornes des paramètres, par couche."""

    layers: tuple[tuple[str, float], ...]  # (nom, profondeur du bas)
    nb_cells: int
    initial_condition: str
    bounds: tuple[dict[str, tuple[float, float]], ...]  # par couche : paramètre → (min, max)


def generation_document(config: TwinConfig) -> dict:
    """Contenu de `generation.yaml` pour le twin mcmc `config` (déterministe)."""
    mcmc = config.inversion_config()
    return {
        "schema": GENERATION_SCHEMA,
        "nb_cells": int(mcmc.nb_cells),
        "initial_condition": config.initial_condition,
        "layers": [
            {"name": layer.name, "z_low": float(layer.z_low),
             "bounds": {name: [float(priors[name].low), float(priors[name].high)] for name in INVERTED_PARAMETERS}}
            for layer, priors in zip(mcmc.layers, mcmc.priors, strict=True)
        ],
    }


def write_generation(inversion: TwinConfig, forward_folder: Path) -> Path:
    """Écrit `generation.yaml` dans le dossier du twin direct du rapport, d'après le twin mcmc (OPX, à la publication)."""
    path = Path(forward_folder) / GENERATION_FILE
    path.write_text(yaml.safe_dump(generation_document(inversion), sort_keys=False, allow_unicode=True), encoding="utf-8")
    return path


def _invalid(path: Path, detail: str) -> UserError:
    return UserError("E_GENERATE_SETTINGS_INVALID", path=str(path), detail=detail)


def read_generation(path: Path) -> Generation:
    """Relit `generation.yaml` ; E_GENERATE_SETTINGS_INVALID si le fichier n'a pas la forme attendue."""
    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise _invalid(path, str(error)) from None
    if not isinstance(document, dict) or document.get("schema") != GENERATION_SCHEMA:
        raise _invalid(path, f"schema: {GENERATION_SCHEMA}")
    layers, bounds = [], []
    raw_layers = document.get("layers")
    if not isinstance(raw_layers, list) or not raw_layers:
        raise _invalid(path, "layers")
    for i, layer in enumerate(raw_layers):
        try:
            layers.append((str(layer["name"]), float(layer["z_low"])))
            bounds.append({name: (float(layer["bounds"][name][0]), float(layer["bounds"][name][1])) for name in INVERTED_PARAMETERS})
        except (KeyError, TypeError, ValueError, IndexError):
            raise _invalid(path, f"layers[{i}]") from None
    nb_cells, initial = document.get("nb_cells"), document.get("initial_condition")
    if isinstance(nb_cells, bool) or not isinstance(nb_cells, int) or nb_cells < 1:
        raise _invalid(path, "nb_cells")
    if not isinstance(initial, str):
        raise _invalid(path, "initial_condition")
    return Generation(tuple(layers), nb_cells, initial, tuple(bounds))


# Échantillons publiés ----------------------------------------------------------------------------------------------


def read_chains(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def parse_sample(text: str) -> tuple[int, int]:
    """« chaîne:itération » → (chaîne, itération) ; E_GENERATE_SAMPLE_FORMAT sinon."""
    parts = text.split(":")
    try:
        if len(parts) != 2:
            raise ValueError
        return int(parts[0]), int(parts[1])
    except ValueError:
        raise UserError("E_GENERATE_SAMPLE_FORMAT", value=text) from None


def lowest_energy_sample(chains: dict[str, np.ndarray]) -> tuple[int, int]:
    """(chaîne, itération d'origine) du plus bas `energy` publié. Une égalité est tranchée par le premier indice
    (itération, puis chaîne) : le choix ne dépend que du fichier."""
    energy = np.asarray(chains["energy"], float)
    row, column = np.unravel_index(int(np.argmin(np.where(np.isnan(energy), np.inf, energy))), energy.shape)
    return int(chains["chain"][column]), int(chains["iteration"][row])


def sample_parameters(chains: dict[str, np.ndarray], chain: int, iteration: int, nb_layers: int) -> list[list[float]]:
    """Paramètres (couches × 4, espace des priors) de l'échantillon publié `chain`:`iteration` ; E_GENERATE_SAMPLE_UNKNOWN."""
    rows = np.flatnonzero(np.asarray(chains["iteration"]) == iteration)
    columns = np.flatnonzero(np.asarray(chains["chain"]) == chain)
    if not rows.size or not columns.size:
        raise UserError("E_GENERATE_SAMPLE_UNKNOWN", chain=chain, iteration=iteration, chains=int(chains["chain"][-1]),
                        first=int(chains["iteration"][0]), last=int(chains["iteration"][-1]), step=_step(chains["iteration"]))
    stacked = np.stack([np.asarray(chains[f"samples.{name}"])[rows[0], columns[0]] for name in INVERTED_PARAMETERS], axis=-1)
    if stacked.shape != (nb_layers, len(INVERTED_PARAMETERS)):
        raise UserError("E_GENERATE_SETTINGS_INVALID", path=GENERATION_FILE, detail=f"layers ({stacked.shape[0]} in {CHAINS_FILE})")
    return stacked.tolist()


def _step(iterations) -> int:
    values = np.asarray(iterations)
    return int(values[1] - values[0]) if len(values) > 1 else int(values[0])


# Paramètres libres -------------------------------------------------------------------------------------------------


def read_free_parameters(path: Path, nb_layers: int) -> list[list[float]]:
    """Paramètres d'un fichier YAML ou JSON : `layers: [{neg_log10_k, porosity, lambda_s, rhos_cs}, …]` (ou la liste seule),
    une entrée par couche de `generation.yaml`, dans l'ordre. E_GENERATE_PARAMS_FILE, E_GENERATE_PARAMS_INVALID."""
    try:
        document = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise UserError("E_GENERATE_PARAMS_FILE", path=str(path), detail=str(error)) from None
    layers = document.get("layers") if isinstance(document, dict) else document
    problems: list[Issue] = []
    if not isinstance(layers, list) or len(layers) != nb_layers:
        raise UserError("E_GENERATE_PARAMS_INVALID", issues=[Issue("layers", "E_GENERATE_PARAMS_LAYERS", prefixed=True, n=nb_layers)])
    values: list[list[float]] = []
    for i, layer in enumerate(layers):
        row = []
        for name in INVERTED_PARAMETERS:
            value = layer.get(name) if isinstance(layer, dict) else None
            if isinstance(value, bool) or not isinstance(value, int | float) or not np.isfinite(value):
                problems.append(Issue(f"layers[{i}].{name}", "E_GENERATE_PARAMS_NUMBER", prefixed=True))
                value = float("nan")
            row.append(float(value))
        values.append(row)
    if problems:
        raise UserError("E_GENERATE_PARAMS_INVALID", issues=problems)
    return values


def check_bounds(values: list[list[float]], generation: Generation) -> None:
    """E_GENERATE_PARAMS_OUT_OF_BOUNDS, avec un problème par valeur hors des bornes de `generation.yaml`."""
    problems = []
    for i, (row, bounds) in enumerate(zip(values, generation.bounds, strict=True)):
        for name, value in zip(INVERTED_PARAMETERS, row, strict=True):
            low, high = bounds[name]
            if not low <= value <= high:
                problems.append(Issue(f"layers[{i}].{name}", "E_GENERATE_BOUND", prefixed=True, value=value, low=low, high=high))
    if problems:
        raise UserError("E_GENERATE_PARAMS_OUT_OF_BOUNDS", issues=problems)


# Disposition du point ----------------------------------------------------------------------------------------------


def locate(point: str | Path) -> tuple[Path, Path, Path]:
    """(dossier du point, twin direct qui porte `generation.yaml`, `chains.npz`) ; E_GENERATE_NO_SETTINGS, E_GENERATE_NO_CHAINS."""
    root = Path(point).resolve()
    if not root.is_dir():
        raise UserError("E_FOLDER_NOT_FOUND", path=str(point))
    carriers = sorted(p.parent for p in (root / TWINS).glob(f"*/{GENERATION_FILE}") if (p.parent / "twin.yaml").is_file())
    if not carriers:
        raise UserError("E_GENERATE_NO_SETTINGS", file=GENERATION_FILE, folder=str(root / TWINS))
    if len(carriers) > 1:
        raise UserError("E_GENERATE_SETTINGS_AMBIGUOUS", file=GENERATION_FILE, twins=", ".join(c.name for c in carriers))
    chains = root / PUBLIC_REPORTS / CHAINS_FILE
    if not chains.is_file():
        raise UserError("E_GENERATE_NO_CHAINS", point=root.name, path=str(chains))
    return root, carriers[0], chains


def check_output(output: Path, point: Path) -> None:
    """E_GENERATE_OUTPUT_FORBIDDEN dans `public_reports/` ou `results/` du point ; E_GENERATE_OUTPUT_EXISTS sur un dossier
    qui n'est pas un état généré (un état généré se réécrit)."""
    for forbidden in (point / PUBLIC_REPORTS, point / RESULTS_FOLDER):
        if output == forbidden.resolve() or forbidden.resolve() in output.parents:
            raise UserError("E_GENERATE_OUTPUT_FORBIDDEN", path=str(output), folder=forbidden.name)
    if output.exists() and not (output.is_dir() and is_state(output)):
        raise UserError("E_GENERATE_OUTPUT_EXISTS", path=str(output))


def is_state(folder: Path) -> bool:
    """Vrai pour un dossier d'état généré : `metadata.json` avec la clé `generated`, et pas de `twin.yaml`."""
    metadata = Path(folder) / "metadata.json"
    if not metadata.is_file() or (Path(folder) / "twin.yaml").is_file():
        return False
    try:
        return STATE_KEY in json.loads(metadata.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False


# Génération --------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class GeneratedState:
    folder: Path
    free_parameters: bool
    chain: int | None  # l'échantillon publié d'où part l'état ; None pour des paramètres libres
    iteration: int | None
    parameters: list[list[float]]
    rmse_total: float


def generate_state(point: str | Path, sample: str | None = None, params: str | Path | None = None,
                   output: str | Path | None = None) -> GeneratedState:
    """Calcule l'état du point `point` par le modèle direct et l'écrit dans `output` (par défaut
    `./<point>_state`). Sans option : l'échantillon publié de plus basse énergie."""
    if sample is not None and params is not None:
        raise UserError("E_GENERATE_SAMPLE_AND_PARAMS")
    root, carrier, chains_path = locate(point)
    generation = read_generation(carrier / GENERATION_FILE)
    target = (Path(output) if output is not None else Path.cwd() / f"{root.name}_state").resolve()
    check_output(target, root)
    chains = read_chains(chains_path)
    nb_layers = len(generation.layers)
    if params is not None:
        values = read_free_parameters(Path(params), nb_layers)
        check_bounds(values, generation)
        chain = iteration = None
    else:
        chain, iteration = parse_sample(sample) if sample is not None else lowest_energy_sample(chains)
        values = sample_parameters(chains, chain, iteration, nb_layers)
    config = load_twin(carrier)
    if config.mode != "direct":
        raise UserError("E_GENERATE_NOT_DIRECT", folder=str(carrier))

    from ..engine import api  # JAX n'est chargé que par le calcul

    layers = [LayerSpec(name, z_low, *row) for (name, z_low), row in zip(generation.layers, values, strict=True)]
    forward = api.run_forward(config.column_inputs(), ForwardConfig(layers, generation.nb_cells))
    metadata = _metadata(config, generation, layers, forward, chain, iteration, file_digest(chains_path), api.engine_version())

    partial = target.with_name(target.name + ".partial")
    if partial.exists():
        shutil.rmtree(partial)
    write_results(partial, forward, None, [layer.name for layer in layers], metadata)
    record_file_digests(partial)
    if target.exists():
        shutil.rmtree(target)
    partial.replace(target)
    return GeneratedState(target, params is not None, chain, iteration, values, float(forward.rmse[3]))


def _metadata(config: TwinConfig, generation: Generation, layers: list[LayerSpec], forward, chain, iteration, chains_sha256: str,
              code_version: str) -> dict:
    """metadata.json de l'état : le format d'un twin direct, sans date ni chemin (deux générations donnent les mêmes octets),
    et la clé `generated` qui dit d'où viennent les paramètres. Ni `twin_id` ni graine."""
    from ..engine.jax.forward import HEAT_REFERENCE_C

    free = chain is None
    return {
        "name": f"{config.point or config.name}_state",
        "point": config.point,
        "mode": "direct",
        "experimental": False,
        "code_version": code_version,
        "results_format": RESULTS_FORMAT,
        "rmse_celsius": {"t1": float(forward.rmse[0]), "t2": float(forward.rmse[1]), "t3": float(forward.rmse[2]), "total": float(forward.rmse[3])},
        "sensor_cell_depths": [float(d) for d in forward.sensor_depths],
        "flux_convention": "q > 0: infiltration (river to aquifer); q < 0: exfiltration (aquifer to river)",
        "conventions": {
            "z": "positive downwards, m",
            "q": "positive downwards (infiltration), m/s",
            "heat_flux": "positive downwards, W/m2",
            "advective_reference_C": HEAT_REFERENCE_C,
        },
        STATE_KEY: {
            "format": STATE_FORMAT,
            "free_parameters": free,
            "sample": None if free else {"chain": chain, "iteration": iteration},
            "chains_sha256": None if free else chains_sha256,
            "nb_cells": generation.nb_cells,
            "layers": [{"name": layer.name, "z_low": layer.z_low,
                        **{name: float(getattr(layer, name)) for name in INVERTED_PARAMETERS}} for layer in layers],
        },
    }
