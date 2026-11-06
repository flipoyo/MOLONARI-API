"""Cycle de vie d'un twin : valider, identifier, vérifier le registre, exécuter, ranger.

    1. l'utilisateur écrit un twin.yaml ;
    2. la config est validée (config.load_twin) ;
    3. le twin_id est calculé et le registre consulté (status) ;
    4. le twin est exécuté en mode direct ou mcmc (run_twin) ;
    5. les résultats sont rangés dans <dossier du twin>/results/<twin_id>/ et le registre du twin mis à jour ;
    6. une config identique relancée est signalée comme doublon : le résultat existant
       est renvoyé sans recalcul.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import yaml

from ..errors import Issue, UserError
from .config import (
    DATA_ROOT_FILE,
    TwinConfig,
    TwinConfigError,
    config_path,
    file_digest,
    frame_digest,
    load_twin,
    without_seed,
)
from .orx import ORX_NAME, ORX_SCHEMA, TWINS_FOLDER
from .provenance import provenance_warnings, strict_problems, twin_provenance
from .registry import (
    RegistryEntry,
    append_entry,
    computed_entry,
    entry_for_name,
    locate_results,
    read_registry,
    results_dir,
)
from .results import RESULTS_FORMAT, record_file_digests, write_results

TWIN_PREFIX = "twin_"
SIGMA_SUFFIX = "_sigmaT"  # twin_mcmc_sigmaT0.08 : twin_mcmc avec un autre σ de température
SIGMA_OPTION = "--sigma-temperature"


@dataclass
class TwinStatus:
    config: TwinConfig
    twin_id: str
    state: str  # « new », « computed » (déjà calculé), « duplicate » (même config qu'un autre twin), « modified »
    source: RegistryEntry | None = None  # exécution qui a produit les résultats
    previous_id: str = ""  # pour « modified » : identifiant enregistré pour ce nom
    results_path: Path | None = None  # où sont les résultats déjà calculés (None : ils iraient dans le dossier du twin)
    legacy_results: bool = False  # les résultats sont à l'ancien emplacement, `<point>/results/<twin_id>/`

    @property
    def results_dir(self) -> Path:
        return self.results_path or results_dir(self.config.folder, self.twin_id)

    @property
    def warnings(self) -> list[Issue]:
        """W_RESULTS_LEGACY_LOCATION si les résultats sont lus à l'ancien emplacement (twins de `examples/`, démo)."""
        return [Issue("results", "W_RESULTS_LEGACY_LOCATION", path=self.results_path)] if self.legacy_results else []

    @property
    def twins_folder(self) -> Path:
        return self.config.folder.parent

    @property
    def has_results(self) -> bool:
        return self.state in ("computed", "duplicate")


@dataclass
class TwinRun:
    status: TwinStatus
    computed: bool  # False : doublon, résultat existant renvoyé sans recalcul
    duration_s: float = 0.0
    warnings: list[Issue] = field(default_factory=list)  # provenance incomplète (W_PROVENANCE_*)


def twin_status(path_or_config, code_version: str | None = None) -> TwinStatus:
    from ..engine import api

    config = path_or_config if isinstance(path_or_config, TwinConfig) else load_twin(path_or_config)
    code_version = code_version or api.engine_version()
    entries = read_registry(config.folder)
    named = entry_for_name(entries, config.name)
    if not config.has_seed:
        return _status_without_seed(config, code_version, entries, named)
    twin_id = config.twin_id(code_version)
    if named is not None and named.twin_id != twin_id:
        return TwinStatus(config, twin_id, "modified", previous_id=named.twin_id)
    source = computed_entry(entries, twin_id)
    found = locate_results(config.folder, source) if source is not None else None
    if found is not None:
        return TwinStatus(config, twin_id, "computed" if source.name == config.name else "duplicate", source, results_path=found[0], legacy_results=found[1])
    return TwinStatus(config, twin_id, "new")


def _status_without_seed(config: TwinConfig, code_version: str, entries: list[RegistryEntry], named: RegistryEntry | None) -> TwinStatus:
    """État d'un twin 2.0.0 mcmc sans `orx.yaml` (lecture du dev). Le `twin_id` contient la graine : il n'est pas calculable
    ici, et c'est celui du registre qui sert. Le twin est « modified » si sa recette sans graine n'est plus celle que
    `metadata.json["config"]` a gardée ; « new » (`twin_id` vide) si le registre ne le connaît pas. Un changement de la
    seule graine ne se voit pas : seul ORX peut le voir."""
    if named is None:
        return TwinStatus(config, "", "new")
    located = locate_results(config.folder, named)
    try:
        stored = json.loads((located[0] / "metadata.json").read_text(encoding="utf-8")).get("config") if located else None
    except (OSError, ValueError):
        return TwinStatus(config, "", "new")
    if json.loads(json.dumps(config.canonical(code_version, with_seed=False))) != stored:
        return TwinStatus(config, "", "modified", previous_id=named.twin_id)
    source = computed_entry(entries, named.twin_id)
    if source is None:
        return TwinStatus(config, "", "new")
    return TwinStatus(config, named.twin_id, "computed" if source.name == config.name else "duplicate", source,
                      results_path=located[0], legacy_results=located[1])


def with_sigma_temperature(path: str | Path, sigma: float, space: str | Path | None = None, seed: int | None = None) -> Path:
    """Dossier du twin mcmc `path` avec σ de température = `sigma` (°C), à lancer à sa place.

    Une config est figée (E_TWIN_FROZEN) : le twin d'origine n'est pas touché. Le σ modifié devient un **twin de dev**
    (spec 4 niveaux §9) : le dossier `<twin>_sigmaT<sigma>` est créé dans `space` (l'operating space du dev, par défaut le
    dossier courant), jamais dans le dépôt du point ni dans `<point>/twins/`, où un dev ne peut pas committer. Il garde la
    recette, avec le σ et le nom changés, et **jamais la graine de référence** : ni `orx.yaml` ni `<point>/orx/<twin>.yaml` ne
    sont copiés. Il porte la graine du dev (`seed`, écrite dans son propre `orx.yaml`) ; sans graine, `run_twin` le refuse
    (E_TWIN_REQUIRED, `orx.mcmc.seed`). Ses résultats et son registre sont dans son dossier, comme pour tout twin. Avec le
    σ du twin, rend le twin lui-même."""
    config = load_twin(path)
    folder = config.folder
    problems = []
    if not sigma > 0 or not math.isfinite(sigma):
        problems.append(Issue(SIGMA_OPTION, "E_TWIN_POSITIVE", prefixed=True))
    if config.mode != "mcmc":
        problems.append(Issue(SIGMA_OPTION, "E_TWIN_MCMC_ONLY", prefixed=True))
    if not config.raw_data:
        problems.append(Issue(SIGMA_OPTION, "E_TWIN_SIGMA_SCHEMA", prefixed=True))
    if seed is not None and (isinstance(seed, bool) or seed < 0):
        problems.append(Issue("--seed", "E_TWIN_MINIMUM", prefixed=True, v=0))
    if problems:
        raise TwinConfigError(problems)
    if sigma == config.sigma_temperature:
        return folder
    space = Path(space or Path.cwd()).resolve()
    if _is_reference_repository(space, folder):
        raise TwinConfigError([Issue(SIGMA_OPTION, "E_TWIN_SIGMA_SPACE", prefixed=True, space=space)])
    target = space / f"{folder.name.split(SIGMA_SUFFIX)[0]}{SIGMA_SUFFIX}{sigma!r}"
    if target.exists():
        found = load_twin(target).sigma_temperature
        if found != sigma:
            raise TwinConfigError([Issue(SIGMA_OPTION, "E_TWIN_SIGMA_FOLDER", prefixed=True, folder=target, sigma=found)])
    else:
        recipe = yaml.safe_load(config.path.read_text(encoding="utf-8"))
        recipe["name"] = target.name
        recipe.pop("legacy_twin_id", None)  # l'identifiant 1.x est celui du twin d'origine
        recipe["sensors"] = {**(recipe.get("sensors") or {}), "sigma_temperature": sigma}
        models = (recipe.get("report") or {}).get("models") or {}
        if models.get("inversion") == folder.name:
            models["inversion"] = target.name
        target.mkdir(parents=True)
        (target / config.path.name).write_text(yaml.safe_dump(recipe, sort_keys=False, allow_unicode=True), encoding="utf-8")
        if (folder / "twin.cgs").is_file():
            shutil.copy2(folder / "twin.cgs", target / "twin.cgs")
    if config.data_root is not None:
        # Hors de l'espace cgs, le twin ne retrouverait pas les données : on lui donne la racine du twin source. Ce n'est pas
        # la recette (twin.yaml est inchangé) et la racine n'entre pas dans canonical() : le twin_id ne dépend pas du lieu.
        (target / DATA_ROOT_FILE).write_text(f"{config.data_root}\n", encoding="utf-8")
    if seed is not None:
        (target / ORX_NAME).write_text(f"schema: {ORX_SCHEMA}\nmcmc:\n  seed: {int(seed)}\n", encoding="utf-8")
    return target


def _is_reference_repository(space: Path, twin_folder: Path) -> bool:
    """Vrai si `space` est le dépôt du point du twin (ou son dossier `twins/`) : la graine de référence y vit, et le dev n'y
    committe pas. Un twin hors de la disposition `<point>/twins/<twin>/` (déjà un twin de dev) n'a pas de point à protéger."""
    twin_folder = twin_folder.resolve()
    return twin_folder.parent.name == TWINS_FOLDER and space in (twin_folder.parent, twin_folder.parent.parent)


def run_twin(path_or_config, on_progress: Callable | None = None, should_cancel: Callable[[], bool] | None = None,
             strict: bool = False) -> TwinRun:
    """Exécute un twin (ou renvoie le résultat existant). `strict` : refuse un code non publié ou des dépôts
    modifiés ou non tagués (E_STRICT_REFUSED, avec tous les motifs), avant tout calcul."""
    from .. import __version__
    from ..engine import api

    config = path_or_config if isinstance(path_or_config, TwinConfig) else load_twin(path_or_config)
    config.require_seed()  # calculer est une opération ORX : sans graine (orx.yaml), le twin se lit mais ne se lance pas
    status = twin_status(config)
    if status.state == "modified":
        raise UserError(
            "E_TWIN_FROZEN", name=config.name, previous=status.previous_id
        )
    provenance = None
    if strict:
        provenance = twin_provenance(config.folder, api.device())
        problems = strict_problems(api.engine_version(), __version__, provenance)
        if problems:
            raise UserError("E_STRICT_REFUSED", issues=problems)
    if status.has_results:
        entries = read_registry(config.folder)
        if status.state == "duplicate" and entry_for_name(entries, config.name) is None:
            append_entry(config.folder, RegistryEntry(
                status.twin_id, config.name, config.mode, "duplicate", _now(), "0", status.source.code_version,
                status.source.results, duplicate_of=status.source.name))
        return TwinRun(status, computed=False, warnings=status.warnings)

    started = time.perf_counter()
    inputs = config.column_inputs()
    if config.mode == "direct":
        forward = api.run_forward(inputs, config.forward_config())
        inversion = None
    else:
        inversion = api.run_inversion(inputs, config.inversion_config(), on_progress=on_progress, should_cancel=should_cancel)
        forward = inversion.best
    duration = time.perf_counter() - started

    code_version = api.engine_version()
    device = api.device()
    provenance = provenance or twin_provenance(config.folder, device)
    target = results_dir(config.folder, status.twin_id)
    partial = target.with_name(target.name + ".partial")
    if partial.exists():
        shutil.rmtree(partial)
    metadata = _metadata(config, status.twin_id, code_version, device, duration, forward, inversion)
    metadata["provenance"] = provenance
    write_results(partial, forward, inversion, [layer.name for layer in config.layers], metadata)
    record_file_digests(partial)
    if target.exists():
        shutil.rmtree(target)
    os.replace(partial, target)  # les résultats apparaissent complets ou pas du tout
    relative = target.relative_to(config.folder).as_posix()
    append_entry(config.folder, RegistryEntry(status.twin_id, config.name, config.mode, "computed", _now(), f"{duration:.3f}", code_version, relative))
    return TwinRun(twin_status(config), computed=True, duration_s=duration, warnings=provenance_warnings(provenance))


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _metadata(config: TwinConfig, twin_id: str, code_version: str, device: str, duration: float, forward, inversion) -> dict:
    from ..engine.jax.forward import HEAT_REFERENCE_C
    measurements = config.load_measurements()
    metadata = {
        "twin_id": twin_id,
        "name": config.name,
        "point": config.point,
        "description": config.description,
        "mode": config.mode,
        "experimental": config.experimental,
        "created_at": _now(),
        "duration_s": round(duration, 3),
        "code_version": code_version,
        "device": device,
        "config": without_seed(config.canonical(code_version)),
        "data": _data_section(config, measurements),
        "rmse_celsius": {"t1": float(forward.rmse[0]), "t2": float(forward.rmse[1]), "t3": float(forward.rmse[2]), "total": float(forward.rmse[3])},
        "sensor_cell_depths": [float(d) for d in forward.sensor_depths],
        "flux_convention": "q > 0: infiltration (river to aquifer); q < 0: exfiltration (aquifer to river)",
        "results_format": RESULTS_FORMAT,
        "instruments": {"sigma_temperature_C": float(config.sigma_temperature), "sigma_pressure_m": float(config.sigma_pressure)},
        "conventions": {
            "z": "positive downwards, m",
            "q": "positive downwards (infiltration), m/s",
            "heat_flux": "positive downwards, W/m2",
            "advective_reference_C": HEAT_REFERENCE_C,
        },
    }
    if inversion is not None:
        metadata["mcmc"] = {
            "burn_in_iterations": int(inversion.burn_in_iterations),
            "burn_in_converged": bool(inversion.burn_in_converged),
            "acceptance_mean": float(inversion.acceptance.mean()),
            "rhat": inversion.rhat.tolist(),
            "best_parameters": {
                layer.name: dict(zip(("neg_log10_k", "porosity", "lambda_s", "rhos_cs"), map(float, inversion.best_parameters[i])))
                for i, layer in enumerate(config.layers)
            },
        }
    return metadata


def _data_section(config: TwinConfig, measurements) -> dict:
    """Données du twin dans metadata.json. 1.x : le fichier de mesures et son empreinte. 2.0.0 : le laboratoire et le
    point dans le dépôt de données, et l'empreinte des mesures nettoyées (celles qui entrent dans le calcul). 2.0.0 avec
    `data.processed` : le dossier des données traitées, le nom du lab et la version du code du manifeste, et la même empreinte."""
    section = {"rows": len(measurements), "start": measurements["date"].iloc[0].isoformat(), "end": measurements["date"].iloc[-1].isoformat()}
    if config.from_processed:
        manifest = config.processed_manifest
        return {"processed": config.data_processed, "lab": manifest["lab"], "code_version": manifest["code_version"],
                "cleaned_sha256": frame_digest(measurements), **section}
    if config.raw_data:
        return {"lab": config.data_lab, "point": config.data_point, "cleaned_sha256": frame_digest(measurements), **section}
    return {"file": config.measurements.name, "sha256": file_digest(config.measurements), **section}


def list_twins(twins_folder: Path) -> list[Path]:
    """Dossiers twin_xx contenant un twin.yaml, triés par numéro (twin_2 avant twin_10)."""
    twins_folder = Path(twins_folder)
    if not twins_folder.is_dir():
        return []
    found = [p for p in twins_folder.iterdir() if p.is_dir() and p.name.lower().startswith(TWIN_PREFIX) and config_path(p).is_file()]

    def key(path: Path):
        suffix = path.name[len(TWIN_PREFIX):]
        return (0, int(suffix), path.name) if suffix.isdigit() else (1, 0, path.name.lower())

    return sorted(found, key=key)


def next_twin_folder(twins_folder: Path) -> Path:
    numbers = [int(p.name[len(TWIN_PREFIX):]) for p in Path(twins_folder).iterdir()
               if p.is_dir() and p.name.lower().startswith(TWIN_PREFIX) and p.name[len(TWIN_PREFIX):].isdigit()] if Path(twins_folder).is_dir() else []
    return Path(twins_folder) / f"{TWIN_PREFIX}{(max(numbers) + 1) if numbers else 1:02d}"


def continue_twin(parent_folder: Path | str, nb_iter_continue: int,
                  on_progress: Callable | None = None,
                  should_cancel: Callable[[], bool] | None = None) -> TwinRun:
    """Reprendre une chaîne MCMC depuis un état sauvegardé et créer un nouveau twin.

    parent_folder : dossier du twin parent (contient twin.yaml et results/)
    nb_iter_continue : nombre d'itérations supplémentaires
    Retourne un TwinRun avec le nouveau twin et ses résultats.
    """
    import numpy as np

    from ..engine import api

    parent_folder = Path(parent_folder).resolve()
    if not (parent_folder / "twin.yaml").is_file():
        raise UserError("E_TWIN_FOLDER_NOT_FOUND", path=parent_folder)

    # Charger la config du parent
    config = load_twin(parent_folder / "twin.yaml")
    parent_status = twin_status(parent_folder)
    parent_twin_id = parent_status.twin_id
    parent_results_dir = parent_status.results_dir

    # Charger l'état du parent
    chain_state_file = parent_results_dir / "chain_state.npz"
    if not chain_state_file.is_file():
        raise UserError("E_CONTINUE_NO_STATE", twin_id=parent_twin_id)

    with np.load(chain_state_file, allow_pickle=False) as data:
        chain_state = {key: np.asarray(data[key]) for key in data.files}

    # Charger les entrées du moteur
    inputs = config.column_inputs()

    # Reprendre l'inversion
    code_version = api.engine_version()
    device = api.device()
    started = time.perf_counter()
    inversion = api.continue_inversion(
        inputs, config.inversion_config(), chain_state, nb_iter_continue,
        on_progress=on_progress, should_cancel=should_cancel
    )
    duration = time.perf_counter() - started
    forward = inversion.best

    # Générer le twin_id du nouveau twin : suffixe incluant itérations
    new_twin_id = f"{parent_twin_id}_+{nb_iter_continue}"

    # Créer un dossier temporaire pour le nouveau twin
    target = results_dir(config.folder, new_twin_id)
    partial = target.with_name(target.name + ".partial")
    if partial.exists():
        shutil.rmtree(partial)

    # Écrire les résultats
    provenance = twin_provenance(config.folder, device)
    metadata = _metadata(config, new_twin_id, code_version, device, duration, forward, inversion)
    metadata["provenance"] = provenance
    metadata["parent_twin_id"] = parent_twin_id
    metadata["continued_iterations"] = nb_iter_continue
    write_results(partial, forward, inversion, [layer.name for layer in config.layers], metadata)
    record_file_digests(partial)
    if target.exists():
        shutil.rmtree(target)
    os.replace(partial, target)

    # Ajouter au registre
    relative = target.relative_to(config.folder).as_posix()
    append_entry(config.folder, RegistryEntry(
        new_twin_id, f"{config.name}_+{nb_iter_continue}", config.mode, "computed",
        _now(), f"{duration:.3f}", code_version, relative
    ))

    status = TwinStatus(config, new_twin_id, "computed", RegistryEntry(
        new_twin_id, f"{config.name}_+{nb_iter_continue}", config.mode, "computed",
        _now(), f"{duration:.3f}", code_version, relative
    ))
    return TwinRun(status, computed=True, duration_s=duration, warnings=provenance_warnings(provenance))
