"""Cycle de vie d'un twin : valider, identifier, vérifier le registre, exécuter, ranger.

    1. l'utilisateur écrit un twin.yaml ;
    2. la config est validée (config.load_twin) ;
    3. le twin_id est calculé et le registre consulté (status) ;
    4. le twin est exécuté en mode direct ou mcmc (run_twin) ;
    5. les résultats sont rangés dans results/<twin_id>/ et le registre mis à jour ;
    6. une config identique relancée est signalée comme doublon : le résultat existant
       est renvoyé sans recalcul.
"""

from __future__ import annotations

import os
import shutil
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from ..errors import Issue, UserError
from .config import (
    TwinConfig,
    config_path,
    file_digest,
    frame_digest,
    load_twin,
    without_seed,
)
from .provenance import provenance_warnings, strict_problems, twin_provenance
from .registry import (
    RegistryEntry,
    append_entry,
    computed_entry,
    entry_for_name,
    read_registry,
    results_dir,
    twins_root,
)
from .results import RESULTS_FORMAT, record_file_digests, write_results

TWIN_PREFIX = "twin_"


@dataclass
class TwinStatus:
    config: TwinConfig
    twin_id: str
    state: str  # « new », « computed » (déjà calculé), « duplicate » (même config qu'un autre twin), « modified »
    source: RegistryEntry | None = None  # exécution qui a produit les résultats
    previous_id: str = ""  # pour « modified » : identifiant enregistré pour ce nom

    @property
    def results_dir(self) -> Path:
        return results_dir(self.twins_folder, self.twin_id)

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
    twin_id = config.twin_id(code_version)
    twins_folder = config.folder.parent
    entries = read_registry(twins_folder)
    named = entry_for_name(entries, config.name)
    if named is not None and named.twin_id != twin_id:
        return TwinStatus(config, twin_id, "modified", previous_id=named.twin_id)
    source = computed_entry(entries, twin_id)
    if source is not None and (twins_root(twins_folder) / source.results / "metadata.json").is_file():
        return TwinStatus(config, twin_id, "computed" if source.name == config.name else "duplicate", source)
    return TwinStatus(config, twin_id, "new")


def run_twin(path_or_config, on_progress: Callable | None = None, should_cancel: Callable[[], bool] | None = None,
             strict: bool = False) -> TwinRun:
    """Exécute un twin (ou renvoie le résultat existant). `strict` : refuse un code non publié ou des dépôts
    modifiés ou non tagués (E_STRICT_REFUSED, avec tous les motifs), avant tout calcul."""
    from .. import __version__
    from ..engine import api

    status = twin_status(path_or_config)
    config = status.config
    twins_folder = status.twins_folder
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
        entries = read_registry(twins_folder)
        if status.state == "duplicate" and entry_for_name(entries, config.name) is None:
            append_entry(twins_folder, RegistryEntry(
                status.twin_id, config.name, config.mode, "duplicate", _now(), "0", status.source.code_version,
                status.source.results, duplicate_of=status.source.name))
        return TwinRun(status, computed=False)

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
    target = results_dir(twins_folder, status.twin_id)
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
    relative = target.relative_to(twins_root(twins_folder)).as_posix()
    append_entry(twins_folder, RegistryEntry(status.twin_id, config.name, config.mode, "computed", _now(), f"{duration:.3f}", code_version, relative))
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
    point dans le dépôt de données, et l'empreinte des mesures nettoyées (celles qui entrent dans le calcul)."""
    section = {"rows": len(measurements), "start": measurements["date"].iloc[0].isoformat(), "end": measurements["date"].iloc[-1].isoformat()}
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
