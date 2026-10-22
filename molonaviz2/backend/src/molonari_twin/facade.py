"""Façade du cœur : seul module du cœur que l'interface importe (avec la racine du paquet et `i18n`).

Les objets sont réexportés tels quels : ce sont les mêmes classes et les mêmes fonctions que dans leur
module d'origine. L'interface appelle les calculs comme attributs du module
(`from molonaviz2 import facade as api`, puis `api.run_forward(...)`), ce qui permet aux tests de les
remplacer. Importer la façade ne charge ni Qt, ni JAX, ni ArviZ : les diagnostics et la ligne de
commande sont importés à l'appel. Aucun autre module du cœur n'importe la façade.
"""

from __future__ import annotations

from dataclasses import MISSING, fields
from pathlib import Path

# Modèles et configurations des calculs.
from .domain.compute import (
    DEFAULT_PRIORS,
    INVERTED_PARAMETERS,
    PARAMETER_LABELS,
    PARAMETER_UNITS,
    PHYSICAL_BOUNDS,
    SETTING_LIMITS,
    STORE_FIELDS,
    ForwardConfig,
    InversionConfig,
    LayerSpec,
    PriorRange,
    default_priors,
    setting_limits,
)
from .domain.models import (
    Lab,
    PointStage,
    PressureSensor,
    SamplingPoint,
    Shaft,
    Study,
    Thermometer,
)

# Calcul.
from .engine.api import (
    device,
    engine_version,
    release_version,
    run_forward,
    run_inversion,
    sampling_layout,
)
from .engine.column import ColumnInputs
from .engine.results import ForwardResult, InversionResult

# Erreurs.
from .errors import Cancelled, Issue, UserError

# Imports de fichiers.
from .io.lab_import import LabImport, import_lab_directory, read_text
from .io.legacy_import import import_legacy_database
from .io.point_import import (
    ParsedMeasures,
    find_point_files,
    read_point_info,
    read_pressure_file,
    read_temperature_file,
)

# Nettoyage.
from .processing.cleanup import (
    IQR_FACTOR,
    VARIABLES,
    ZSCORE_LIMIT,
    CleanupResult,
    CleanupSettings,
    clean,
    read_cleaned_csv,
    write_cleaned_csv,
)

# Questions.
from .questions import (
    MEASUREMENT_LEVELS,
    Answer,
    Quantity,
    chain_data,
    coverage,
    exchanged_volume,
    measurement_interval,
    observed_at,
    predictive_band,
    sensor_intervals,
    sensor_samples,
    z_score,
)

# Projet (.molo).
from .storage.project import SUFFIX, ComputationInfo, Project

# Twins.
from .twin.config import LAYER_KEYS as TWIN_LAYER_KEYS
from .twin.config import MCMC_KEYS as TWIN_MCMC_KEYS
from .twin.config import TwinConfig, TwinConfigError, config_path
from .twin.create import create_twin
from .twin.results import load_results
from .twin.run import TwinStatus, list_twins, run_twin, twin_status


def summary_table(result: InversionResult, layer_names: list[str]):
    """Tableau de synthèse a posteriori (ArviZ, chargé à l'appel)."""
    from .engine.diagnostics import summary_table as table

    return table(result, layer_names)


def export_netcdf(result: InversionResult, layer_names: list[str], path: str | Path) -> None:
    """Échantillons a posteriori au format NetCDF d'ArviZ (chargé à l'appel)."""
    from .engine.diagnostics import export_netcdf as export

    export(result, layer_names, path)


def twin_defaults() -> dict[str, object]:
    """Valeurs par défaut des champs d'un twin (ceux qui en ont une)."""
    return {f.name: f.default for f in fields(TwinConfig) if f.default is not MISSING}


def twin_cli(argv: list[str]) -> int:
    """Ligne de commande des twins (`molonaviz2 twin …`), sans Qt."""
    from .twin.cli import main

    return main(argv)


__all__ = [
    "DEFAULT_PRIORS",
    "INVERTED_PARAMETERS",
    "IQR_FACTOR",
    "MEASUREMENT_LEVELS",
    "PARAMETER_LABELS",
    "PARAMETER_UNITS",
    "PHYSICAL_BOUNDS",
    "SETTING_LIMITS",
    "STORE_FIELDS",
    "SUFFIX",
    "TWIN_LAYER_KEYS",
    "TWIN_MCMC_KEYS",
    "VARIABLES",
    "ZSCORE_LIMIT",
    "Answer",
    "Cancelled",
    "CleanupResult",
    "CleanupSettings",
    "ColumnInputs",
    "ComputationInfo",
    "ForwardConfig",
    "ForwardResult",
    "InversionConfig",
    "InversionResult",
    "Issue",
    "Lab",
    "LabImport",
    "LayerSpec",
    "ParsedMeasures",
    "PointStage",
    "PressureSensor",
    "PriorRange",
    "Project",
    "Quantity",
    "SamplingPoint",
    "Shaft",
    "Study",
    "Thermometer",
    "TwinConfigError",
    "TwinStatus",
    "UserError",
    "chain_data",
    "clean",
    "config_path",
    "coverage",
    "create_twin",
    "default_priors",
    "device",
    "engine_version",
    "exchanged_volume",
    "export_netcdf",
    "find_point_files",
    "import_lab_directory",
    "import_legacy_database",
    "list_twins",
    "load_results",
    "measurement_interval",
    "observed_at",
    "predictive_band",
    "read_cleaned_csv",
    "read_point_info",
    "read_pressure_file",
    "read_temperature_file",
    "read_text",
    "release_version",
    "run_forward",
    "run_inversion",
    "run_twin",
    "sampling_layout",
    "sensor_intervals",
    "sensor_samples",
    "setting_limits",
    "summary_table",
    "twin_cli",
    "twin_defaults",
    "twin_status",
    "write_cleaned_csv",
    "z_score",
]
