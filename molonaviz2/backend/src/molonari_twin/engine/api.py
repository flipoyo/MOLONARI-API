"""Point d'entrée du moteur de calcul utilisé par l'application."""

from __future__ import annotations

import functools
import sys
from collections.abc import Callable
from pathlib import Path

from .. import __version__
from ..domain.compute import ForwardConfig, InversionConfig
from ..errors import UserError
from .column import ColumnInputs
from .results import ForwardResult, InversionResult


@functools.cache
def engine_version() -> str:
    """Version du code chargé : `molonari-twin X.Y.Z[+g<commit>[.dirty]] · jax A.B.C`.

    Elle entre dans chaque twin_id. Calculée une fois par processus : elle décrit le code chargé."""
    import jax

    return f"molonari-twin {__version__}{local_suffix()} · jax {jax.__version__}"


def release_version() -> str:
    """Version publiée du code chargé, sans suffixe local (celle qu'il aura une fois tagué)."""
    import jax

    return f"molonari-twin {__version__} · jax {jax.__version__}"


def local_suffix() -> str:
    """Suffixe local de la version : vide pour l'exécutable figé, sans git ou hors dépôt (voir gitinfo)."""
    if getattr(sys, "frozen", False):
        return ""
    project = Path(__file__).resolve().parents[3]  # backend/src/molonari_twin/engine/api.py → backend/
    if not (project / "pyproject.toml").is_file():  # installé hors de son dépôt
        return ""
    from ..gitinfo import version_suffix

    return version_suffix(project, f"molonari-twin-v{__version__}")


def device() -> str:
    from .jax.device import device_description

    return device_description()


def run_forward(inputs: ColumnInputs, config: ForwardConfig) -> ForwardResult:
    issues = config.validate(inputs.column_depth)
    if issues:
        raise UserError("E_SETTINGS_INVALID", issues=issues)
    from .jax.forward import run_forward as _run

    return _run(inputs, [layer.physical() for layer in config.layers], [layer.z_low for layer in config.layers], config.nb_cells)


def sampling_layout(config: InversionConfig, nb_times: int | None = None):
    """Itérations et champs (T, q) par chaîne pour ces réglages : voir jax.dream.SamplingLayout."""
    from .jax.dream import sampling_layout as _layout

    return _layout(config, nb_times)


def run_inversion(
    inputs: ColumnInputs,
    config: InversionConfig,
    on_progress: Callable | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> InversionResult:
    from .jax.dream import DreamSampler

    return DreamSampler(inputs, config).run(on_progress=on_progress, should_cancel=should_cancel)
