"""Point d'entrée du moteur de calcul utilisé par l'application."""

from __future__ import annotations

from collections.abc import Callable

from .. import __version__
from ..domain.compute import ForwardConfig, InversionConfig
from ..errors import UserError
from .column import ColumnInputs
from .results import ForwardResult, InversionResult


def engine_version() -> str:
    import jax

    return f"molonaviz2 {__version__} · jax {jax.__version__}"


def device() -> str:
    from .jax.device import device_description

    return device_description()


def run_forward(inputs: ColumnInputs, config: ForwardConfig) -> ForwardResult:
    issues = config.validate(inputs.column_depth)
    if issues:
        raise UserError("\n".join(issue.message for issue in issues))
    from .jax.forward import run_forward as _run

    return _run(inputs, [layer.physical() for layer in config.layers], [layer.z_low for layer in config.layers], config.nb_cells)


def run_inversion(
    inputs: ColumnInputs,
    config: InversionConfig,
    on_progress: Callable | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> InversionResult:
    from .jax.dream import DreamSampler

    return DreamSampler(inputs, config).run(on_progress=on_progress, should_cancel=should_cancel)
