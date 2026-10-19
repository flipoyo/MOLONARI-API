"""Analyse a posteriori : histogrammes et traces (le diagnostic ArviZ est dans engine.diagnostics)."""

from __future__ import annotations

import numpy as np

from ...domain.compute import INVERTED_PARAMETERS, PARAMETER_LABELS
from ...engine.diagnostics import export_netcdf, inference_data, summary_table
from ...engine.results import InversionResult
from ...i18n import t
from .style import CHAIN_COLORMAP

__all__ = ["export_netcdf", "histograms", "inference_data", "summary_table", "traces"]


def histograms(figure, result: InversionResult, layer: int, bins: int) -> None:
    figure.clear()
    axes = figure.subplots(2, 2)
    flat = result.samples[:, :, layer, :].reshape(-1, len(INVERTED_PARAMETERS))
    for p, (ax, name) in enumerate(zip(axes.flat, INVERTED_PARAMETERS)):
        ax.hist(flat[:, p], bins=bins, color="#1f6feb", alpha=0.8)
        ax.axvline(result.best_parameters[layer, p], color="#c0392b", lw=1.2, label=t("Best"))
        ax.set_title(PARAMETER_LABELS[name], fontsize=9, loc="left")
        ax.set_yticks([])
    axes.flat[0].legend(fontsize=8)


def traces(figure, result: InversionResult, layer: int) -> None:
    figure.clear()
    axes = figure.subplots(len(INVERTED_PARAMETERS), 1, sharex=True)
    iterations = np.arange(result.samples.shape[0])
    for p, (ax, name) in enumerate(zip(axes, INVERTED_PARAMETERS)):
        for chain in range(result.samples.shape[1]):
            ax.plot(iterations, result.samples[:, chain, layer, p], lw=0.6, color=CHAIN_COLORMAP(chain % 10))
        ax.set_ylabel(PARAMETER_LABELS[name], fontsize=9)
    axes[-1].set_xlabel(t("Iteration after burn-in (per chain)"))
