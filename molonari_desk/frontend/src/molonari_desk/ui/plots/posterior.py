"""Analyse a posteriori : histogrammes et traces (le diagnostic ArviZ passe par la façade : summary_table, export_netcdf)."""

from __future__ import annotations

from molonari_twin.facade import INVERTED_PARAMETERS, PARAMETER_LABELS, InversionResult

from ...i18n import t
from .style import CHAIN_COLORMAP

__all__ = ["chain_traces", "histograms"]


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


def chain_traces(figure, answer, layer: int) -> None:
    """Traces des paramètres de la couche `layer` et de l'énergie, une courbe par chaîne, en fonction de
    l'itération calculée. `answer` est la réponse de questions.chain_data ; chaque chaîne garde sa
    couleur (d'après son numéro) et porte le gid « chain-<n> »."""
    figure.clear()
    axes = figure.subplots(len(INVERTED_PARAMETERS) + 1, 1, sharex=True)
    iterations = answer.data["iteration"].value
    chains = answer.data["chain"].value
    rows = [(PARAMETER_LABELS[name], answer.data[f"samples.{name}"].value[:, :, layer]) for name in INVERTED_PARAMETERS]
    rows.append((t("Energy"), answer.data["energy"].value))
    for ax, (label, values) in zip(axes, rows):
        for j, chain in enumerate(chains):
            ax.plot(iterations, values[:, j], lw=0.6, color=CHAIN_COLORMAP(int(chain) % 10), gid=f"chain-{int(chain)}")
        ax.set_ylabel(label, fontsize=9)
    axes[-1].set_xlabel(t("Computed iteration after burn-in"))
