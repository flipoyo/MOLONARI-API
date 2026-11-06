"""Figures d'analyse (Matplotlib) : cartes profondeur × temps, profils, comparaison aux mesures.

Conventions : profondeur croissante vers le bas, axes avec unités, palettes
perceptuellement uniformes ; flux signés centrés sur 0. Chaque figure s'exporte en
PNG, SVG ou PDF.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.colors import TwoSlopeNorm
from matplotlib.figure import Figure
from PySide6 import QtWidgets

from ...i18n import t
from .style import (
    FLUX_CMAP,
    PREDICTIVE_COLOR,
    SIMULATED_COLOR,
    TEMPERATURE_CMAP,
    THERMOMETER_COLORS,
    style_mpl,
)

EXPORT_FILTER = "PNG (*.png);;SVG (*.svg);;PDF (*.pdf)"


class FigurePanel(QtWidgets.QWidget):
    """Figure Matplotlib avec barre d'outils (zoom, déplacement) et export."""

    def __init__(self, parent=None, height: float = 4.0):
        super().__init__(parent)
        self.figure = Figure(figsize=(7, height), layout="constrained")
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.canvas.setMinimumHeight(220)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        self.toolbar.setIconSize(self.toolbar.iconSize() * 0.8)
        export = QtWidgets.QPushButton(t("Export figure…"))
        export.clicked.connect(self.export_dialog)
        top = QtWidgets.QHBoxLayout()
        top.addWidget(self.toolbar, 1)
        top.addWidget(export)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(top)
        layout.addWidget(self.canvas, 1)

    def draw(self) -> None:
        style_mpl(self.figure)
        self.canvas.draw_idle()

    def clear(self, message: str = "") -> None:
        self.figure.clear()
        if message:
            ax = self.figure.add_subplot()
            ax.axis("off")
            ax.text(0.5, 0.5, message, ha="center", va="center", wrap=True)
        self.draw()

    def export(self, path: str | Path) -> None:
        self.figure.savefig(path, dpi=200)

    def export_dialog(self) -> None:
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, t("Export figure"), "", EXPORT_FILTER)
        if path:
            self.export(path)


# Dates numériques : lisibles quelle que soit la langue de l'interface (pas de noms de mois anglais).
DATE_FORMATS = ["%Y", "%m/%Y", "%d/%m", "%H:%M", "%H:%M", "%S.%f"]
DATE_OFFSET_FORMATS = ["", "%Y", "%m/%Y", "%d/%m/%Y", "%d/%m/%Y", "%d/%m/%Y %H:%M"]


def _date_axis(ax) -> None:
    locator = mdates.AutoDateLocator()
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator, formats=DATE_FORMATS, offset_formats=DATE_OFFSET_FORMATS))


def depth_time_map(ax, times, depths, values, label: str, signed: bool = False, sensor_depths=None):
    """Carte profondeur × temps. `values` : (profondeurs, temps)."""
    times = pd.to_datetime(times).to_numpy()
    values = np.asarray(values, float)
    if signed:
        limit = float(np.nanmax(np.abs(values))) or 1.0
        norm = TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)
        mesh = ax.pcolormesh(times, depths, values, cmap=FLUX_CMAP, norm=norm, shading="nearest", rasterized=True)
    else:
        mesh = ax.pcolormesh(times, depths, values, cmap=TEMPERATURE_CMAP, shading="nearest", rasterized=True)
    ax.set_ylim(float(np.max(depths)), float(np.min(depths)) if np.min(depths) > 0 else 0.0)
    ax.set_ylabel(t("Depth (m)"))
    if sensor_depths is not None:
        for depth in sensor_depths:
            ax.axhline(depth, color="white", lw=0.6, ls="--", alpha=0.7)
    _date_axis(ax)
    colorbar = ax.figure.colorbar(mesh, ax=ax, pad=0.01)
    colorbar.set_label(label)
    return mesh


def sensor_comparison(ax_list, times, observed, simulated, band=None, unit="°C", rmse=None, measured_band=None, measured_label=None):
    """Températures mesurées et simulées aux 3 capteurs.

    band : bande de quantiles du modèle (facultative) ; measured_band : (bas, haut), intervalle
    d'incertitude autour des mesures (facultatif), légendé par measured_label."""
    times = pd.to_datetime(times).to_numpy()
    for i, ax in enumerate(ax_list):
        if measured_band is not None:
            ax.fill_between(times, measured_band[0][:, i], measured_band[1][:, i], color=THERMOMETER_COLORS[i], alpha=0.22, lw=0,
                            label=measured_label)
        ax.plot(times, observed[:, i], color=THERMOMETER_COLORS[i], lw=1.2, label=t("Measured"))
        if band is not None:
            ax.fill_between(times, band[0][:, i], band[1][:, i], color=SIMULATED_COLOR, alpha=0.2, lw=0, label=t("Credible interval"))
        ax.plot(times, simulated[:, i], color=SIMULATED_COLOR, lw=1.0, ls="--", label=t("Simulated"))
        title = t("Sensor {n}").format(n=i + 1)
        if rmse is not None:
            title += f" — RMSE {rmse[i]:.3f} {'°F' if unit == '°F' else 'K' if unit == 'K' else '°C'}"
        ax.set_title(title, fontsize=9, loc="left")
        ax.set_ylabel(unit)
        _date_axis(ax)
    ax_list[0].legend(loc="upper right", fontsize=8)


def predictive_band(ax_list, times, low, high, label: str) -> None:
    """Bande prédictive (bas, haut) de forme (T′, 3) aux 3 capteurs, en contour pour laisser voir les autres bandes.

    La collection porte le gid « predictive », et la légende du premier graphe donne `label`."""
    times = pd.to_datetime(times).to_numpy()
    for i, ax in enumerate(ax_list):
        band = ax.fill_between(times, low[:, i], high[:, i], facecolor=PREDICTIVE_COLOR, alpha=0.12, edgecolor=PREDICTIVE_COLOR,
                               lw=0.7, label=label if i == 0 else None)
        band.set_gid("predictive")
    ax_list[0].legend(loc="upper right", fontsize=8)


def profiles(ax, depths, curves: list[tuple[str, np.ndarray]], unit: str, observed=None):
    """Profils verticaux à quelques dates choisies."""
    for i, (label, values) in enumerate(curves):
        color = THERMOMETER_COLORS[i % len(THERMOMETER_COLORS)]
        ax.plot(values, depths, color=color, lw=1.4, label=label)
    if observed is not None:
        obs_depths, obs_values = observed
        for i, values in enumerate(obs_values):
            ax.plot(values, obs_depths, "o", color=THERMOMETER_COLORS[i % len(THERMOMETER_COLORS)], ms=4)
    ax.set_ylim(float(np.max(depths)), 0.0)
    ax.set_xlabel(unit)
    ax.set_ylabel(t("Depth (m)"))
    ax.legend(fontsize=8)


def robust_ylim(ax, values, factor: float = 5.0) -> bool:
    """Cale l'axe vertical sur les valeurs courantes quand quelques pics écraseraient l'échelle.

    Retourne True si des valeurs sortent du cadre (l'appelant peut le signaler).
    """
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    if values.size < 10:
        return False
    low, high = np.percentile(values, [1, 99])
    spread = max(high - low, abs(high), abs(low), 1e-30)
    if np.max(values) - np.min(values) <= factor * spread:
        return False
    margin = 0.25 * (high - low if high > low else spread)
    ax.set_ylim(low - margin, high + margin)
    ax.text(0.01, 0.98, t("Some peaks are off-scale (zoom out to see them)."), transform=ax.transAxes, fontsize=7, va="top", alpha=0.7)
    return True


def time_series_with_band(ax, times, values, band=None, label: str = "", color: str = SIMULATED_COLOR, y_label: str = ""):
    times = pd.to_datetime(times).to_numpy()
    if band is not None:
        ax.fill_between(times, band[0], band[1], color=color, alpha=0.2, lw=0, label=t("Credible interval"))
    ax.plot(times, values, color=color, lw=1.2, label=label)
    ax.axhline(0.0, color="#888888", lw=0.6)
    ax.set_ylabel(y_label)
    _date_axis(ax)
    robust_ylim(ax, values)
    ax.legend(fontsize=8, loc="upper right")
