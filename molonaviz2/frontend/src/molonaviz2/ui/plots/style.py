"""Style commun des graphiques, cohérent avec la palette de l'application (clair/sombre).

- Thermomètres : 4 teintes de viridis, du moins profond (sombre) au plus profond (clair),
  pour que l'ordre de profondeur se lise dans la couleur.
- Rivière / capteur de pression : orange, distinct de viridis.
- Valeurs écartées : rouge.
- Cartes : viridis pour les températures ; RdBu_r centré sur 0 pour les flux (signés).
"""

from __future__ import annotations

import matplotlib

matplotlib.use("QtAgg")

# Ces imports restent après matplotlib.use : ils chargent matplotlib et Qt.
import pyqtgraph as pg
from matplotlib import cm
from PySide6 import QtGui, QtWidgets

THERMOMETER_COLORS = [matplotlib.colors.to_hex(cm.viridis(x)) for x in (0.05, 0.35, 0.62, 0.85)]
RIVER_COLOR = "#d55e00"
PRESSURE_COLOR = "#0072b2"
REJECTED_COLOR = "#c0392b"
SIMULATED_COLOR = "#444444"
PREDICTIVE_COLOR = "#7b3294"  # bande prédictive : violet, distinct des thermomètres et du gris de la bande crédible
CHAIN_COLORMAP = cm.tab10
TEMPERATURE_CMAP = "viridis"
FLUX_CMAP = "RdBu_r"

pg.setConfigOptions(antialias=True)


def palette_colors() -> tuple[str, str, str]:
    """(fond, texte, grille) d'après la palette courante."""
    palette = QtWidgets.QApplication.palette()
    background = palette.color(QtGui.QPalette.ColorRole.Base).name()
    text = palette.color(QtGui.QPalette.ColorRole.Text).name()
    grid = palette.color(QtGui.QPalette.ColorRole.Mid).name()
    return background, text, grid


def style_pg(plot: pg.PlotWidget) -> None:
    background, text, _ = palette_colors()
    plot.setBackground(background)
    item = plot.getPlotItem()
    for name in ("left", "bottom", "right"):
        axis = item.getAxis(name)
        axis.setPen(pg.mkPen(text))
        axis.setTextPen(pg.mkPen(text))
    item.showGrid(x=True, y=True, alpha=0.15)


def style_mpl(figure) -> None:
    background, text, grid = palette_colors()
    figure.set_facecolor(background)
    for ax in figure.axes:
        ax.set_facecolor(background)
        ax.tick_params(colors=text, which="both")
        for spine in ax.spines.values():
            spine.set_color(grid)
        ax.xaxis.label.set_color(text)
        ax.yaxis.label.set_color(text)
        ax.title.set_color(text)
        legend = ax.get_legend()
        if legend is not None:
            legend.get_frame().set_facecolor(background)
            legend.get_frame().set_edgecolor(grid)
            for label in legend.get_texts():
                label.set_color(text)
