"""Séries temporelles interactives (PyQtGraph) : mesures et nettoyage.

Zoom à la molette, déplacement à la souris ; les graphes d'un même onglet partagent
l'axe des temps (zoomer sur l'un zoome sur les autres).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from .style import REJECTED_COLOR, style_pg


def to_epoch(dates) -> np.ndarray:
    return pd.to_datetime(pd.Series(dates)).to_numpy().astype("datetime64[ms]").astype(np.int64) / 1000.0


class TimeSeriesPlot(pg.PlotWidget):
    def __init__(self, y_label: str = "", parent=None):
        super().__init__(parent=parent, axisItems={"bottom": pg.DateAxisItem()})
        style_pg(self)
        self.setLabel("left", y_label)
        self.addLegend(offset=(-10, 10))
        self.setMinimumHeight(180)

    def set_y_label(self, label: str) -> None:
        self.setLabel("left", label)

    def plot_series(self, dates, values, name: str, color: str, width: float = 1.5) -> None:
        x = to_epoch(dates)
        self.plot(x, np.asarray(values, float), pen=pg.mkPen(color, width=width), name=name)

    def plot_points(self, dates, values, name: str, color: str = REJECTED_COLOR, size: int = 7) -> pg.ScatterPlotItem:
        x = to_epoch(dates)
        item = pg.ScatterPlotItem(x, np.asarray(values, float), size=size, brush=pg.mkBrush(color), pen=None, name=name)
        self.addItem(item)
        legend = self.getPlotItem().legend
        if legend is not None and len(x):
            legend.addItem(item, name)
        return item

    def reset(self) -> None:
        self.clear()
        legend = self.getPlotItem().legend
        if legend is not None:
            legend.clear()


class RegionSelector(QtCore.QObject):
    """Bande sélectionnable sur un graphe temporel, pour exclure une période à la main."""

    def __init__(self, plot: TimeSeriesPlot):
        super().__init__(plot)
        self._plot = plot
        self.region = pg.LinearRegionItem(brush=pg.mkBrush(192, 57, 43, 40))
        self.region.setZValue(10)
        self.region.hide()
        plot.addItem(self.region, ignoreBounds=True)

    def show_default(self) -> None:
        (x0, x1), _ = self._plot.viewRange()
        width = x1 - x0
        self.region.setRegion((x0 + 0.4 * width, x0 + 0.6 * width))
        if self.region.scene() is None:
            self._plot.addItem(self.region, ignoreBounds=True)
        self.region.show()

    def hide(self) -> None:
        self.region.hide()

    def bounds(self) -> tuple[pd.Timestamp, pd.Timestamp] | None:
        if not self.region.isVisible():
            return None
        a, b = self.region.getRegion()
        return pd.Timestamp(a, unit="s"), pd.Timestamp(b, unit="s")


def link_time_axes(plots: list[pg.PlotWidget]) -> None:
    for other in plots[1:]:
        other.setXLink(plots[0])


def stacked(widgets: list[QtWidgets.QWidget]) -> QtWidgets.QSplitter:
    splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
    for widget in widgets:
        splitter.addWidget(widget)
    return splitter
