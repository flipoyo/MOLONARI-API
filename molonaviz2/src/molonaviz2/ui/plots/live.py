"""Suivi en direct de l'inversion (PyQtGraph) : énergie par chaîne, acceptation, R-hat."""

from __future__ import annotations

import math

import numpy as np
import pyqtgraph as pg
from matplotlib import colors
from PySide6 import QtWidgets

from ...i18n import t
from .style import CHAIN_COLORMAP, style_pg


class LiveMonitor(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.energy = pg.PlotWidget()
        self.acceptance = pg.PlotWidget()
        self.rhat = pg.PlotWidget()
        for plot, label in (
            (self.energy, t("Energy (−log likelihood)")),
            (self.acceptance, t("Acceptance rate")),
            (self.rhat, t("Largest R-hat")),
        ):
            style_pg(plot)
            plot.setLabel("left", label)
            plot.setLabel("bottom", t("Iteration"))
            plot.setMinimumHeight(140)
            layout.addWidget(plot)
        self.acceptance.setYRange(0, 1)
        self.rhat.setLogMode(y=True)
        self._threshold = pg.InfiniteLine(angle=0, pen=pg.mkPen("#c0392b", style=pg.QtCore.Qt.PenStyle.DashLine))
        self.rhat.addItem(self._threshold)
        self.reset(threshold=1.2, nb_chains=0)

    def reset(self, threshold: float, nb_chains: int) -> None:
        self.energy.clear()
        self.acceptance.clear()
        self.rhat.clear()
        self.rhat.addItem(self._threshold)
        self._threshold.setValue(math.log10(threshold))
        self._x: list[int] = []
        self._energies: list[np.ndarray] = []
        self._acc: list[float] = []
        self._rhat_x: list[int] = []
        self._rhat: list[float] = []
        self._offset = 0
        self._phase = None
        self._curves = [
            self.energy.plot(pen=pg.mkPen(colors.to_hex(CHAIN_COLORMAP(i % 10)), width=1)) for i in range(nb_chains)
        ]
        self._acc_curve = self.acceptance.plot(pen=pg.mkPen("#1f6feb", width=1.5))
        self._rhat_curve = self.rhat.plot(pen=pg.mkPen("#1f6feb", width=1.5), symbol="o", symbolSize=4)
        self._burn_in_line = None

    def update(self, progress) -> None:
        if progress.phase not in ("burn_in", "sampling"):
            return
        if progress.phase == "sampling" and self._phase == "burn_in":
            self._offset = self._x[-1] if self._x else 0
            self._burn_in_line = pg.InfiniteLine(self._offset, angle=90, pen=pg.mkPen("#888888", style=pg.QtCore.Qt.PenStyle.DotLine))
            for plot in (self.energy, self.acceptance, self.rhat):
                plot.addItem(pg.InfiniteLine(self._offset, angle=90, pen=pg.mkPen("#888888", style=pg.QtCore.Qt.PenStyle.DotLine)))
        self._phase = progress.phase
        x = self._offset + progress.iteration
        self._x.append(x)
        energies = np.asarray(progress.chain_energies, float)
        self._energies.append(energies)
        self._acc.append(progress.acceptance)
        data = np.stack(self._energies)
        xs = np.asarray(self._x)
        finite = np.isfinite(data)
        for j, curve in enumerate(self._curves):
            if j < data.shape[1]:
                curve.setData(xs[finite[:, j]], data[finite[:, j], j])
        self._acc_curve.setData(xs, np.asarray(self._acc))
        if np.isfinite(progress.rhat_max) and progress.phase == "burn_in":
            self._rhat_x.append(x)
            self._rhat.append(max(progress.rhat_max, 1.0))
            self._rhat_curve.setData(np.asarray(self._rhat_x), np.asarray(self._rhat))
