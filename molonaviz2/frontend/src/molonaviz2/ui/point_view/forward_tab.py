"""Onglet « Modèle direct » : couches, lancement (F5), comparaison immédiate aux mesures."""

from __future__ import annotations

import numpy as np
from molonari_twin import facade as api
from molonari_twin.facade import ForwardConfig, ForwardResult, LayerSpec, setting_limits
from PySide6 import QtCore, QtWidgets

from ...i18n import t
from ...workers.tasks import Task, TaskManager
from ..dialogs.notice import attach
from ..plots.maps import FigurePanel, sensor_comparison
from ..units import temperature_unit
from ..widgets import NumberField, heading, hint
from .context import PointContext
from .layers import LayersTable


class ForwardTab(QtWidgets.QWidget):
    show_results = QtCore.Signal(int)  # identifiant du calcul

    def __init__(self, context: PointContext, tasks: TaskManager, parent=None):
        super().__init__(parent)
        self.context = context
        self.tasks = tasks
        self._last: ForwardResult | None = None

        self.layers = LayersTable("values", context.column_depth)
        add = QtWidgets.QPushButton(t("Add a layer"))
        add.clicked.connect(self.layers.add_layer)
        remove = QtWidgets.QPushButton(t("Remove selected layer"))
        remove.clicked.connect(self.layers.remove_selected)
        buttons = QtWidgets.QHBoxLayout()
        buttons.addWidget(add)
        buttons.addWidget(remove)
        buttons.addStretch()
        self.nb_cells = NumberField(ForwardConfig(layers=[]).nb_cells, t("cells"), integer=True, **setting_limits("nb_cells"))
        attach(self.nb_cells, "numerics.nb_cells")
        cells = QtWidgets.QFormLayout()
        cells.addRow(t("Discretisation"), self.nb_cells)
        self.issues = QtWidgets.QLabel()
        self.issues.setProperty("role", "error")
        self.issues.setWordWrap(True)
        self.run_button = QtWidgets.QPushButton(t("Run direct model  (F5)"))
        self.run_button.setDefault(True)
        self.run_button.clicked.connect(self.run)

        side = QtWidgets.QWidget()
        side_layout = QtWidgets.QVBoxLayout(side)
        side_layout.addWidget(heading(t("Layers")))
        side_layout.addWidget(hint(t("From top to bottom. Permeability is entered as −log₁₀ k, with k in m² (12 means k = 10⁻¹² m²).")))
        side_layout.addWidget(self.layers)
        side_layout.addLayout(buttons)
        side_layout.addLayout(cells)
        side_layout.addWidget(self.issues)
        side_layout.addWidget(self.run_button)
        side_layout.addStretch()
        side.setMinimumWidth(520)

        self.summary = QtWidgets.QLabel(t("No direct model computed yet."))
        self.summary.setWordWrap(True)
        self.details_button = QtWidgets.QPushButton(t("Open detailed results"))
        self.details_button.setEnabled(False)
        self.details_button.clicked.connect(lambda: self.show_results.emit(self._last_id))
        self.figure = FigurePanel(height=5)
        right = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right)
        top = QtWidgets.QHBoxLayout()
        top.addWidget(self.summary, 1)
        top.addWidget(self.details_button)
        right_layout.addLayout(top)
        right_layout.addWidget(self.figure, 1)

        splitter = QtWidgets.QSplitter()
        splitter.addWidget(side)
        splitter.addWidget(right)
        splitter.setStretchFactor(1, 1)
        layout = QtWidgets.QHBoxLayout(self)
        layout.addWidget(splitter)

        self._last_id = -1
        self.layers.edited.connect(self._validate)
        self.nb_cells.valueChanged.connect(self._validate)
        tasks.task_started.connect(lambda _: self._validate())
        tasks.task_finished.connect(lambda _: self._validate())
        temperature_unit.changed.connect(self._draw)
        self._load_defaults()
        self._load_last()

    def _load_defaults(self) -> None:
        previous = self.context.project.computations(self.context.study, self.context.name, "direct")
        if previous:
            config = ForwardConfig.from_dict(previous[0].config)
            self.layers.set_layers(config.layers)
            self.nb_cells.set_value(config.nb_cells)
        else:
            self.layers.set_layers([LayerSpec(t("Layer {n}").format(n=1), self.context.column_depth)])
        self._validate()

    def load_config(self, config: ForwardConfig) -> None:
        """Recharge une configuration enregistrée (pour relancer un calcul à l'identique)."""
        self.layers.set_layers(config.layers)
        self.nb_cells.set_value(config.nb_cells)
        self._validate()

    def config(self) -> ForwardConfig:
        return ForwardConfig(self.layers.layers(), int(self.nb_cells.value()))

    def _validate(self) -> None:
        problems = self.layers.validate()
        if not self.nb_cells.is_valid():
            problems.append(t("Check the number of cells."))
        if self.tasks.busy:
            problems.append(t("A computation is already running."))
        self.issues.setText("\n".join(problems))
        self.run_button.setEnabled(not problems)

    # Calcul ---------------------------------------------------------------------------
    def run(self) -> None:
        if not self.run_button.isEnabled():
            return
        config = self.config()
        inputs = self.context.column_inputs()
        task = Task(t("Direct model — {point}").format(point=self.context.name), lambda ctx: api.run_forward(inputs, config))
        task.signals.succeeded.connect(lambda result: self._finished(task, config, result))
        self.tasks.start(task)

    def _finished(self, task: Task, config: ForwardConfig, result: ForwardResult) -> None:
        self._last_id = self.context.project.add_computation(
            self.context.study, self.context.name, "direct", config.to_dict(), result.to_arrays(), task.duration, api.engine_version(), api.device()
        )
        self._last = result
        self.context.notify()
        self._show()

    def _load_last(self) -> None:
        previous = self.context.project.computations(self.context.study, self.context.name, "direct")
        if previous:
            self._last_id = previous[0].id
            self._last = ForwardResult.from_arrays(self.context.project.load_result(previous[0].id))
            self._show()

    def _show(self) -> None:
        result = self._last
        if result is None:
            return
        rmse = temperature_unit.convert_difference(result.rmse)
        unit = temperature_unit.unit
        self.summary.setText(
            t("RMSE: sensor 1 {a:.3f}, sensor 2 {b:.3f}, sensor 3 {c:.3f}, overall {d:.3f} ({unit})").format(
                a=rmse[0], b=rmse[1], c=rmse[2], d=rmse[3], unit="K" if unit == "K" else unit
            )
        )
        self.details_button.setEnabled(True)
        self._draw()

    def _draw(self) -> None:
        if self._last is None:
            return
        result = self._last
        self.figure.figure.clear()
        axes = self.figure.figure.subplots(3, 1, sharex=True)
        sensor_comparison(
            list(np.atleast_1d(axes)),
            result.times,
            temperature_unit.convert(result.observed),
            temperature_unit.convert(result.simulated_at_sensors()),
            unit=temperature_unit.unit,
            rmse=temperature_unit.convert_difference(result.rmse),
        )
        self.figure.draw()
