"""Vue d'un point : en-tête d'avancement et onglets du flux de travail.

Les onglets dont les prérequis manquent sont désactivés, avec une infobulle qui
explique quoi faire : impossible de lancer un calcul sans mesures nettoyées.
"""

from __future__ import annotations

from PySide6 import QtCore, QtGui, QtWidgets

from ...domain.models import PointStage
from ...i18n import t
from ...storage.project import Project
from ...workers.tasks import TaskManager
from ..theme import ACCENT, muted_color
from .cleanup_tab import CleanupTab
from .context import STAGE_LABELS, PointContext
from .forward_tab import ForwardTab
from .info_tab import InfoTab
from .inversion_tab import InversionTab
from .measures_tab import MeasuresTab
from .results_tab import ResultsTab


class StageBar(QtWidgets.QLabel):
    """« Mesures brutes › Nettoyées › Modèle direct › Inversion » ; étapes faites en couleur d'accent."""

    def set_stage(self, stage: PointStage) -> None:
        parts = []
        for value, label in STAGE_LABELS.items():
            text = t(label)
            if value <= stage:
                parts.append(f"<span style='color:{ACCENT}; font-weight:600'>{text}</span>")
            else:
                parts.append(f"<span style='color:{muted_color()}'>{text}</span>")
        self.setText(" &nbsp;›&nbsp; ".join(parts))


class PointView(QtWidgets.QWidget):
    stage_changed = QtCore.Signal()

    INFO, MEASURES, CLEANUP, FORWARD, INVERSION, RESULTS = range(6)

    def __init__(self, project: Project, study: str, name: str, tasks: TaskManager, parent=None):
        super().__init__(parent)
        self.context = PointContext(project, study, name, self)
        self.tasks = tasks
        title = QtWidgets.QLabel(f"<b>{name}</b>  <span style='color:{muted_color()}'>{study}</span>")
        title.setProperty("role", "heading")
        self.stage_bar = StageBar()
        header = QtWidgets.QHBoxLayout()
        header.addWidget(title)
        header.addStretch()
        header.addWidget(self.stage_bar)

        self.tabs = QtWidgets.QTabWidget()
        self.info_tab = InfoTab(self.context)
        self.measures_tab = MeasuresTab(self.context)
        self.cleanup_tab = CleanupTab(self.context)
        self.forward_tab = ForwardTab(self.context, tasks)
        self.inversion_tab = InversionTab(self.context, tasks)
        self.results_tab = ResultsTab(self.context)
        for widget, label in (
            (self.info_tab, t("Info")),
            (self.measures_tab, t("Measurements")),
            (self.cleanup_tab, t("Cleaning")),
            (self.forward_tab, t("Direct model")),
            (self.inversion_tab, t("Inversion")),
            (self.results_tab, t("Results")),
        ):
            self.tabs.addTab(widget, label)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addLayout(header)
        layout.addWidget(self.tabs, 1)

        self.forward_tab.show_results.connect(self.show_results)
        self.inversion_tab.show_results.connect(self.show_results)
        self.results_tab.load_forward_config.connect(self._reuse_forward)
        self.results_tab.load_inversion_config.connect(self._reuse_inversion)
        self.context.changed.connect(self._update_stage)
        QtGui.QShortcut(QtGui.QKeySequence("F5"), self, self.run_current)
        QtGui.QShortcut(QtGui.QKeySequence("Escape"), self, self.tasks.cancel)
        self._update_stage()
        self.tabs.setCurrentIndex(self.CLEANUP if self.context.stage == PointStage.RAW else self.MEASURES)

    def _update_stage(self) -> None:
        stage = self.context.stage
        self.stage_bar.set_stage(stage)
        cleaned = stage >= PointStage.CLEANED
        computed = stage >= PointStage.DIRECT_MODEL
        for index in (self.FORWARD, self.INVERSION):
            self.tabs.setTabEnabled(index, cleaned)
            self.tabs.setTabToolTip(index, "" if cleaned else t("Clean the measurements first (Cleaning tab)."))
        self.tabs.setTabEnabled(self.RESULTS, computed)
        self.tabs.setTabToolTip(self.RESULTS, "" if computed else t("Run the direct model or an inversion first."))
        self.stage_changed.emit()

    def show_results(self, computation_id: int) -> None:
        self.results_tab.show_computation(computation_id)
        self.tabs.setCurrentIndex(self.RESULTS)

    def run_current(self) -> None:
        if self.tabs.currentIndex() == self.FORWARD:
            self.forward_tab.run()
        elif self.tabs.currentIndex() == self.INVERSION:
            self.inversion_tab.run()

    def _reuse_forward(self, config) -> None:
        self.forward_tab.load_config(config)
        self.tabs.setCurrentIndex(self.FORWARD)

    def _reuse_inversion(self, config) -> None:
        self.inversion_tab.load_config(config)
        self.tabs.setCurrentIndex(self.INVERSION)
