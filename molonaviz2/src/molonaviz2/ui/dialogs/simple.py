"""Petits dialogues : nouvelle étude, ajout d'un capteur, à propos, guide de démarrage."""

from __future__ import annotations

from itertools import pairwise

from PySide6 import QtCore, QtWidgets

from ... import __version__
from ...domain.models import PressureSensor, Shaft, Thermometer
from ...i18n import N_, t
from ..widgets import NumberField, heading, hint


class NewStudyDialog(QtWidgets.QDialog):
    def __init__(self, labs: list[str], existing: list[str], parent=None):
        super().__init__(parent)
        self.setWindowTitle(t("New study"))
        self._existing = set(existing)
        self.name = QtWidgets.QLineEdit()
        self.lab = QtWidgets.QComboBox()
        self.lab.addItems(labs)
        self.error = QtWidgets.QLabel()
        self.error.setProperty("role", "error")
        form = QtWidgets.QFormLayout()
        form.addRow(t("Name"), self.name)
        form.addRow(t("Laboratory"), self.lab)
        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Cancel)
        self.create = buttons.addButton(t("Create"), QtWidgets.QDialogButtonBox.ButtonRole.AcceptRole)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout = QtWidgets.QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(hint(t("A study gathers sampling points measured with the sensors of one laboratory.")))
        layout.addWidget(self.error)
        layout.addWidget(buttons)
        self.name.textChanged.connect(self._validate)
        self._validate()
        self.setMinimumWidth(420)

    def _validate(self) -> None:
        name = self.name.text().strip()
        error = t("A study named \"{name}\" already exists.").format(name=name) if name in self._existing else ""
        self.error.setText(error)
        self.create.setEnabled(bool(name) and not error and self.lab.count() > 0)


class AddSensorDialog(QtWidgets.QDialog):
    """Ajout manuel d'un thermomètre, d'un capteur de pression ou d'une tige à un laboratoire."""

    def __init__(self, lab_name: str, thermometers: list[str], parent=None):
        super().__init__(parent)
        self.setWindowTitle(t("Add a sensor to {lab}").format(lab=lab_name))
        self.kind = QtWidgets.QComboBox()
        self.kind.addItems([t("Thermometer"), t("Pressure sensor"), t("Shaft")])
        self.stack = QtWidgets.QStackedWidget()

        self.th_name, self.th_manu, self.th_ref = QtWidgets.QLineEdit(), QtWidgets.QLineEdit(), QtWidgets.QLineEdit()
        self.th_sigma = NumberField(0.05, "°C", minimum=0.0)
        page = QtWidgets.QWidget()
        form = QtWidgets.QFormLayout(page)
        form.addRow(t("Name"), self.th_name)
        form.addRow(t("Manufacturer"), self.th_manu)
        form.addRow(t("Reference"), self.th_ref)
        form.addRow(t("Uncertainty σ"), self.th_sigma)
        self.stack.addWidget(page)

        self.ps_name, self.ps_logger = QtWidgets.QLineEdit(), QtWidgets.QLineEdit()
        self.ps_date = QtWidgets.QDateTimeEdit(QtCore.QDateTime.currentDateTime())
        self.ps_date.setCalendarPopup(True)
        self.ps_intercept = NumberField(0.0, "V")
        self.ps_dudh = NumberField(1.0, "V/m")
        self.ps_dudt = NumberField(0.0, "V/°C")
        self.ps_sigma = NumberField(0.01, "m", minimum=0.0)
        self.ps_thermo = QtWidgets.QComboBox()
        self.ps_thermo.addItems(thermometers)
        page = QtWidgets.QWidget()
        form = QtWidgets.QFormLayout(page)
        form.addRow(t("Name"), self.ps_name)
        form.addRow(t("Datalogger"), self.ps_logger)
        form.addRow(t("Calibration date"), self.ps_date)
        form.addRow(t("Intercept"), self.ps_intercept)
        form.addRow("dU/dH", self.ps_dudh)
        form.addRow("dU/dT", self.ps_dudt)
        form.addRow(t("Uncertainty σ"), self.ps_sigma)
        form.addRow(t("Thermometer"), self.ps_thermo)
        self.stack.addWidget(page)

        self.sh_name, self.sh_logger = QtWidgets.QLineEdit(), QtWidgets.QLineEdit()
        self.sh_thermo = QtWidgets.QComboBox()
        self.sh_thermo.addItems(thermometers)
        self.sh_depths = [NumberField(d, "m", minimum=0.0) for d in (0.1, 0.2, 0.3, 0.4)]
        page = QtWidgets.QWidget()
        form = QtWidgets.QFormLayout(page)
        form.addRow(t("Name"), self.sh_name)
        form.addRow(t("Datalogger"), self.sh_logger)
        form.addRow(t("Thermometer"), self.sh_thermo)
        for i, field in enumerate(self.sh_depths):
            form.addRow(t("Depth of sensor {n}").format(n=i + 1), field)
        self.stack.addWidget(page)

        self.error = QtWidgets.QLabel()
        self.error.setProperty("role", "error")
        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Cancel)
        self.add = buttons.addButton(t("Add"), QtWidgets.QDialogButtonBox.ButtonRole.AcceptRole)
        buttons.accepted.connect(self._accept)
        buttons.rejected.connect(self.reject)
        layout = QtWidgets.QVBoxLayout(self)
        top = QtWidgets.QFormLayout()
        top.addRow(t("Type"), self.kind)
        layout.addLayout(top)
        layout.addWidget(self.stack)
        layout.addWidget(self.error)
        layout.addWidget(buttons)
        self.kind.currentIndexChanged.connect(self.stack.setCurrentIndex)
        if not thermometers:
            self.kind.model().item(1).setEnabled(False)
            self.kind.model().item(2).setEnabled(False)
            self.error.setText(t("Add a thermometer first: pressure sensors and shafts refer to one."))

    def item(self) -> Thermometer | PressureSensor | Shaft | None:
        index = self.kind.currentIndex()
        if index == 0:
            if not self.th_name.text().strip() or self.th_sigma.value() is None:
                return None
            return Thermometer(self.th_name.text().strip(), self.th_manu.text().strip(), self.th_ref.text().strip(), self.th_sigma.value())
        if index == 1:
            values = [f.value() for f in (self.ps_intercept, self.ps_dudh, self.ps_dudt, self.ps_sigma)]
            if not self.ps_name.text().strip() or None in values or values[1] == 0:
                return None
            return PressureSensor(self.ps_name.text().strip(), self.ps_logger.text().strip(), self.ps_date.dateTime().toPython(),
                                  values[0], values[1], values[2], values[3], self.ps_thermo.currentText())
        depths = [f.value() for f in self.sh_depths]
        if not self.sh_name.text().strip() or None in depths or any(b <= a for a, b in pairwise(depths)) or depths[0] <= 0:
            return None
        return Shaft(self.sh_name.text().strip(), self.sh_logger.text().strip(), self.sh_thermo.currentText(), tuple(depths))

    def _accept(self) -> None:
        if self.item() is None:
            self.error.setText(t("Fill in every field. dU/dH cannot be 0; shaft depths must increase."))
            return
        self.accept()


class AboutDialog(QtWidgets.QDialog):
    def __init__(self, engine: str, device: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle(t("About Molonaviz"))
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(heading(f"Molonaviz {__version__}"))
        text = QtWidgets.QLabel(
            t("Bayesian inversion of temperature and pressure measurements in the hyporheic zone (MOLONARI project).")
            + "<br><br>"
            + t("Computation engine: {engine}").format(engine=engine)
            + "<br>"
            + t("Device: {device}").format(device=device)
        )
        text.setWordWrap(True)
        layout.addWidget(text)
        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.setMinimumWidth(440)


GUIDE_STEPS = [
    (N_("Create or open a project"), N_("File › New project. A project is a single .molo file that holds everything.")),
    (N_("Import a laboratory"), N_("Laboratory › Import a laboratory: the folder of thermometers, pressure sensors and shafts.")),
    (N_("Create a study"), N_("Study › New study, linked to that laboratory.")),
    (N_("Import a point"), N_("Point › Import a point: select the point folder; files and fields are filled in automatically.")),
    (N_("Clean the measurements"), N_("In the Cleaning tab, choose the period and the outlier methods, then apply.")),
    (N_("Run the direct model"), N_("Define the layers and run the model (F5) to compare with the measurements.")),
    (N_("Run the inversion"), N_("Set the prior ranges and run DREAM. Follow convergence live; cancel at any time (Esc).")),
]


class QuickGuideDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(t("Getting started"))
        layout = QtWidgets.QVBoxLayout(self)
        for i, (title, body) in enumerate(GUIDE_STEPS, start=1):
            label = QtWidgets.QLabel(f"<b>{i}. {t(title)}</b><br>{t(body)}")
            label.setWordWrap(True)
            layout.addWidget(label)
        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.setMinimumWidth(520)
