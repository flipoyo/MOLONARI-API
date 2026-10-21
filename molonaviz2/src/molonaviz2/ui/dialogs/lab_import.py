"""Import d'un laboratoire : choix du dossier, aperçu des capteurs lus et des fichiers écartés."""

from __future__ import annotations

from pathlib import Path

from PySide6 import QtWidgets

from ...errors import UserError
from ...i18n import t
from ...io.lab_import import LabImport, import_lab_directory
from ..errors import error_text
from ..widgets import format_number, hint


class LabImportDialog(QtWidgets.QDialog):
    def __init__(self, existing_names: list[str], parent=None):
        super().__init__(parent)
        self.setWindowTitle(t("Import a laboratory"))
        self.resize(720, 560)
        self._existing = set(existing_names)
        self.result_import: LabImport | None = None

        self.folder = QtWidgets.QLineEdit()
        self.folder.setReadOnly(True)
        self.folder.setPlaceholderText(t("Folder containing temperature_sensors, pressure_sensors and shafts"))
        browse = QtWidgets.QPushButton(t("Choose folder…"))
        browse.clicked.connect(self._browse)
        self.name = QtWidgets.QLineEdit()
        self.name_error = QtWidgets.QLabel()
        self.name_error.setProperty("role", "error")

        form = QtWidgets.QFormLayout()
        row = QtWidgets.QHBoxLayout()
        row.addWidget(self.folder, 1)
        row.addWidget(browse)
        form.addRow(t("Folder"), row)
        form.addRow(t("Laboratory name"), self.name)
        form.addRow("", self.name_error)

        self.tree = QtWidgets.QTreeWidget()
        self.tree.setHeaderLabels([t("Sensor"), t("Details")])
        self.tree.setColumnWidth(0, 200)
        self.issues = QtWidgets.QListWidget()
        self.issues.setMaximumHeight(110)
        self.issues_label = QtWidgets.QLabel(t("Files not imported"))

        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Cancel)
        self.import_button = buttons.addButton(t("Import"), QtWidgets.QDialogButtonBox.ButtonRole.AcceptRole)
        self.import_button.setEnabled(False)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(hint(t("Nothing is written to the project until you click Import.")))
        layout.addWidget(self.tree, 1)
        layout.addWidget(self.issues_label)
        layout.addWidget(self.issues)
        layout.addWidget(buttons)
        self.issues_label.hide()
        self.issues.hide()
        self.name.textChanged.connect(self._validate)

    def _browse(self) -> None:
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, t("Laboratory folder"))
        if folder:
            self.load(Path(folder))

    def load(self, folder: Path) -> None:
        self.folder.setText(str(folder))
        self.tree.clear()
        self.issues.clear()
        try:
            self.result_import = import_lab_directory(folder)
        except UserError as exc:
            self.result_import = None
            self.issues_label.show()
            self.issues.show()
            self.issues.addItem(error_text(exc))
            self._validate()
            return
        lab = self.result_import.lab
        if not self.name.text():
            self.name.setText(lab.name)
        groups = (
            (t("Thermometers"), lab.thermometers, lambda s: f"{s.manufacturer} {s.reference} · σ = {format_number(s.sigma)} °C"),
            (
                t("Pressure sensors"),
                lab.pressure_sensors,
                lambda s: f"{s.datalogger} · dU/dH = {format_number(s.du_dh)} · dU/dT = {format_number(s.du_dt)} · "
                + t("thermometer {name}").format(name=s.thermometer),
            ),
            (t("Shafts"), lab.shafts, lambda s: t("depths {d} m").format(d=", ".join(format_number(d) for d in s.depths))),
        )
        for title, items, describe in groups:
            parent = QtWidgets.QTreeWidgetItem([f"{title} ({len(items)})"])
            self.tree.addTopLevelItem(parent)
            for item in items:
                QtWidgets.QTreeWidgetItem(parent, [item.name, describe(item)])
            parent.setExpanded(True)
        for issue in self.result_import.issues:
            self.issues.addItem(str(issue))
        has_issues = bool(self.result_import.issues)
        self.issues_label.setVisible(has_issues)
        self.issues.setVisible(has_issues)
        self._validate()

    def _validate(self) -> None:
        name = self.name.text().strip()
        error = ""
        if self.result_import is not None:
            if not name:
                error = t("The laboratory needs a name.")
            elif name in self._existing:
                error = t("A laboratory named \"{name}\" already exists.").format(name=name)
        self.name_error.setText(error)
        self.import_button.setEnabled(self.result_import is not None and not error)

    def lab(self):
        assert self.result_import is not None
        self.result_import.lab.name = self.name.text().strip()
        return self.result_import.lab
