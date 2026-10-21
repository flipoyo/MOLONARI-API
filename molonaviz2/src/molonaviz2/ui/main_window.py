"""Fenêtre principale : projet, menus, panneaux, onglets des points ouverts."""

from __future__ import annotations

from pathlib import Path

from PySide6 import QtCore, QtGui, QtWidgets

from ..domain.models import Study
from ..engine import api
from ..i18n import LANGUAGES, t
from ..io.legacy_import import import_legacy_database
from ..logs import bridge, log
from ..settings import TEMPERATURE_UNITS, Settings
from ..storage.project import SUFFIX, Project
from ..twin.config import TwinConfigError
from ..twin.run import list_twins, twin_status
from ..workers.tasks import Task, TaskManager
from .dialogs.lab_import import LabImportDialog
from .dialogs.point_import import PointImportDialog
from .dialogs.notice import NoticeDialog, key_for
from .dialogs.simple import AboutDialog, AddSensorDialog, NewStudyDialog, QuickGuideDialog
from .errors import error_text, report_exception
from .explorer import ProjectExplorer
from .point_view.view import PointView
from .twin_window import TwinWindow
from .units import temperature_unit
from .widgets import ask, confirm_destructive, heading, hint, message

PROJECT_FILTER = "Molonaviz (*.molo)"


class WelcomePage(QtWidgets.QWidget):
    """Écran sans projet ouvert : créer, ouvrir ou importer un projet, choisir un twin, projets récents."""

    def __init__(self, window: MainWindow):
        super().__init__()
        self.window = window
        layout = QtWidgets.QVBoxLayout(self)
        layout.addStretch()
        box = QtWidgets.QVBoxLayout()
        box.addWidget(heading("Molonaviz"))
        box.addWidget(hint(t("Bayesian inversion of temperature and pressure measurements in the hyporheic zone.")))
        row = QtWidgets.QHBoxLayout()
        new = QtWidgets.QPushButton(t("New project…"))
        new.clicked.connect(window.new_project)
        open_ = QtWidgets.QPushButton(t("Open a project…"))
        open_.clicked.connect(window.open_project_dialog)
        legacy = QtWidgets.QPushButton(t("Import a Molonaviz database…"))
        legacy.clicked.connect(window.import_legacy)
        for button in (new, open_, legacy):
            row.addWidget(button)
        box.addLayout(row)

        self.twins_title = QtWidgets.QLabel(t("Twins"))
        self.twins_title.setProperty("role", "heading")
        self.twin_folder_label = hint("")
        choose = QtWidgets.QPushButton(t("Choose the twins folder…"))
        choose.clicked.connect(window.choose_twin_folder)
        self.reveal_twins = QtWidgets.QPushButton(t("Show in file browser"))
        self.reveal_twins.clicked.connect(window.reveal_twin_folder)
        twin_row = QtWidgets.QHBoxLayout()
        twin_row.addWidget(choose)
        twin_row.addWidget(self.reveal_twins)
        twin_row.addStretch()
        self.twins = QtWidgets.QListWidget()
        self.twins.setMaximumHeight(200)
        self.twins.setToolTip(t("Double-click a twin to open it."))
        self.twins.itemActivated.connect(lambda item: window.open_twin(item.data(QtCore.Qt.ItemDataRole.UserRole)))
        self.twins_empty = hint("")
        box.addSpacing(12)
        box.addWidget(self.twins_title)
        box.addWidget(self.twin_folder_label)
        box.addLayout(twin_row)
        box.addWidget(self.twins)
        box.addWidget(self.twins_empty)
        box.addSpacing(12)

        self.recent_title = QtWidgets.QLabel(t("Recent projects"))
        self.recent = QtWidgets.QListWidget()
        self.recent.setMaximumHeight(180)
        self.recent.itemActivated.connect(lambda item: window.open_project(item.data(QtCore.Qt.ItemDataRole.UserRole)))
        box.addWidget(self.recent_title)
        box.addWidget(self.recent)
        guide = QtWidgets.QPushButton(t("Getting started"))
        guide.setFlat(True)
        guide.clicked.connect(lambda: QuickGuideDialog(window).exec())
        box.addWidget(guide, 0, QtCore.Qt.AlignmentFlag.AlignLeft)
        center = QtWidgets.QHBoxLayout()
        center.addStretch()
        container = QtWidgets.QWidget()
        container.setLayout(box)
        container.setMaximumWidth(560)
        center.addWidget(container, 1)
        center.addStretch()
        layout.addLayout(center)
        layout.addStretch()

    def refresh(self, recent: list[str]) -> None:
        self.recent.clear()
        for path in recent:
            item = QtWidgets.QListWidgetItem(f"{Path(path).name}    {Path(path).parent}")
            item.setData(QtCore.Qt.ItemDataRole.UserRole, path)
            self.recent.addItem(item)
        self.recent.setVisible(bool(recent))
        self.recent_title.setVisible(bool(recent))

    def refresh_twins(self, folder: Path | None) -> None:
        self.twins.clear()
        self.reveal_twins.setEnabled(folder is not None and folder.is_dir())
        if folder is None:
            self.twin_folder_label.setText(t("No twins folder chosen yet."))
            self.twins_empty.setText(t("Choose the folder that contains your twin_xx folders (each with a twin.yaml)."))
        elif not folder.is_dir():
            self.twin_folder_label.setText(str(folder))
            self.twins_empty.setText(t("Twin folder not found: {path}").format(path=folder))
        else:
            self.twin_folder_label.setText(str(folder))
            twins = list_twins(folder)
            for path in twins:
                item = QtWidgets.QListWidgetItem(f"{path.name}    {describe_twin(path)}")
                item.setData(QtCore.Qt.ItemDataRole.UserRole, str(path))
                item.setToolTip(str(path / "twin.yaml"))
                self.twins.addItem(item)
            self.twins_empty.setText("" if twins else t(
                "No twin in this folder yet. Create one from the Results tab of a point (Create a twin), or write a twin_xx/twin.yaml."))
        self.twins.setVisible(self.twins.count() > 0)
        self.twins_empty.setVisible(bool(self.twins_empty.text()))


def describe_twin(path: Path) -> str:
    """Résumé d'une ligne : mode, identifiant, état dans le registre (ou première erreur)."""
    try:
        status = twin_status(path)
    except TwinConfigError as exc:
        return t("invalid: {issue}").format(issue=error_text(exc.issues[0]))
    state = {
        "new": t("not run yet"),
        "computed": t("computed"),
        "duplicate": t("duplicate of {name}").format(name=status.source.name if status.source else ""),
        "modified": t("modified after it was run (frozen)"),
    }[status.state]
    return f"{status.config.mode} · {status.twin_id[:8]} · {state}"


class MainWindow(QtWidgets.QMainWindow):
    language_change_requested = QtCore.Signal(str)

    def __init__(self, settings: Settings, project: Project | None = None):
        super().__init__()
        self.settings = settings
        self.project: Project | None = None
        self._notice: NoticeDialog | None = None
        self.tasks = TaskManager(self)
        self.twin_windows: dict[str, TwinWindow] = {}
        self.setWindowTitle("Molonaviz")
        self.resize(1440, 900)

        self.welcome = WelcomePage(self)
        self.point_tabs = QtWidgets.QTabWidget()
        self.point_tabs.setTabsClosable(True)
        self.point_tabs.setDocumentMode(True)
        self.point_tabs.tabCloseRequested.connect(self._close_point_tab)
        self.empty_project = hint(t("Double-click a point in the Project panel to open it, or import one (Point › Import a point)."))
        self.empty_project.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.central = QtWidgets.QStackedWidget()
        for widget in (self.welcome, self.empty_project, self.point_tabs):
            self.central.addWidget(widget)
        self.setCentralWidget(self.central)

        self.explorer = ProjectExplorer()
        self.explorer.point_activated.connect(self.open_point)
        self.explorer.context_requested.connect(self._explorer_menu)
        self.project_dock = self._dock(t("Project"), self.explorer, QtCore.Qt.DockWidgetArea.LeftDockWidgetArea, "project")
        self.details = QtWidgets.QLabel()
        self.details.setWordWrap(True)
        self.details.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop | QtCore.Qt.AlignmentFlag.AlignLeft)
        self.details.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        self.details.setMargin(8)
        self.explorer.selection_described.connect(self.details.setText)
        self.details_dock = self._dock(t("Details"), self.details, QtCore.Qt.DockWidgetArea.LeftDockWidgetArea, "details")
        self.log_view = QtWidgets.QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(5000)
        clear = QtWidgets.QPushButton(t("Clear"))
        clear.clicked.connect(self.log_view.clear)
        log_panel = QtWidgets.QWidget()
        log_layout = QtWidgets.QVBoxLayout(log_panel)
        log_layout.setContentsMargins(4, 4, 4, 4)
        log_layout.addWidget(self.log_view)
        log_layout.addWidget(clear, 0, QtCore.Qt.AlignmentFlag.AlignRight)
        self.log_dock = self._dock(t("Log"), log_panel, QtCore.Qt.DockWidgetArea.BottomDockWidgetArea, "log")
        self.log_dock.hide()
        bridge.record.connect(self._append_log)

        self._build_status_bar()
        self._build_menus()
        self.tasks.task_started.connect(self._task_started)
        self.tasks.task_finished.connect(self._task_finished)
        geometry, state = settings.window_state()
        if geometry is not None:
            self.restoreGeometry(geometry)
        if state is not None:
            self.restoreState(state)
        if project is not None:
            self._set_project(project)
        else:
            self._set_project(None)

    # Construction ---------------------------------------------------------------------
    def _dock(self, title: str, widget: QtWidgets.QWidget, area, name: str) -> QtWidgets.QDockWidget:
        dock = QtWidgets.QDockWidget(title, self)
        dock.setObjectName(f"dock_{name}")
        dock.setWidget(widget)
        dock.setFeatures(QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetMovable | QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetClosable)
        self.addDockWidget(area, dock)
        return dock

    def _build_status_bar(self) -> None:
        status = self.statusBar()
        self.task_label = QtWidgets.QLabel()
        self.task_progress = QtWidgets.QProgressBar()
        self.task_progress.setMaximumWidth(160)
        self.task_progress.setRange(0, 0)
        self.task_cancel = QtWidgets.QPushButton(t("Cancel"))
        self.task_cancel.clicked.connect(self.tasks.cancel)
        for widget in (self.task_label, self.task_progress, self.task_cancel):
            status.addWidget(widget)
            widget.hide()
        self.project_label = QtWidgets.QLabel()
        status.addPermanentWidget(self.project_label)
        device = QtWidgets.QLabel(api.device())
        device.setToolTip(t("Computation device. GPU is used only with CUDA (Linux); otherwise computations run in float64 on the CPU."))
        status.addPermanentWidget(device)

    def _action(self, menu: QtWidgets.QMenu, text: str, slot, shortcut: str | None = None) -> QtGui.QAction:
        action = menu.addAction(text, slot)
        if shortcut:
            action.setShortcut(QtGui.QKeySequence(shortcut))
        return action

    def _build_menus(self) -> None:
        bar = self.menuBar()
        file_menu = bar.addMenu(t("&File"))
        file_menu.setToolTipsVisible(True)
        self._action(file_menu, t("New project…"), self.new_project, "Ctrl+N")
        self._action(file_menu, t("Open a project…"), self.open_project_dialog, "Ctrl+O")
        self.recent_menu = file_menu.addMenu(t("Recent projects"))
        self.twins_menu = file_menu.addMenu(t("Twins"))
        self.twins_menu.setToolTipsVisible(True)
        self.twins_menu.aboutToShow.connect(self._refresh_twins_menu)
        self._refresh_twins_menu()
        file_menu.addSeparator()
        self._action(file_menu, t("Import a Molonaviz database…"), self.import_legacy)
        file_menu.addSeparator()
        self.close_action = self._action(file_menu, t("Close project"), self.close_project, "Ctrl+W")
        self._action(file_menu, t("Quit"), self.close, "Ctrl+Q")

        lab_menu = bar.addMenu(t("&Laboratory"))
        lab_menu.setToolTipsVisible(True)
        self.import_lab_action = self._action(lab_menu, t("Import a laboratory…"), self.import_lab, "Ctrl+L")
        self.add_sensor_action = self._action(lab_menu, t("Add a sensor…"), lambda: self.add_sensor(None))

        study_menu = bar.addMenu(t("&Study"))
        study_menu.setToolTipsVisible(True)
        self.new_study_action = self._action(study_menu, t("New study…"), self.new_study, "Ctrl+Shift+N")

        point_menu = bar.addMenu(t("&Point"))
        point_menu.setToolTipsVisible(True)
        self.import_point_action = self._action(point_menu, t("Import a point…"), lambda: self.import_point(None), "Ctrl+I")

        view_menu = bar.addMenu(t("&View"))
        units = view_menu.addMenu(t("Temperature unit"))
        group = QtGui.QActionGroup(self)
        for unit in TEMPERATURE_UNITS:
            action = units.addAction(unit, lambda u=unit: self.set_temperature_unit(u))
            action.setCheckable(True)
            action.setChecked(unit == temperature_unit.unit)
            group.addAction(action)
        languages = view_menu.addMenu(t("Language"))
        group = QtGui.QActionGroup(self)
        for code, name in LANGUAGES.items():
            action = languages.addAction(name, lambda c=code: self.language_change_requested.emit(c))
            action.setCheckable(True)
            action.setChecked(code == self.settings.language)
            group.addAction(action)
        view_menu.addSeparator()
        for dock in (self.project_dock, self.details_dock, self.log_dock):
            view_menu.addAction(dock.toggleViewAction())

        help_menu = bar.addMenu(t("&Help"))
        self._action(help_menu, t("Getting started"), lambda: QuickGuideDialog(self).exec(), "F1")
        self._action(help_menu, t("Parameter guide"), self.open_notice, "Shift+F1")
        self._action(help_menu, t("Show log folder"), self._show_logs)
        self._action(help_menu, t("About Molonaviz"), lambda: AboutDialog(api.engine_version(), api.device(), self).exec())

    def open_notice(self, key: str | None = None) -> NoticeDialog:
        """Notice des paramètres, ouverte sur la fiche du champ qui a le focus s'il en a une."""
        key = key or key_for(QtWidgets.QApplication.focusWidget())
        if self._notice is None:
            self._notice = NoticeDialog(self)
        if key:
            self._notice.show_entry(key)
        self._notice.show()
        self._notice.raise_()
        self._notice.activateWindow()
        return self._notice

    # Projet ---------------------------------------------------------------------------
    def _set_project(self, project: Project | None) -> None:
        self.project = project
        self.explorer.set_project(project)
        self.point_tabs.clear()
        for dock in (self.project_dock, self.details_dock):
            dock.setVisible(project is not None)
        if project is None:
            self.welcome.refresh(self.settings.recent_projects())
            self.welcome.refresh_twins(self.settings.twin_folder)
            self.central.setCurrentWidget(self.welcome)
            self.setWindowTitle("Molonaviz")
            self.project_label.setText("")
        else:
            self.settings.add_recent_project(project.path)
            self.central.setCurrentWidget(self.empty_project)
            self.setWindowTitle(f"{project.path.stem} — Molonaviz")
            self.project_label.setText(str(project.path))
            log.info("Project opened: %s", project.path)
        self._refresh_actions()
        self._refresh_recent_menu()

    def _refresh_actions(self) -> None:
        has_project = self.project is not None
        labs = self.project.lab_names() if has_project else []
        studies = self.project.studies() if has_project else []
        rules = (
            (self.close_action, has_project, t("No project is open.")),
            (self.import_lab_action, has_project, t("Create or open a project first (File menu).")),
            (self.add_sensor_action, bool(labs), t("Import or create a laboratory first.")),
            (self.new_study_action, bool(labs), t("A study needs a laboratory: import one first (Laboratory menu).")),
            (self.import_point_action, bool(studies), t("Create a study first (Study menu).")),
        )
        for action, enabled, reason in rules:
            action.setEnabled(enabled)
            action.setToolTip("" if enabled else reason)
            action.setStatusTip("" if enabled else reason)

    def _refresh_recent_menu(self) -> None:
        self.recent_menu.clear()
        recent = self.settings.recent_projects()
        for path in recent:
            self.recent_menu.addAction(Path(path).name, lambda p=path: self.open_project(p)).setToolTip(path)
        self.recent_menu.setEnabled(bool(recent))

    def _refresh_twins_menu(self) -> None:
        menu = self.twins_menu
        menu.clear()
        folder = self.settings.twin_folder
        twins = list_twins(folder) if folder is not None else []
        for path in twins:
            action = menu.addAction(path.name, lambda p=path: self.open_twin(p))
            action.setToolTip(describe_twin(path))
        if folder is not None and not twins:
            menu.addAction(t("No twin in this folder")).setEnabled(False)
        menu.addSeparator()
        menu.addAction(t("Choose the twins folder…"), self.choose_twin_folder)
        reveal = menu.addAction(t("Show in file browser"), self.reveal_twin_folder)
        reveal.setEnabled(folder is not None and folder.is_dir())

    def choose_twin_folder(self) -> bool:
        current = self.settings.twin_folder
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, t("Twins folder"), str(current or Path.home()))
        if not folder:
            return False
        self.settings.twin_folder = folder
        log.info("Twins folder: %s", folder)
        self.welcome.refresh_twins(self.settings.twin_folder)
        return True

    def reveal_twin_folder(self) -> None:
        if self.settings.twin_folder is not None:
            QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(self.settings.twin_folder)))

    def open_twin(self, path: str | Path) -> TwinWindow:
        key = str(Path(path).resolve())
        window = self.twin_windows.get(key)
        if window is None or not window.isVisible():
            window = TwinWindow(path, self.tasks, self)
            window.setWindowFlag(QtCore.Qt.WindowType.Window)
            window.destroyed.connect(lambda *_: self.twin_windows.pop(key, None))
            self.twin_windows[key] = window
        else:
            window.refresh()
        window.show()
        window.raise_()
        return window

    def create_twin_from_computation(self, context, info) -> Path | None:
        """Fige un calcul de l'application en twin : twin.yaml + mesures nettoyées du point."""
        from ..domain.compute import ForwardConfig, InversionConfig
        from ..twin.create import create_twin

        folder = self.settings.twin_folder
        if folder is None or not folder.is_dir():
            message(self, t("Create a twin"), t("Choose the twins folder first: twins are stored there."))
            if not self.choose_twin_folder():
                return None
            folder = self.settings.twin_folder
        sensors = context.sensors()
        if info.kind == "inversion":
            config = InversionConfig.from_dict(info.config)
            sigma_t = config.sigma_t
        else:
            config = ForwardConfig.from_dict(info.config)
            sigma_t = context.default_sigma_t()
        created = create_twin(
            folder, context.project.cleaned(context.study, context.name), context.name,
            (t("Created from the inversion of {date}") if info.kind == "inversion" else t("Created from the direct model of {date}")).format(
                date=info.created_at.strftime("%Y-%m-%d %H:%M")),
            sensors.shaft.depths, context.point.offset, sigma_t, sensors.pressure.sigma, config,
        )
        log.info("Twin created: %s", created)
        if ask(self, t("Twin created"), t("{name} was created in {folder}. Its configuration is now frozen.").format(name=created.name, folder=folder), t("Open")):
            self.open_twin(created)
        return created

    def _project_changed(self) -> None:
        self.explorer.refresh()
        self._refresh_actions()

    def new_project(self) -> None:
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, t("New project"), str(Path.home() / f"projet{SUFFIX}"), PROJECT_FILTER)
        if path:
            self._replace_project(lambda: Project.create(path))

    def open_project_dialog(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, t("Open a project"), str(Path.home()), PROJECT_FILTER)
        if path:
            self.open_project(path)

    def open_project(self, path: str | Path) -> None:
        self._replace_project(lambda: Project.open(path))

    def _replace_project(self, factory) -> bool:
        if not self._confirm_stop_task():
            return False
        project = factory()  # une erreur ici laisse le projet courant ouvert
        if self.project is not None:
            self.project.close()
        self._set_project(project)
        return True

    def close_project(self) -> None:
        if self.project is not None and self._confirm_stop_task():
            self.project.close()
            self._set_project(None)

    def import_legacy(self) -> None:
        source, _ = QtWidgets.QFileDialog.getOpenFileName(self, t("Molonaviz database"), str(Path.home()), "Molonari.sqlite (*.sqlite);;SQLite (*.sqlite *.db)")
        if not source:
            return
        if self.project is None:
            message(self, t("Import a Molonaviz database"), t("Choose where to save the new project that will receive the data."))
            target, _ = QtWidgets.QFileDialog.getSaveFileName(self, t("New project"), str(Path(source).with_suffix(SUFFIX)), PROJECT_FILTER)
            if not target or not self._replace_project(lambda: Project.create(target)):
                return
        report = import_legacy_database(Path(source), self.project)
        self._project_changed()
        text = t("Imported: {labs} laboratories, {studies} studies, {points} points ({cleaned} with cleaned measurements).").format(
            labs=report.labs, studies=report.studies, points=report.points, cleaned=report.cleaned_points
        )
        text += "\n\n" + t("Previous computation results are not imported: run the models again.")
        message(self, t("Import finished"), text, "\n".join(error_text(warning) for warning in report.warnings))

    # Laboratoires, études, points -----------------------------------------------------
    def import_lab(self) -> None:
        dialog = LabImportDialog(self.project.lab_names(), self)
        if dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            self.project.add_lab(dialog.lab())
            log.info("Laboratory imported: %s", dialog.lab().name)
            self._project_changed()

    def add_sensor(self, lab_name: str | None) -> None:
        labs = self.project.lab_names()
        if lab_name is None:
            if len(labs) == 1:
                lab_name = labs[0]
            else:
                lab_name, ok = QtWidgets.QInputDialog.getItem(self, t("Add a sensor"), t("Laboratory"), labs, 0, False)
                if not ok:
                    return
        lab = self.project.get_lab(lab_name)
        dialog = AddSensorDialog(lab_name, [th.name for th in lab.thermometers], self)
        if dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            self.project.add_lab_item(lab_name, dialog.item())
            self._project_changed()

    def new_study(self) -> None:
        dialog = NewStudyDialog(self.project.lab_names(), [s.name for s in self.project.studies()], self)
        if dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            self.project.add_study(Study(dialog.name.text().strip(), dialog.lab.currentText()))
            self._project_changed()

    def _current_study(self) -> str | None:
        item = self.explorer.currentItem()
        while item is not None:
            kind = item.data(0, QtCore.Qt.ItemDataRole.UserRole)
            payload = item.data(0, QtCore.Qt.ItemDataRole.UserRole + 1)
            if kind == "study":
                return payload
            if kind == "point":
                return payload[0]
            item = item.parent()
        studies = [s.name for s in self.project.studies()]
        if len(studies) == 1:
            return studies[0]
        study, ok = QtWidgets.QInputDialog.getItem(self, t("Import a point"), t("Study"), studies, 0, False)
        return study if ok else None

    def import_point(self, study: str | None) -> None:
        study = study or self._current_study()
        if study is None:
            return
        dialog = PointImportDialog(study, self.project.study_lab(study), self.project.point_names(study), self)
        if dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            point, raw_t, raw_p = dialog.point_data()
            self.project.add_point(point, raw_t, raw_p)
            log.info("Point imported: %s (%s)", point.name, study)
            self._project_changed()
            self.explorer.select_point(study, point.name)
            self.open_point(study, point.name)

    def open_point(self, study: str, name: str) -> None:
        for index in range(self.point_tabs.count()):
            view = self.point_tabs.widget(index)
            if view.context.study == study and view.context.name == name:
                self.point_tabs.setCurrentIndex(index)
                return
        view = PointView(self.project, study, name, self.tasks)
        view.stage_changed.connect(self.explorer.refresh)
        view.results_tab.create_twin_requested.connect(lambda info, v=view: self.create_twin_from_computation(v.context, info))
        index = self.point_tabs.addTab(view, name)
        self.point_tabs.setTabToolTip(index, f"{name} — {study}")
        self.point_tabs.setCurrentIndex(index)
        self.central.setCurrentWidget(self.point_tabs)

    def _close_point_tab(self, index: int) -> None:
        view = self.point_tabs.widget(index)
        self.point_tabs.removeTab(index)
        view.deleteLater()
        if self.point_tabs.count() == 0 and self.project is not None:
            self.central.setCurrentWidget(self.empty_project)

    def _close_views(self, study: str | None = None, name: str | None = None) -> None:
        for index in reversed(range(self.point_tabs.count())):
            context = self.point_tabs.widget(index).context
            if (study is None or context.study == study) and (name is None or context.name == name):
                self._close_point_tab(index)

    def _explorer_menu(self, kind: str, payload, position: QtCore.QPoint) -> None:
        menu = QtWidgets.QMenu(self)
        if kind == "lab":
            menu.addAction(t("Add a sensor…"), lambda: self.add_sensor(payload))
            menu.addAction(t("Delete laboratory…"), lambda: self._delete_lab(payload))
        elif kind == "study":
            menu.addAction(t("Import a point…"), lambda: self.import_point(payload))
            menu.addAction(t("Delete study…"), lambda: self._delete_study(payload))
        elif kind == "point":
            study, name = payload
            menu.addAction(t("Open"), lambda: self.open_point(study, name))
            menu.addAction(t("Reset to raw measurements…"), lambda: self._reset_point(study, name))
            menu.addAction(t("Delete point…"), lambda: self._delete_point(study, name))
        if not menu.isEmpty():
            menu.exec(position)

    def _delete_lab(self, name: str) -> None:
        if confirm_destructive(self, t("Delete laboratory"), t("The laboratory and all its sensors will be deleted."), name, t("Delete")):
            self.project.delete_lab(name)
            self._project_changed()

    def _delete_study(self, name: str) -> None:
        count = len(self.project.point_names(name))
        if confirm_destructive(self, t("Delete study"), t("The study and its {n} point(s), with all measurements and results, will be deleted.").format(n=count), name, t("Delete")):
            self._close_views(study=name)
            self.project.delete_study(name)
            self._project_changed()

    def _delete_point(self, study: str, name: str) -> None:
        if confirm_destructive(self, t("Delete point"), t("The point, its measurements and its results will be deleted."), name, t("Delete")):
            self._close_views(study, name)
            self.project.delete_point(study, name)
            self._project_changed()

    def _reset_point(self, study: str, name: str) -> None:
        if confirm_destructive(self, t("Reset point"), t("Cleaned measurements and all results of this point will be deleted. Raw measurements are kept."), name, t("Reset")):
            self._close_views(study, name)
            self.project.reset_point(study, name)
            self._project_changed()

    # Tâches ---------------------------------------------------------------------------
    def _task_started(self, task: Task) -> None:
        self.task_label.setText(task.name)
        for widget in (self.task_label, self.task_progress, self.task_cancel):
            widget.show()
        task.signals.failed.connect(lambda exc: report_exception(exc, task.name))

    def _task_finished(self, task: Task) -> None:
        for widget in (self.task_label, self.task_progress, self.task_cancel):
            widget.hide()
        if not task.cancel_requested:
            log.info("Task finished: %s (%.1f s)", task.name, task.duration)

    def _confirm_stop_task(self) -> bool:
        if not self.tasks.busy:
            return True
        if not ask(self, t("A computation is running"), t("Stop the running computation? Its results will not be saved."), t("Stop")):
            return False
        self.tasks.cancel()
        self.tasks.wait()
        QtCore.QCoreApplication.processEvents()
        return True

    # Divers ---------------------------------------------------------------------------
    def set_temperature_unit(self, unit: str) -> None:
        self.settings.temperature_unit = unit
        temperature_unit.set(unit)

    def _append_log(self, level: str, text: str) -> None:
        stamp = QtCore.QTime.currentTime().toString("HH:mm:ss")
        self.log_view.appendPlainText(f"{stamp}  {text}")
        if level in ("WARNING", "ERROR") and not self.log_dock.isVisible():
            self.log_dock.show()

    def _show_logs(self) -> None:
        from ..logs import log_directory

        QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(log_directory())))

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:  # noqa: N802
        if not self._confirm_stop_task():
            event.ignore()
            return
        self.settings.save_window_state(self.saveGeometry(), self.saveState())
        if self.project is not None and not getattr(self, "_keep_project_open", False):
            self.project.close()
        event.accept()

    def detach_project(self) -> Project | None:
        """Pour changer de langue : la nouvelle fenêtre reprend le projet ouvert."""
        self._keep_project_open = True
        return self.project

