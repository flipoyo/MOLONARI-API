"""Point d'entrée de Molonaviz 2."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PySide6 import QtCore, QtWidgets

from . import __version__


class Application(QtCore.QObject):
    """Crée la fenêtre principale et la recrée quand la langue change (le projet reste ouvert)."""

    def __init__(self, app: QtWidgets.QApplication, project_path: str | None = None):
        super().__init__()
        from .i18n import install_language
        from .settings import Settings
        from .ui.units import temperature_unit

        self.app = app
        self.settings = Settings()
        install_language(app, self.settings.language)
        temperature_unit.set(self.settings.temperature_unit)
        self.window = None
        project = None
        if project_path:
            from molonaviz2.facade import Project

            from .ui.errors import report_exception

            try:
                project = Project.open(project_path)
            except Exception as exc:  # noqa: BLE001 — affiché, l'application démarre quand même
                report_exception(exc)
        self._show(project)

    def _show(self, project) -> None:
        from .ui.main_window import MainWindow

        self.window = MainWindow(self.settings, project)
        self.window.language_change_requested.connect(self.change_language)
        self.window.show()

    def change_language(self, language: str) -> None:
        from .i18n import install_language

        if language == self.settings.language:
            return
        old = self.window
        if old.tasks.busy:
            from .i18n import t
            from .ui.widgets import message

            message(old, t("Language"), t("Wait for the running computation to finish before changing the language."))
            return
        self.settings.language = language
        install_language(self.app, language)
        project = old.detach_project()
        old.close()
        old.deleteLater()
        self._show(project)


def selftest() -> int:
    """Vérifie l'application installée sans fenêtre visible : modèle direct, mini-inversion,
    diagnostic ArviZ, export NetCDF, construction de la fenêtre principale."""
    import os
    import tempfile

    import numpy as np

    from molonaviz2 import facade as api
    from molonaviz2.facade import (
        ColumnInputs,
        ForwardConfig,
        InversionConfig,
        LayerSpec,
    )

    n = 96
    i = np.arange(n)
    inputs = ColumnInputs(
        times=(np.datetime64("2020-01-01T00:00") + i * np.timedelta64(15, "m")).astype("datetime64[s]"),
        head=np.full(n, 0.05),
        t_river=12 + 2 * np.sin(2 * np.pi * i / 96),
        t_sensors=np.stack([12 + np.sin(2 * np.pi * i / 96 - k) for k in (1, 2, 3)], axis=1),
        t_aquifer=np.full(n, 11.0),
        sensor_depths=(0.1, 0.2, 0.3, 0.4),
        offset=0.0,
        sigma_pressure=0.01,
        sigma_temperature=0.1,
    )
    checks: list[tuple[str, bool]] = []
    result = api.run_forward(inputs, ForwardConfig([LayerSpec("l", 0.4)], nb_cells=40))
    checks.append(("direct model", bool(np.all(np.isfinite(result.temperature))) and result.temperature.shape == (40, n)))

    config = InversionConfig(layers=[LayerSpec("l", 0.4)], nb_cells=20, nb_chains=7, nb_iter=8, burn_in_max=4, n_field_samples=7)
    inversion = api.run_inversion(inputs, config)
    checks.append(("inversion", inversion.samples.shape == (8, 7, 1, 4)))

    table = api.summary_table(inversion, ["l"])
    checks.append(("ArviZ summary", len(table) == 4))
    with tempfile.TemporaryDirectory() as folder:
        path = os.path.join(folder, "posterior.nc")
        api.export_netcdf(inversion, ["l"], path)
        checks.append(("NetCDF export", os.path.getsize(path) > 0))

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    from .i18n import install_language
    from .settings import Settings
    from .ui.main_window import MainWindow

    install_language(app, "fr")
    with tempfile.TemporaryDirectory() as folder:
        window = MainWindow(Settings(QtCore.QSettings(os.path.join(folder, "s.ini"), QtCore.QSettings.Format.IniFormat)))
        checks.append(("main window", window.menuBar().actions()[0].text() == "&Fichier"))
        window.close()

    ok = all(passed for _, passed in checks)
    for name, passed in checks:
        print(f"  {'ok  ' if passed else 'FAIL'} {name}")
    print(f"Molonaviz {__version__} self-test: {'OK' if ok else 'FAILED'} ({api.engine_version()}, {api.device()})")
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv if argv is None else argv)
    if len(argv) > 1 and argv[1] == "twin":
        from molonaviz2.facade import twin_cli

        return twin_cli(argv[2:])
    parser = argparse.ArgumentParser(prog="molonaviz2", description="Molonaviz 2")
    parser.add_argument("project", nargs="?", help="project file (.molo) to open")
    parser.add_argument("--selftest", action="store_true", help="run a small computation without a window and exit")
    parser.add_argument("--version", action="version", version=f"Molonaviz {__version__}")
    args, qt_args = parser.parse_known_args(argv[1:])
    if args.selftest:
        return selftest()

    QtCore.QCoreApplication.setOrganizationName("MOLONARI")
    QtCore.QCoreApplication.setApplicationName("Molonaviz2")
    QtCore.QCoreApplication.setApplicationVersion(__version__)
    app = QtWidgets.QApplication([argv[0], *qt_args])

    from .logs import setup_logging
    from .ui.errors import install_excepthook
    from .ui.theme import apply_theme

    setup_logging()
    apply_theme(app)
    install_excepthook()
    controller = Application(app, args.project if args.project and Path(args.project).exists() else None)  # noqa: F841 (garde la fenêtre en vie)
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
