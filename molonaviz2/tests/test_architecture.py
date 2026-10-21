"""Architecture : le cœur (futur backend) s'importe et s'exécute sans Qt ni interface, et ne produit
aucune phrase traduite ; l'interface n'ajoute pas de nouvel import du cœur hors façade.

Ce fichier ne dépend pas de PySide6 : la CI le lance seul (job core-without-qt).
"""

import ast
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[1] / "src" / "molonaviz2"
EXAMPLES = Path(__file__).resolve().parents[1] / "examples" / "twins"

# Modules et paquets de premier niveau de molonaviz2 ("__init__" : le paquet lui-même).
CORE = {"__init__", "errors", "gitinfo", "i18n", "domain", "io", "processing", "engine", "twin", "storage", "questions"}
INTERFACE = {"__main__", "app", "settings", "notice", "logs", "ui", "workers"}
# Sous-modules du cœur qui relèvent de l'interface.
INTERFACE_SUBMODULES = {"i18n/qt.py"}

# Imports interdits au cœur (préfixes de noms de modules).
FORBIDDEN = ["PySide6", *(f"molonaviz2.{name}" for name in sorted(INTERFACE - {"__main__"})), "molonaviz2.i18n.qt"]

# Délégations à l'interface, importées seulement à l'appel par des fonctions que seule l'interface
# appelle : (fichier, module importé).
INTERFACE_DELEGATES = {("i18n/__init__.py", "molonaviz2.i18n.qt")}  # t() et install_language()

# Bords du cœur autorisés à traduire avec translate() : fichier → raison.
TRANSLATING_EDGES = {"twin/cli.py": "ligne de commande des twins (choix du propriétaire, A1)"}

# Ce que l'interface appelle normalement dans le cœur.
FACADE = {
    "molonaviz2", "molonaviz2.errors", "molonaviz2.i18n",
    "molonaviz2.engine.api", "molonaviz2.twin.run", "molonaviz2.twin.results",
    "molonaviz2.twin.cli",  # point d'entrée de la ligne de commande (aiguillage de __main__)
}
# Imports hors façade relevés au début de A1 (identiques à CGS-MOLO 59aaf3e), tolérés jusqu'à la
# migration du frontend (partie C). La liste ne peut que raccourcir.
TOLERATED = {
    ("molonaviz2.app", "molonaviz2.domain.compute"),
    ("molonaviz2.app", "molonaviz2.engine.column"),
    ("molonaviz2.app", "molonaviz2.storage.project"),
    ("molonaviz2.notice", "molonaviz2.domain.compute"),
    ("molonaviz2.notice", "molonaviz2.processing.cleanup"),
    ("molonaviz2.notice", "molonaviz2.twin.config"),
    ("molonaviz2.ui.dialogs.lab_import", "molonaviz2.io.lab_import"),
    ("molonaviz2.ui.dialogs.point_import", "molonaviz2.domain.models"),
    ("molonaviz2.ui.dialogs.point_import", "molonaviz2.io.lab_import"),
    ("molonaviz2.ui.dialogs.point_import", "molonaviz2.io.point_import"),
    ("molonaviz2.ui.dialogs.simple", "molonaviz2.domain.models"),
    ("molonaviz2.ui.explorer", "molonaviz2.storage.project"),
    ("molonaviz2.ui.main_window", "molonaviz2.domain.compute"),
    ("molonaviz2.ui.main_window", "molonaviz2.domain.models"),
    ("molonaviz2.ui.main_window", "molonaviz2.io.legacy_import"),
    ("molonaviz2.ui.main_window", "molonaviz2.storage.project"),
    ("molonaviz2.ui.main_window", "molonaviz2.twin.config"),
    ("molonaviz2.ui.main_window", "molonaviz2.twin.create"),
    ("molonaviz2.ui.plots.posterior", "molonaviz2.domain.compute"),
    ("molonaviz2.ui.plots.posterior", "molonaviz2.engine.diagnostics"),
    ("molonaviz2.ui.plots.posterior", "molonaviz2.engine.results"),
    ("molonaviz2.ui.point_view.cleanup_tab", "molonaviz2.domain.models"),
    ("molonaviz2.ui.point_view.cleanup_tab", "molonaviz2.processing.cleanup"),
    ("molonaviz2.ui.point_view.context", "molonaviz2.domain.models"),
    ("molonaviz2.ui.point_view.context", "molonaviz2.engine.column"),
    ("molonaviz2.ui.point_view.context", "molonaviz2.storage.project"),
    ("molonaviz2.ui.point_view.forward_tab", "molonaviz2.domain.compute"),
    ("molonaviz2.ui.point_view.forward_tab", "molonaviz2.engine.results"),
    ("molonaviz2.ui.point_view.inversion_tab", "molonaviz2.domain.compute"),
    ("molonaviz2.ui.point_view.inversion_tab", "molonaviz2.engine.results"),
    ("molonaviz2.ui.point_view.layers", "molonaviz2.domain.compute"),
    ("molonaviz2.ui.point_view.results_tab", "molonaviz2.domain.compute"),
    ("molonaviz2.ui.point_view.results_tab", "molonaviz2.engine.results"),
    ("molonaviz2.ui.point_view.results_tab", "molonaviz2.storage.project"),
    ("molonaviz2.ui.point_view.view", "molonaviz2.domain.models"),
    ("molonaviz2.ui.point_view.view", "molonaviz2.storage.project"),
    ("molonaviz2.ui.twin_window", "molonaviz2.twin.config"),
    ("molonaviz2.workers.tasks", "molonaviz2.engine.jax.dream"),
}


def top_level(path: Path) -> str:
    return path.relative_to(SOURCE).parts[0].removesuffix(".py")


def core_files() -> list[Path]:
    return [
        path for path in sorted(SOURCE.rglob("*.py"))
        if top_level(path) in CORE and path.relative_to(SOURCE).as_posix() not in INTERFACE_SUBMODULES
    ]


def module_name(path: Path) -> str:
    parts = list(path.relative_to(SOURCE.parent).with_suffix("").parts)
    return ".".join(parts[:-1] if parts[-1] == "__init__" else parts)


def is_module(dotted: str) -> bool:
    base = SOURCE.joinpath(*dotted.split(".")[1:])
    return base.with_suffix(".py").is_file() or (base / "__init__.py").is_file()


def imported_modules(path: Path):
    """(ligne, module importé) avec résolution des imports relatifs et des sous-modules (`from ..engine import api`)."""
    name = module_name(path)
    package = name if path.name == "__init__.py" else name.rsplit(".", 1)[0]
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.ImportFrom):
            if node.level:
                base = package.split(".")[: len(package.split(".")) - node.level + 1]
                module = ".".join(base + ([node.module] if node.module else []))
            else:
                module = node.module or ""
            for alias in node.names:
                full = f"{module}.{alias.name}"
                yield node.lineno, full if is_module(full) else module
        elif isinstance(node, ast.Import):
            for alias in node.names:
                yield node.lineno, alias.name


# 1. Classement ------------------------------------------------------------------------------------


def test_every_module_is_classified():
    names = {path.name.removesuffix(".py") for path in SOURCE.iterdir() if path.suffix == ".py" or (path / "__init__.py").is_file()}
    assert CORE.isdisjoint(INTERFACE)
    unclassified = sorted(names - CORE - INTERFACE)
    assert not unclassified, f"module(s) à classer dans CORE ou INTERFACE : {unclassified}"


# 2. Import et exécution sans Qt ------------------------------------------------------------------

HOOKED = textwrap.dedent(
    """
    import importlib, pkgutil, runpy, sys

    FORBIDDEN = {forbidden!r}

    class Forbid:
        def find_spec(self, name, path=None, target=None):
            if any(name == f or name.startswith(f + ".") for f in FORBIDDEN):
                raise ImportError(f"import interdit au cœur : {{name}}")

    sys.meta_path.insert(0, Forbid())

    import molonaviz2
    for info in pkgutil.walk_packages(molonaviz2.__path__, "molonaviz2."):
        top = info.name.split(".")[1]
        if top in {core!r} and info.name != "molonaviz2.i18n.qt":
            importlib.import_module(info.name)

    from molonaviz2.twin.cli import main
    for command in ("validate", "id", "run"):
        code = main([command, {twin!r}])
        assert code == 0, (command, code)

    # Le point d'entrée (python -m molonaviz2 twin …) aiguille vers la CLI sans charger Qt.
    sys.argv = ["molonaviz2", "twin", "validate", {twin!r}]
    try:
        runpy.run_module("molonaviz2", run_name="__main__", alter_sys=True)
    except SystemExit as exit:
        assert exit.code == 0, exit.code
    print("OK", sorted(m for m in sys.modules if m.startswith("PySide6")))
    """
)


def test_core_imports_and_runs_without_qt(tmp_path):
    twin = shutil.copytree(EXAMPLES / "twin_01", tmp_path / "twin_01")
    code = HOOKED.format(forbidden=FORBIDDEN, core=sorted(CORE), twin=str(twin))
    completed = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=600, check=False)
    assert completed.returncode == 0, completed.stderr[-4000:]
    assert completed.stdout.strip().endswith("OK []"), completed.stdout


def test_core_does_not_import_the_interface():
    problems = []
    for path in core_files():
        for line, module in imported_modules(path):
            if (path.relative_to(SOURCE).as_posix(), module) in INTERFACE_DELEGATES:
                continue
            if any(module == f or module.startswith(f + ".") for f in FORBIDDEN):
                problems.append(f"{path.relative_to(SOURCE)}:{line} importe {module}")
    assert not problems, "\n".join(problems)


# 3. Aucune phrase traduite dans le cœur ----------------------------------------------------------


def test_core_does_not_translate():
    problems = []
    for path in core_files():
        relative = path.relative_to(SOURCE).as_posix()
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                name = func.id if isinstance(func, ast.Name) else func.attr if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name) and func.value.id == "i18n" else None
                if name == "t" or (name == "translate" and relative not in TRANSLATING_EDGES):
                    problems.append(f"{relative}:{node.lineno} appelle {name}(...)")
            if isinstance(node, ast.ImportFrom) and node.module and node.module.split(".")[-1] == "i18n":
                for alias in node.names:
                    if alias.name in ("t", "install_language"):
                        problems.append(f"{relative}:{node.lineno} importe {alias.name}")
    assert not problems, "\n".join(problems)


def test_n_marker_comes_from_pure_i18n():
    problems = []
    for path in core_files():
        relative = path.relative_to(SOURCE).as_posix()
        tree = ast.parse(path.read_text(encoding="utf-8"))
        uses = any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "N_" for n in ast.walk(tree))
        defines = any(isinstance(n, ast.FunctionDef) and n.name == "N_" for n in tree.body)
        if relative == "i18n/__init__.py":
            assert defines
            continue
        if defines:
            problems.append(f"{relative} redéfinit N_")
        imported = {module for _, module in imported_modules(path) if module == "molonaviz2.i18n"}
        if uses and not any(isinstance(n, ast.ImportFrom) and any(a.name == "N_" for a in n.names) for n in ast.walk(tree)):
            problems.append(f"{relative} utilise N_ sans l'importer")
        elif uses and not imported:
            problems.append(f"{relative} importe N_ d'ailleurs que molonaviz2.i18n")
    assert not problems, "\n".join(problems)


# 4. Façade -----------------------------------------------------------------------------------------


def test_interface_uses_the_facade_or_tolerated_imports():
    found = set()
    for path in sorted(SOURCE.rglob("*.py")):
        if top_level(path) not in INTERFACE:
            continue
        for _, module in imported_modules(path):
            parts = module.split(".")
            if parts[0] == "molonaviz2" and (len(parts) == 1 or parts[1] in CORE) and module not in FACADE and not module.startswith("molonaviz2.questions"):
                found.add((module_name(path), module))
    new = sorted(found - TOLERATED)
    gone = sorted(TOLERATED - found)
    assert not new, f"nouvel import hors façade (passer par engine/api, twin/run, twin/results ou questions) : {new}"
    assert not gone, f"import toléré disparu : le retirer de TOLERATED : {gone}"
