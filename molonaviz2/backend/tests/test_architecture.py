"""Architecture du backend : molonari_twin s'importe et s'exécute sans Qt, sans bibliothèque graphique et
sans le frontend, ne produit aucune phrase traduite, et n'offre au frontend qu'une porte, sa façade.

Ce fichier ne dépend pas de PySide6 : le job backend de la CI le lance sans Qt installé.
"""

import ast
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[1] / "src" / "molonari_twin"
EXAMPLES = Path(__file__).resolve().parents[2] / "examples" / "twins"

# Modules et paquets de premier niveau de molonari_twin ("__init__" : le paquet lui-même).
MODULES = {"__init__", "__main__", "errors", "gitinfo", "i18n", "domain", "io", "processing", "engine", "twin", "storage", "questions", "facade"}

# Imports interdits au backend (préfixes de noms de modules) : Qt, bibliothèques graphiques, frontend.
FORBIDDEN = ["PySide6", "pyqtgraph", "matplotlib", "molonaviz2"]

# Bords du backend autorisés à traduire avec translate() : fichier → raison.
TRANSLATING_EDGES = {"twin/cli.py": "ligne de commande des twins (choix du propriétaire, A1)"}


def files() -> list[Path]:
    return sorted(SOURCE.rglob("*.py"))


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
    unclassified = sorted(names - MODULES)
    assert not unclassified, f"module(s) à classer dans MODULES : {unclassified}"


# 2. Import et exécution sans Qt ------------------------------------------------------------------

HOOKED = textwrap.dedent(
    """
    import importlib, pkgutil, runpy, sys

    FORBIDDEN = {forbidden!r}

    # Paquets interdits rendus absents, comme s'ils n'étaient pas installés : leur import échoue, et
    # importlib.util.find_spec (par lequel ArviZ sonde matplotlib) répond None.
    for name in FORBIDDEN:
        sys.modules[name] = None

    import molonari_twin
    for info in pkgutil.walk_packages(molonari_twin.__path__, "molonari_twin."):
        importlib.import_module(info.name)

    from molonari_twin.twin.cli import main
    for command in ("validate", "id", "run"):
        code = main([command, {twin!r}])
        assert code == 0, (command, code)

    # Le point d'entrée (python -m molonari_twin …) lance la ligne de commande.
    sys.argv = ["molonari-twin", "validate", {twin!r}]
    try:
        runpy.run_module("molonari_twin", run_name="__main__", alter_sys=True)
    except SystemExit as exit:
        assert exit.code == 0, exit.code
    print("OK", sorted(m for m, module in sys.modules.items() if m.split(".")[0] in FORBIDDEN and module is not None))
    """
)


def test_backend_imports_and_runs_without_qt(tmp_path):
    twin = shutil.copytree(EXAMPLES / "twin_01", tmp_path / "twin_01")
    code = HOOKED.format(forbidden=FORBIDDEN, twin=str(twin))
    completed = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=600, check=False)
    assert completed.returncode == 0, completed.stderr[-4000:]
    assert completed.stdout.strip().endswith("OK []"), completed.stdout


def test_backend_does_not_import_qt_graphics_or_the_frontend():
    problems = []
    for path in files():
        for line, module in imported_modules(path):
            if any(module == f or module.startswith(f + ".") for f in FORBIDDEN):
                problems.append(f"{path.relative_to(SOURCE)}:{line} importe {module}")
    assert not problems, "\n".join(problems)


# 3. Aucune phrase traduite dans le backend --------------------------------------------------------


def test_backend_does_not_translate():
    problems = []
    for path in files():
        relative = path.relative_to(SOURCE).as_posix()
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                name = func.id if isinstance(func, ast.Name) else func.attr if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name) and func.value.id == "i18n" else None
                if name == "t" or (name == "translate" and relative not in TRANSLATING_EDGES):
                    problems.append(f"{relative}:{node.lineno} appelle {name}(...)")
            if isinstance(node, ast.FunctionDef) and node.name in ("t", "install_language"):
                problems.append(f"{relative}:{node.lineno} définit {node.name}() (traduction Qt : frontend)")
    assert not problems, "\n".join(problems)


def test_n_marker_comes_from_pure_i18n():
    problems = []
    for path in files():
        relative = path.relative_to(SOURCE).as_posix()
        tree = ast.parse(path.read_text(encoding="utf-8"))
        uses = any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "N_" for n in ast.walk(tree))
        defines = any(isinstance(n, ast.FunctionDef) and n.name == "N_" for n in tree.body)
        if relative == "i18n/__init__.py":
            assert defines
            continue
        if defines:
            problems.append(f"{relative} redéfinit N_")
        imported = {module for _, module in imported_modules(path) if module == "molonari_twin.i18n"}
        if uses and not any(isinstance(n, ast.ImportFrom) and any(a.name == "N_" for a in n.names) for n in ast.walk(tree)):
            problems.append(f"{relative} utilise N_ sans l'importer")
        elif uses and not imported:
            problems.append(f"{relative} importe N_ d'ailleurs que molonari_twin.i18n")
    assert not problems, "\n".join(problems)


# 4. Façade -----------------------------------------------------------------------------------------


def test_backend_does_not_import_the_facade():
    problems = []
    for path in files():
        relative = path.relative_to(SOURCE).as_posix()
        if relative == "facade.py":
            continue
        for line, module in imported_modules(path):
            if module == "molonari_twin.facade" or module.startswith("molonari_twin.facade."):
                problems.append(f"{relative}:{line} importe la façade")
    assert not problems, "\n".join(problems)


def test_facade_import_is_light():
    code = (
        "import sys, molonari_twin.facade\n"
        "print(sorted({n.split('.')[0] for n in sys.modules if n.split('.')[0] in ('PySide6', 'jax', 'arviz')}))"
    )
    completed = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300, check=False)
    assert completed.returncode == 0, completed.stderr[-4000:]
    assert completed.stdout.strip() == "[]", f"modules lourds chargés par la façade : {completed.stdout.strip()}"


def test_facade_exports_exist_and_are_the_original_objects():
    import importlib

    from molonari_twin import facade
    from molonari_twin.engine.jax import dream

    assert not [name for name in facade.__all__ if not hasattr(facade, name)]
    for name in facade.__all__:
        value = getattr(facade, name)
        origin = getattr(value, "__module__", None)
        reexported = origin and origin.startswith("molonari_twin.") and origin != "molonari_twin.facade"
        if (isinstance(value, type) or callable(value)) and reexported:
            assert getattr(importlib.import_module(origin), value.__name__) is value, name
    assert facade.Cancelled is dream.Cancelled
