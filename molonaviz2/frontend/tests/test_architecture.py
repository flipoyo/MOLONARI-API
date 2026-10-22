"""Architecture du frontend : molonaviz2 n'appelle le backend que par sa façade (molonari_twin.facade),
ne calcule rien lui-même, et `molonaviz2 twin …` reste utilisable sans Qt."""

import ast
import shutil
import subprocess
import sys
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[1] / "src" / "molonaviz2"
EXAMPLES = Path(__file__).resolve().parents[2] / "examples" / "twins"

# Modules et paquets de premier niveau de molonaviz2 ("__init__" : le paquet lui-même).
MODULES = {"__init__", "__main__", "app", "settings", "notice", "logs", "i18n", "ui", "workers"}

# Seul module du backend que le frontend importe.
FACADE = "molonari_twin.facade"
# Bibliothèques de calcul : le frontend ne calcule rien, il passe par la façade.
COMPUTING = ("jax", "jaxlib", "arviz", "scipy")


def imported_modules(path: Path):
    """(ligne, module importé, absolu) ; `from molonari_twin import facade` donne molonari_twin.facade."""
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.ImportFrom) and not node.level and node.module:
            if node.module == "molonari_twin":
                for alias in node.names:
                    yield node.lineno, f"molonari_twin.{alias.name}"
            else:
                yield node.lineno, node.module
        elif isinstance(node, ast.Import):
            for alias in node.names:
                yield node.lineno, alias.name


def test_every_module_is_classified():
    names = {path.name.removesuffix(".py") for path in SOURCE.iterdir() if path.suffix == ".py" or (path / "__init__.py").is_file()}
    unclassified = sorted(names - MODULES)
    assert not unclassified, f"module(s) à classer dans MODULES : {unclassified}"


def test_frontend_uses_only_the_facade_of_the_backend():
    found = []
    for path in sorted(SOURCE.rglob("*.py")):
        for line, module in imported_modules(path):
            if module.split(".")[0] == "molonari_twin" and module != FACADE:
                found.append(f"{path.relative_to(SOURCE)}:{line} importe {module}")
    assert not found, "import du backend hors façade (passer par molonari_twin.facade) :\n" + "\n".join(found)


def test_frontend_does_not_compute():
    found = []
    for path in sorted(SOURCE.rglob("*.py")):
        for line, module in imported_modules(path):
            if module.split(".")[0] in COMPUTING:
                found.append(f"{path.relative_to(SOURCE)}:{line} importe {module}")
    assert not found, "bibliothèque de calcul importée par le frontend :\n" + "\n".join(found)


def test_twin_alias_does_not_load_qt(tmp_path):
    twin = shutil.copytree(EXAMPLES / "twin_01", tmp_path / "twin_01")
    code = (
        "import runpy, sys\n"
        f"sys.argv = ['molonaviz2', 'twin', 'validate', {str(twin)!r}]\n"
        "try:\n"
        "    runpy.run_module('molonaviz2', run_name='__main__', alter_sys=True)\n"
        "except SystemExit as exit:\n"
        "    assert exit.code == 0, exit.code\n"
        "print('OK', sorted(m for m in sys.modules if m.startswith('PySide6')))\n"
    )
    completed = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300, check=False)
    assert completed.returncode == 0, completed.stderr[-4000:]
    assert completed.stdout.strip().endswith("OK []"), completed.stdout

