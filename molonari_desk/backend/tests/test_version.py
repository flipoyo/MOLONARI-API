"""Version du backend : une seule source (backend/pyproject.toml), au format SemVer 2.0.0."""

import importlib.util
from pathlib import Path

import molonari_twin
import pytest

WORKSPACE = Path(__file__).resolve().parents[2]
ROOT = WORKSPACE / "backend"
_spec = importlib.util.spec_from_file_location("check_version", WORKSPACE / "tools" / "check_version.py")
check_version = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(check_version)


def test_version_changelog_and_tag_are_consistent():
    assert check_version.problems(root=ROOT, tag_prefix="molonari-twin-v") == []
    version = check_version.declared_version(ROOT)
    assert check_version.problems(f"molonari-twin-v{version}", ROOT, "molonari-twin-v") == []
    assert check_version.problems("molonari-twin-v9.9.9", ROOT, "molonari-twin-v") != []
    assert check_version.package_for_tag(f"molonari-twin-v{version}") == "molonari-twin-v"


def test_package_version_is_read_from_pyproject():
    assert molonari_twin.__version__ == check_version.declared_version(ROOT)


@pytest.mark.parametrize("version", ["1.0", "v1.0.0", "01.0.0", "1.0.0-dev.1", "1.0.0-alpha", "1.0.0+g1234"])
def test_invalid_versions_are_reported(tmp_path, version):
    (tmp_path / "pyproject.toml").write_text(f'[project]\nversion = "{version}"\n')
    (tmp_path / "CHANGELOG.md").write_text(f"## [{version}]\n")
    assert check_version.problems(root=tmp_path) != []


def test_engine_version_carries_the_package_version():
    """La version du code entre dans engine_version(), donc dans chaque twin_id, avec le suffixe local éventuel."""
    import re

    from molonari_twin.engine.api import engine_version, local_suffix, release_version

    version = re.escape(check_version.declared_version(ROOT))
    assert re.fullmatch(rf"molonari-twin {version}(\+g[0-9a-f]{{12}}(\.dirty)?)? · jax \S+", engine_version()), engine_version()
    assert engine_version() == release_version().replace(" · jax", f"{local_suffix()} · jax", 1)


def test_frozen_application_has_no_local_suffix(monkeypatch):
    from molonari_twin.engine import api

    monkeypatch.setattr(api.sys, "frozen", True, raising=False)
    assert api.local_suffix() == ""


def test_jax_version_is_read_without_importing_jax():
    """engine_version() identifie le code sans charger jax : l'API qui lit des résultats n'a pas à le faire.
    La chaîne est celle que donnerait jax.__version__, donc aucun twin_id ne change."""
    import subprocess
    import sys
    import textwrap

    code = textwrap.dedent(
        """
        import sys
        from molonari_twin.engine.api import engine_version, jax_version, release_version
        print(engine_version(), release_version(), jax_version(), sep="|")
        print("jax" in sys.modules)
        import jax
        print(jax.__version__)
        """
    )
    completed = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300, check=False)
    assert completed.returncode == 0, completed.stderr
    versions, loaded, imported = completed.stdout.strip().splitlines()
    engine, release, installed = versions.split("|")
    assert loaded == "False"
    assert installed == imported
    assert engine.endswith(f" · jax {imported}") and release.endswith(f" · jax {imported}")


def test_twin_status_does_not_load_jax():
    """Identifier un twin et retrouver ses résultats ne charge ni jax ni ArviZ."""
    import subprocess
    import sys

    example = ROOT.parent / "examples" / "twins" / "twin_03"
    code = (
        "import sys\n"
        "from molonari_twin.twin.run import twin_status\n"
        f"status = twin_status({str(example)!r})\n"
        "print(status.state)\n"
        "print(sorted({n.split('.')[0] for n in sys.modules} & {'jax', 'jaxlib', 'arviz'}))"
    )
    completed = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300, check=False)
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip().splitlines()[-1] == "[]"
