"""Version : une seule source (pyproject.toml), au format SemVer 2.0.0."""

import importlib.util
from pathlib import Path

import pytest

import molonaviz2

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("check_version", ROOT / "tools" / "check_version.py")
check_version = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(check_version)


def test_version_changelog_and_tag_are_consistent():
    assert check_version.problems() == []
    version = check_version.declared_version()
    assert check_version.problems(f"molonaviz2-v{version}") == []
    assert check_version.problems("molonaviz2-v9.9.9") != []


def test_package_version_is_read_from_pyproject():
    assert molonaviz2.__version__ == check_version.declared_version()


@pytest.mark.parametrize("version", ["1.0", "v1.0.0", "01.0.0", "1.0.0-dev.1", "1.0.0-alpha", "1.0.0+g1234"])
def test_invalid_versions_are_reported(tmp_path, version):
    (tmp_path / "pyproject.toml").write_text(f'[project]\nversion = "{version}"\n')
    (tmp_path / "CHANGELOG.md").write_text(f"## [{version}]\n")
    assert check_version.problems(root=tmp_path) != []


def test_engine_version_carries_the_package_version():
    """La version du code entre dans engine_version(), donc dans chaque twin_id, avec le suffixe local éventuel."""
    import re

    from molonaviz2.engine.api import engine_version, local_suffix, release_version

    version = re.escape(check_version.declared_version())
    assert re.fullmatch(rf"molonaviz2 {version}(\+g[0-9a-f]{{12}}(\.dirty)?)? · jax \S+", engine_version()), engine_version()
    assert engine_version() == release_version().replace(" · jax", f"{local_suffix()} · jax", 1)


def test_frozen_application_has_no_local_suffix(monkeypatch):
    from molonaviz2.engine import api

    monkeypatch.setattr(api.sys, "frozen", True, raising=False)
    assert api.local_suffix() == ""
