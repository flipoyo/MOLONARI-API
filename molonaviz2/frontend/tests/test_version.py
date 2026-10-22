"""Version de l'application : une seule source (frontend/pyproject.toml), au format SemVer 2.0.0."""

import importlib.util
from pathlib import Path

import molonaviz2
from molonari_twin.facade import backend_version

WORKSPACE = Path(__file__).resolve().parents[2]
ROOT = WORKSPACE / "frontend"
_spec = importlib.util.spec_from_file_location("check_version", WORKSPACE / "tools" / "check_version.py")
check_version = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(check_version)


def test_version_changelog_and_tag_are_consistent():
    assert check_version.problems(root=ROOT, tag_prefix="molonaviz2-v") == []
    version = check_version.declared_version(ROOT)
    assert check_version.problems(f"molonaviz2-v{version}", ROOT, "molonaviz2-v") == []
    assert check_version.problems("molonaviz2-v9.9.9", ROOT, "molonaviz2-v") != []
    assert check_version.package_for_tag(f"molonaviz2-v{version}") == "molonaviz2-v"
    assert check_version.package_for_tag("autre-v1.0.0") is None


def test_package_version_is_read_from_pyproject():
    assert molonaviz2.__version__ == check_version.declared_version(ROOT)


def test_backend_version_is_the_one_of_its_own_package():
    assert backend_version == check_version.declared_version(WORKSPACE / "backend")
