"""État git d'un chemin et suffixe local de version, sur des dépôts temporaires (git init dans tmp_path)."""

import pytest
from conftest import commit_file
from conftest import git_run as run

from molonaviz2 import gitinfo
from molonaviz2.gitinfo import clean_url, repository_state, version_suffix


@pytest.fixture
def repo(tmp_path):
    """Dépôt avec un projet `molonaviz2/` et un dossier `twins/`, tagué molonaviz2-v1.0.0."""
    repo = tmp_path / "repo"
    repo.mkdir()
    run(repo, "init", "-q")
    commit_file(repo, "molonaviz2/code.py", "x = 1\n")
    commit_file(repo, "twins/twin_01/twin.yaml", "name: twin_01\n")
    run(repo, "tag", "molonaviz2-v1.0.0")
    return repo


def test_clean_url_removes_credentials():
    assert clean_url("https://user:ghp_secret@github.com/org/repo.git") == "https://github.com/org/repo.git"
    assert clean_url("https://ghp_secret@github.com/org/repo.git") == "https://github.com/org/repo.git"
    assert clean_url("ssh://git@host:22/org/repo.git") == "ssh://host:22/org/repo.git"
    assert clean_url("git@github.com:org/repo.git") == "git@github.com:org/repo.git"  # forme scp : pas de mot de passe possible
    assert clean_url("https://github.com/org/repo.git") == "https://github.com/org/repo.git"
    assert clean_url(None) is None


def test_tagged_clean_repository_on_a_branch(repo):
    run(repo, "remote", "add", "origin", "https://user:ghp_secret@github.com/org/repo.git")
    state = repository_state(repo / "twins" / "twin_01")
    assert state.path == "twins/twin_01" and state.commit == run(repo, "rev-parse", "HEAD")
    assert state.tags == ["molonaviz2-v1.0.0"] and state.branch == "main" and not state.detached_head
    assert state.clean is True and state.remote == "https://github.com/org/repo.git"
    assert "ghp_secret" not in str(state.to_dict())


def test_detached_head_without_remote(repo):
    run(repo, "checkout", "-q", "--detach", "HEAD")
    state = repository_state(repo)
    assert state.branch is None and state.detached_head and state.remote is None and state.path == "."


def test_changes_are_limited_to_the_path(repo):
    (repo / "molonaviz2" / "code.py").write_text("x = 2\n")
    assert repository_state(repo / "molonaviz2").clean is False
    assert repository_state(repo / "twins").clean is True
    (repo / "molonaviz2" / "code.py").write_text("x = 1\n")
    (repo / "twins" / "nouveau.txt").write_text("non suivi\n")
    assert repository_state(repo / "twins").clean is False  # un fichier non suivi compte


def test_outside_a_repository_and_without_git(tmp_path, monkeypatch):
    assert repository_state(tmp_path) is None
    assert version_suffix(tmp_path, "molonaviz2-v1.0.0") == ""
    monkeypatch.setenv("PATH", "")
    with pytest.raises(gitinfo.GitUnavailable):
        repository_state(tmp_path)
    assert version_suffix(tmp_path, "molonaviz2-v1.0.0") == ""


def test_version_suffix(repo):
    project = repo / "molonaviz2"
    tag = "molonaviz2-v1.0.0"
    assert version_suffix(project, tag) == ""  # tag exact et propre
    commit_file(repo, "README.md", "hors du projet\n")
    assert version_suffix(project, tag) == ""  # commit après le tag, hors de molonaviz2/
    commit_file(repo, "molonaviz2/code.py", "x = 2\n")
    short = run(repo, "rev-parse", "--short=12", "HEAD")
    assert version_suffix(project, tag) == f"+g{short}" and len(short) == 12
    (project / "code.py").write_text("x = 3\n")
    assert version_suffix(project, tag) == f"+g{short}.dirty"
    assert version_suffix(project, "molonaviz2-v9.9.9") == f"+g{short}.dirty"  # tag absent


def test_untagged_version_has_a_local_suffix(repo):
    short = run(repo, "rev-parse", "--short=12", "HEAD")
    assert version_suffix(repo / "molonaviz2", "molonaviz2-v2.0.0") == f"+g{short}"
