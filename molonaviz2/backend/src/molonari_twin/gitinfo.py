"""État git d'un chemin : seule partie de molonari_twin qui appelle git.

Sert à la provenance des twins (twin/provenance.py) et au suffixe local de la version du code
(engine/api.py::engine_version). Sans git, hors dépôt ou en cas d'échec, les fonctions renvoient
None ou des valeurs vides : rien ne lève d'exception, l'appelant décide quoi signaler.
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

GIT_TIMEOUT_S = 10
SHORT_COMMIT = 12  # longueur du commit dans le suffixe +g<commit>


class GitUnavailable(Exception):
    """git n'est pas installé (ou pas dans le PATH)."""


def git(args: list[str], cwd: str | Path) -> str | None:
    """Sortie de `git <args>` dans `cwd` (sans le saut de ligne final), ou None si la commande échoue.

    Lève GitUnavailable si git est introuvable."""
    env = {**os.environ, "GIT_OPTIONAL_LOCKS": "0", "LC_ALL": "C"}  # git status n'écrit pas l'index
    try:
        completed = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, timeout=GIT_TIMEOUT_S, env=env, check=False)
    except FileNotFoundError:
        raise GitUnavailable from None
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout.rstrip("\n") if completed.returncode == 0 else None


def clean_url(url: str | None) -> str | None:
    """URL sans identifiants : `https://user:token@host/x` devient `https://host/x`.

    Une URL de forme scp (`git@host:org/repo`) est gardée : elle ne peut pas contenir de mot de passe."""
    if not url or "://" not in url:
        return url
    parts = urlsplit(url)
    if "@" not in parts.netloc:
        return url
    return urlunsplit(parts._replace(netloc=parts.netloc.rsplit("@", 1)[1]))


@dataclass(frozen=True)
class RepositoryState:
    path: str  # chemin concerné, relatif à la racine du dépôt
    remote: str | None  # URL sans identifiants
    commit: str | None
    tags: list[str]  # tags exacts du commit
    branch: str | None  # None si HEAD est détaché
    detached_head: bool
    clean: bool | None  # aucune modification (ni fichier non suivi) sous `path`

    def to_dict(self) -> dict:
        return asdict(self)


def repository_root(path: str | Path) -> Path | None:
    """Racine du dépôt git qui contient `path`, ou None (hors dépôt). Lève GitUnavailable sans git."""
    path = Path(path).resolve()
    folder = path if path.is_dir() else path.parent
    top = git(["rev-parse", "--show-toplevel"], folder)
    return Path(top).resolve() if top else None


def repository_state(path: str | Path) -> RepositoryState | None:
    """État du dépôt qui contient `path`, limité à ce chemin ; None hors dépôt. Lève GitUnavailable sans git."""
    root = repository_root(path)
    if root is None:
        return None
    relative = Path(path).resolve().relative_to(root).as_posix()
    remotes = (git(["remote"], root) or "").split()
    remote = "origin" if "origin" in remotes else (remotes[0] if remotes else None)
    url = git(["remote", "get-url", remote], root) if remote else None
    commit = git(["rev-parse", "HEAD"], root)
    tags = sorted((git(["tag", "--points-at", "HEAD"], root) or "").split()) if commit else []
    branch = git(["symbolic-ref", "--short", "-q", "HEAD"], root)
    status = git(["status", "--porcelain", "--", relative], root)
    return RepositoryState(
        path=relative,
        remote=clean_url(url),
        commit=commit,
        tags=tags,
        branch=branch or None,
        detached_head=not branch,
        clean=None if status is None else status == "",
    )


def version_suffix(project_dir: str | Path, tag: str) -> str:
    """Suffixe local de la version du projet : `+g<commit>` si `tag` est absent ou si le projet en
    diffère, puis `.dirty` si le projet a des modifications non commitées. Vide sans git ou hors dépôt."""
    try:
        root = repository_root(project_dir)
        if root is None:
            return ""
        relative = Path(project_dir).resolve().relative_to(root).as_posix()
        commit = git(["rev-parse", f"--short={SHORT_COMMIT}", "HEAD"], root)
        if commit is None:
            return ""
        tagged = git(["rev-parse", "-q", "--verify", f"refs/tags/{tag}"], root) is not None
        same_as_tag = tagged and git(["diff", "--quiet", tag, "HEAD", "--", relative], root) is not None
        dirty = (git(["status", "--porcelain", "--", relative], root) or "") != ""
    except GitUnavailable:
        return ""
    if same_as_tag and not dirty:
        return ""
    return f"+g{commit}" + (".dirty" if dirty else "")
