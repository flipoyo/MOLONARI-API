"""Vérifie la version des paquets de l'espace (SemVer 2.0.0) : pyproject.toml, CHANGELOG et tag.

    python tools/check_version.py                         # version et CHANGELOG des deux paquets
    python tools/check_version.py molonari-twin-v0.6.0    # et le tag, pour le paquet de son préfixe
    python tools/check_version.py molonaviz2-v0.6.0

Utilisé par la CI (.github/workflows/molonaviz2-version.yml) et par les tests test_version.py des deux paquets.
"""

from __future__ import annotations

import re
import sys
import tomllib
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[1]
# Préfixe de tag → (paquet, dossier). Chaque paquet a sa version, son CHANGELOG et ses tags.
PACKAGES = {
    "molonari-twin-v": ("molonari-twin", WORKSPACE / "backend"),
    "molonaviz2-v": ("molonaviz2", WORKSPACE / "frontend"),
}
ROOT = WORKSPACE / "frontend"  # paquet par défaut des fonctions ci-dessous (l'application)
TAG_PREFIX = "molonaviz2-v"
# Expression régulière officielle de semver.org.
SEMVER = re.compile(
    r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)"
    r"(?:-((?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*)(?:\.(?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*))*))?"
    r"(?:\+([0-9a-zA-Z-]+(?:\.[0-9a-zA-Z-]+)*))?$"
)
# Pré-versions admises (elles ont aussi un sens en PEP 440 : a, b, rc).
PRERELEASES = ("alpha", "beta", "rc")


def declared_version(root: Path = ROOT) -> str:
    return tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))["project"]["version"]


def problems(tag: str | None = None, root: Path = ROOT, tag_prefix: str = TAG_PREFIX) -> list[str]:
    version = declared_version(root)
    match = SEMVER.match(version)
    if match is None:
        return [f"pyproject.toml: {version!r} is not a SemVer 2.0.0 version"]
    found = []
    if match.group(5) is not None:
        found.append(f"pyproject.toml: build metadata (+{match.group(5)}) is not allowed in a release")
    if match.group(4) is not None:
        label, _, number = match.group(4).partition(".")
        if label not in PRERELEASES or not number.isdigit():
            found.append(f"pyproject.toml: pre-release must be -alpha.N, -beta.N or -rc.N, not -{match.group(4)}")
    changelog = root / "CHANGELOG.md"
    if not changelog.is_file():
        found.append("CHANGELOG.md: missing")
    elif f"## [{version}]" not in changelog.read_text(encoding="utf-8"):
        found.append(f"CHANGELOG.md: no section '## [{version}]'")
    if tag and tag != tag_prefix + version:
        found.append(f"tag {tag!r} does not match pyproject.toml (expected {tag_prefix + version!r})")
    return found


def package_for_tag(tag: str) -> str | None:
    """Préfixe du paquet visé par un tag (None si le préfixe est inconnu)."""
    return next((prefix for prefix in PACKAGES if tag.startswith(prefix)), None)


def main(argv: list[str]) -> int:
    tag = argv[1] if len(argv) > 1 and argv[1] else None
    if tag and package_for_tag(tag) is None:
        print(f"error: tag {tag!r}: unknown prefix (expected one of {sorted(PACKAGES)})", file=sys.stderr)
        return 1
    failed = False
    for prefix, (name, root) in PACKAGES.items():
        if tag and package_for_tag(tag) != prefix:
            continue
        found = problems(tag, root, prefix)
        for problem in found:
            print(f"error: {name}: {problem}", file=sys.stderr)
        if not found:
            print(f"{name} {declared_version(root)} : {'version, CHANGELOG et tag' if tag else 'version et CHANGELOG'} cohérents")
        failed = failed or bool(found)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
