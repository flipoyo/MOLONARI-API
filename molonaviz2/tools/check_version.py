"""Vérifie la version de Molonaviz 2 (SemVer 2.0.0) : pyproject.toml, CHANGELOG et tag.

    python tools/check_version.py                      # version et CHANGELOG
    python tools/check_version.py molonaviz2-v0.1.0    # et le tag correspondant

Utilisé par la CI (.github/workflows/molonaviz2-version.yml) et par tests/test_version.py.
"""

from __future__ import annotations

import re
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
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


def problems(tag: str | None = None, root: Path = ROOT) -> list[str]:
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
    if f"## [{version}]" not in (root / "CHANGELOG.md").read_text(encoding="utf-8"):
        found.append(f"CHANGELOG.md: no section '## [{version}]'")
    if tag and tag != TAG_PREFIX + version:
        found.append(f"tag {tag!r} does not match pyproject.toml (expected {TAG_PREFIX + version!r})")
    return found


def main(argv: list[str]) -> int:
    tag = argv[1] if len(argv) > 1 and argv[1] else None
    found = problems(tag)
    for problem in found:
        print(f"error: {problem}", file=sys.stderr)
    if not found:
        print(f"molonaviz2 {declared_version()} : {'version, CHANGELOG et tag' if tag else 'version et CHANGELOG'} cohérents")
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
