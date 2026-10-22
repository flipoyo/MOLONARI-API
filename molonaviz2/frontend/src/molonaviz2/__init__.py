"""Molonaviz 2 : application de bureau pour l'inversion bayésienne des mesures MOLONARI."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("molonaviz2")  # source unique : pyproject.toml, telle qu'écrite (SemVer)
except PackageNotFoundError:  # sources lues sans installation
    __version__ = "0+unknown"
