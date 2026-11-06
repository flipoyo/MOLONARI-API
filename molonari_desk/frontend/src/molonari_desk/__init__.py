"""MOLONARI Desk : application de bureau pour l'inversion bayésienne des mesures MOLONARI."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("molonari-desk")  # source unique : pyproject.toml, telle qu'écrite (SemVer)
except PackageNotFoundError:  # sources lues sans installation
    __version__ = "0+unknown"
