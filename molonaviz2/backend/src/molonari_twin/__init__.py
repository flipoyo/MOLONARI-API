"""molonari-twin : backend des twins MOLONARI (données, calcul, twins, questions), sans interface."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("molonari-twin")  # source unique : pyproject.toml, telle qu'écrite (SemVer)
except PackageNotFoundError:  # sources lues sans installation
    __version__ = "0+unknown"
