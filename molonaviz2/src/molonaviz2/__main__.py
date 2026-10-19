"""Permet `python -m molonaviz2` et sert de point d'entrée à l'exécutable PyInstaller."""

import sys

from molonaviz2.app import main

if __name__ == "__main__":
    sys.exit(main())
