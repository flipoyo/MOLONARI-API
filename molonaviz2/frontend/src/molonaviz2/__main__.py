"""Point d'entrée : `python -m molonaviz2`, le script `molonaviz2` et l'exécutable PyInstaller.

`molonaviz2 twin …` part vers la ligne de commande des twins sans importer Qt ; le reste ouvre
l'application de bureau (molonaviz2.app).
"""

import sys


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv if argv is None else argv)
    if len(argv) > 1 and argv[1] == "twin":
        from molonaviz2.facade import twin_cli

        return twin_cli(argv[2:])
    from molonaviz2.app import main as app_main

    return app_main(argv)


if __name__ == "__main__":
    sys.exit(main())
