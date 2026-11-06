"""Point d'entrée : `python -m molonari_desk`, le script `molonari-desk` et l'exécutable PyInstaller.

`molonari-desk twin …` part vers la ligne de commande des twins sans importer Qt ; le reste ouvre
l'application de bureau (molonari_desk.app).
"""

import sys


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv if argv is None else argv)
    if len(argv) > 1 and argv[1] == "twin":
        from molonari_twin.facade import twin_cli

        return twin_cli(argv[2:])
    from molonari_desk.app import main as app_main

    return app_main(argv)


if __name__ == "__main__":
    sys.exit(main())
