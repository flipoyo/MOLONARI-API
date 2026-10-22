"""Point d'entrée : la commande `molonari-twin` et `python -m molonari_twin` (ligne de commande des twins)."""

import sys


def main(argv: list[str] | None = None) -> int:
    from molonari_twin.twin.cli import main as twin_main

    return twin_main(list(sys.argv[1:] if argv is None else argv))


if __name__ == "__main__":
    sys.exit(main())
