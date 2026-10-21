"""Ligne de commande des twins.

    molonaviz2 twin validate <twin_xx>     étape 2 : valide la config
    molonaviz2 twin id <twin_xx>           étape 3 : affiche le twin_id et l'état dans le registre
    molonaviz2 twin run <twin_xx>          étapes 4-6 : exécute (ou signale un doublon) et range les résultats
    molonaviz2 twin list <dossier twins>   liste les twins et leur état

Codes de sortie : 0 succès, 1 config invalide ou erreur, 2 usage incorrect.
"""

from __future__ import annotations

import argparse
import sys

from ..errors import UserError
from ..i18n import render, t


def _setup_translation() -> None:
    from PySide6 import QtCore

    from ..i18n import install_language
    from ..settings import Settings

    QtCore.QCoreApplication.setOrganizationName("MOLONARI")
    QtCore.QCoreApplication.setApplicationName("Molonaviz2")
    app = QtCore.QCoreApplication.instance() or QtCore.QCoreApplication([])
    install_language(app, Settings().language)


def state_label(status) -> str:
    if status.state == "new":
        return t("valid, not run yet")
    if status.state == "computed":
        return t("already computed")
    if status.state == "duplicate":
        return t("duplicate of {name}").format(name=status.source.name)
    return t("configuration modified after it was run (frozen)")


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="molonaviz2 twin", description=t("Digital twins of sampling points"))
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("validate", "id", "run"):
        sub.add_parser(command).add_argument("twin", help="twin_xx folder or twin.yaml")
    sub.add_parser("list").add_argument("folder", help="twins folder")
    args = parser.parse_args(argv)
    _setup_translation()

    from .config import TwinConfigError, load_twin
    from .run import list_twins, run_twin, twin_status

    try:
        if args.command == "validate":
            config = load_twin(args.twin)
            print(t("{name}: valid configuration ({mode} mode).").format(name=config.name, mode=config.mode))
        elif args.command == "id":
            status = twin_status(args.twin)
            print(status.twin_id)
            print(f"{status.config.name}: {state_label(status)}")
        elif args.command == "run":
            shown = []

            def progress(p):
                if p.phase in ("burn_in", "sampling"):
                    shown.append(True)
                    print(f"  {p.phase} {p.iteration}/{p.total}    ", end="\r", flush=True)

            run = run_twin(args.twin, on_progress=progress)
            if shown:
                print()  # termine la ligne de progression
            status = run.status
            if run.computed:
                print(t("{name} computed in {seconds:.1f} s → {folder}").format(name=status.config.name, seconds=run.duration_s, folder=status.results_dir))
            elif status.state == "computed":
                print(t("Already computed: {name} (id {id}) was run on {date}. Existing results: {folder}").format(
                    name=status.config.name, id=status.twin_id, date=status.source.created_at, folder=status.results_dir))
            else:
                print(t("Duplicate: {name} has the same configuration as {source} (id {id}). Existing results: {folder}").format(
                    name=status.config.name, source=status.source.name, id=status.twin_id, folder=status.results_dir))
            if status.config.experimental:
                print(t("Note: mcmc mode is experimental."))
        else:
            for path in list_twins(args.folder):
                try:
                    status = twin_status(path)
                    print(f"{path.name:12s} {status.config.mode:7s} {status.twin_id}  {state_label(status)}")
                except TwinConfigError as exc:
                    print(f"{path.name:12s} {t('invalid')}: {render(exc.issues[0], t)}")
    except TwinConfigError as exc:
        print(t("Invalid configuration:"), file=sys.stderr)
        for issue in exc.issues:
            print(f"  - {render(issue, t)}", file=sys.stderr)
        return 1
    except UserError as exc:
        print(render(exc, t), file=sys.stderr)
        return 1
    return 0
