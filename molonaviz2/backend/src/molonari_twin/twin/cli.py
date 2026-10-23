"""Ligne de commande des twins.

    molonari-twin validate <twin_xx>     étape 2 : valide la config
    molonari-twin id <twin_xx>           étape 3 : affiche le twin_id et l'état dans le registre
    molonari-twin run <twin_xx>          étapes 4-6 : exécute (ou signale un doublon) et range les résultats
    molonari-twin list <dossier twins>   liste les twins et leur état
    molonari-twin serve --local          API HTTP sur 127.0.0.1 (extra `server`), calculs dans des workers

(aussi `python -m molonari_twin …`, et `molonaviz2 twin …` depuis l'application)

Codes de sortie : 0 succès, 1 config invalide ou erreur, 2 usage incorrect.

Langue des messages : variable d'environnement MOLONARI_LANG (fr, en), sinon son ancien nom MOLONAVIZ_LANG,
sinon fr. Ce module est
un bord du cœur : il traduit sans Qt, avec i18n.translate (exception nommée de tests/test_architecture.py).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from ..errors import UserError
from ..i18n import DEFAULT_LANGUAGE, LANGUAGES, render, translate

LOCAL_HOSTS = ("127.0.0.1", "localhost", "::1")


def language() -> str:
    """Langue de la ligne de commande : MOLONARI_LANG, sinon son ancien nom MOLONAVIZ_LANG, si la langue est
    proposée ; sinon la langue par défaut."""
    value = os.environ.get("MOLONARI_LANG", os.environ.get("MOLONAVIZ_LANG", DEFAULT_LANGUAGE))
    return value if value in LANGUAGES else DEFAULT_LANGUAGE


def tr(text: str) -> str:
    """Traduit un texte anglais dans la langue de la ligne de commande."""
    return translate(text, language())


def state_label(status) -> str:
    if status.state == "new":
        return tr("valid, not run yet")
    if status.state == "computed":
        return tr("already computed")
    if status.state == "duplicate":
        return tr("duplicate of {name}").format(name=status.source.name)
    return tr("configuration modified after it was run (frozen)")


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="molonari-twin", description=tr("Digital twins of sampling points"))
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("validate", "id", "run"):
        sub.add_parser(command).add_argument("twin", help="twin_xx folder or twin.yaml")
    sub.choices["run"].add_argument("--strict", action="store_true",
                                    help="refuse an unpublished code version or modified or untagged repositories")
    sub.add_parser("list").add_argument("folder", help="twins folder")
    serve_parser = sub.add_parser("serve")
    serve_parser.add_argument("--local", action="store_true", help="listen on 127.0.0.1 (default, and the only mode until authentication exists)")
    serve_parser.add_argument("--host", default="127.0.0.1", help=argparse.SUPPRESS)
    serve_parser.add_argument("--port", type=int, default=8765)
    serve_parser.add_argument("--twins-folder", default=".", help="folder that contains the twin_xx folders")
    serve_parser.add_argument("--tasks-db", default=None, help="task database (default: $XDG_CACHE_HOME/molonari-twin/tasks.sqlite)")
    serve_parser.add_argument("--max-workers", type=int, default=1, help="tasks computed at the same time")
    args = parser.parse_args(argv)
    if args.command == "serve" and args.host not in LOCAL_HOSTS:
        serve_parser.error(tr("only a local address is allowed until authentication exists"))

    from .config import TwinConfigError, load_twin
    from .run import list_twins, run_twin, twin_status

    try:
        if args.command == "serve":
            return serve(args)
        if args.command == "validate":
            config = load_twin(args.twin)
            print(tr("{name}: valid configuration ({mode} mode).").format(name=config.name, mode=config.mode))
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

            run = run_twin(args.twin, on_progress=progress, strict=args.strict)
            if shown:
                print()  # termine la ligne de progression
            for warning in run.warnings:
                print(tr("Warning:"), render(warning, tr), file=sys.stderr)
            status = run.status
            if run.computed:
                print(tr("{name} computed in {seconds:.1f} s → {folder}").format(name=status.config.name, seconds=run.duration_s, folder=status.results_dir))
            elif status.state == "computed":
                print(tr("Already computed: {name} (id {id}) was run on {date}. Existing results: {folder}").format(
                    name=status.config.name, id=status.twin_id, date=status.source.created_at, folder=status.results_dir))
            else:
                print(tr("Duplicate: {name} has the same configuration as {source} (id {id}). Existing results: {folder}").format(
                    name=status.config.name, source=status.source.name, id=status.twin_id, folder=status.results_dir))
            if status.config.experimental:
                print(tr("Note: mcmc mode is experimental."))
        else:
            for path in list_twins(args.folder):
                try:
                    status = twin_status(path)
                    print(f"{path.name:12s} {status.config.mode:7s} {status.twin_id}  {state_label(status)}")
                except TwinConfigError as exc:
                    print(f"{path.name:12s} {tr('invalid')}: {render(exc.issues[0], tr)}")
    except TwinConfigError as exc:
        print(tr("Invalid configuration:"), file=sys.stderr)
        for issue in exc.issues:
            print(f"  - {render(issue, tr)}", file=sys.stderr)
        return 1
    except UserError as exc:
        print(render(exc, tr), file=sys.stderr)
        return 1
    return 0


def serve(args) -> int:
    """Lance l'API sur une adresse locale, avec un LocalTaskRunner ; les workers sont arrêtés à la sortie."""
    try:
        import uvicorn

        from ..server.app import create_app
    except ImportError:
        raise UserError("E_SERVER_EXTRA_MISSING") from None
    from ..tasks.local import LocalTaskRunner, default_db_path
    from ..tasks.runner import task_key

    db = Path(args.tasks_db) if args.tasks_db else default_db_path()
    folder = Path(args.twins_folder).resolve()
    runner = LocalTaskRunner(db, max_workers=args.max_workers)
    print(tr("Local server: http://{host}:{port} (twins: {folder}, tasks: {db})").format(
        host=args.host, port=args.port, folder=folder, db=db))
    try:
        # L'appareil de calcul est lu par un worker (qui importe JAX) : l'API, elle, ne charge pas le moteur.
        device = runner.submit("diag.device", {}, key=task_key("diag.device", {}))
        uvicorn.run(create_app(runner, folder, device_task=device.id), host=args.host, port=args.port)
    finally:
        runner.close()
    return 0
