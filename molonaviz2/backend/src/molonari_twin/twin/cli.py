"""Ligne de commande des twins.

    molonari-twin validate <twin_xx>     étape 2 : valide la config
    molonari-twin id <twin_xx>           étape 3 : affiche le twin_id et l'état dans le registre
    molonari-twin run <twin_xx>          étapes 4-6 : exécute (ou signale un doublon) et range les résultats
    molonari-twin list <dossier twins>   liste les twins et leur état
    molonari-twin report <twin_xx>       écrit le rapport public (« Public Report/ ») d'un twin calculé et de son space
    molonari-twin questions [<twin_xx>]  liste les questions du catalogue (celles d'un twin, s'il est donné)
    molonari-twin ask <twin_xx> <question>   pose une question à un twin calculé (--params '{"z": 0.2}', JSON du corps du POST)
    molonari-twin lab build <lab> --point <point>   construit processed_data/<point>/ (mesures nettoyées et manifeste) d'après lab.yaml
    molonari-twin serve --local          API HTTP sur 127.0.0.1 (extra `server`), calculs dans des workers

`questions` et `ask` sont une couche mince sur TwinAPI, sans FastAPI : avec `--json`, ils écrivent le JSON de /v1
(`GET /v1/questions`, `POST /v1/twins/{ref}/questions/{question}`) ; une erreur écrit le JSON d'erreur de /v1 sur la
sortie standard (code de sortie 1). `--level {user,admin}` (user par défaut) : le user n'interroge que les échanges
d'eau, la chaleur et la température, et reçoit des réponses réduites à une liste blanche ; l'admin, tout le catalogue.
Ce niveau oriente l'usage, ce n'est pas une sécurité (api/access.py). `serve --local --level` le donne à l'API HTTP :
au niveau user, lancer un twin (`POST …/run`) et suivre les tâches sont refusés (403).

(aussi `python -m molonari_twin …`, et `molonaviz2 twin …` depuis l'application)

Codes de sortie : 0 succès, 1 config invalide ou erreur, 2 usage incorrect.

Langue des messages : variable d'environnement MOLONARI_LANG (fr, en), sinon son ancien nom MOLONAVIZ_LANG,
sinon fr. Ce module est
un bord du cœur : il traduit sans Qt, avec i18n.translate (exception nommée de tests/test_architecture.py).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from ..errors import Issue, UserError
from ..i18n import DEFAULT_LANGUAGE, LANGUAGES, render, translate

LOCAL_HOSTS = ("127.0.0.1", "localhost", "::1")
LEVELS = ("user", "admin")  # api.access.LEVELS, sans importer l'API pour les seuls choix de l'analyseur (tests/test_questions_cli.py les compare)
SHORT_ARRAY = 6  # un tableau plus long est résumé (taille, min, max)


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
    report_parser = sub.add_parser("report")
    report_parser.add_argument("twin", help="twin_xx folder or twin.yaml (the twin that carries the report key)")
    report_parser.add_argument("--output", default=None, help="folder to write (default: 'Public Report' in the twin folder)")
    add_question_commands(sub)
    lab_parser = sub.add_parser("lab", help="virtual lab (lab.yaml)")
    lab_build = lab_parser.add_subparsers(dest="lab_command", required=True).add_parser(
        "build", help="clean a point of the lab and write processed_data/<point>/ (measurements.csv and manifest.json)")
    lab_build.add_argument("lab", help="folder of the lab (the one that holds lab.yaml) or lab.yaml")
    lab_build.add_argument("--point", required=True, help="point to build: a key of 'points' in lab.yaml")
    serve_parser = sub.add_parser("serve")
    serve_parser.add_argument("--local", action="store_true", help="listen on 127.0.0.1 (default, and the only mode until authentication exists)")
    serve_parser.add_argument("--host", default="127.0.0.1", help=argparse.SUPPRESS)
    serve_parser.add_argument("--port", type=int, default=8765)
    serve_parser.add_argument("--twins-folder", default=".", help="folder that contains the twin_xx folders")
    serve_parser.add_argument("--tasks-db", default=None, help="task database (default: $XDG_CACHE_HOME/molonari-twin/tasks.sqlite)")
    serve_parser.add_argument("--max-workers", type=int, default=1, help="tasks computed at the same time")
    serve_parser.add_argument("--level", choices=LEVELS, default="user", help="access level (default: user: water, heat and temperature only)")
    args = parser.parse_args(argv)
    if args.command == "serve" and args.host not in LOCAL_HOSTS:
        serve_parser.error(tr("only a local address is allowed until authentication exists"))

    from .config import TwinConfigError, load_twin
    from .lab import LabConfigError
    from .run import list_twins, run_twin, twin_status

    try:
        if args.command == "serve":
            return serve(args)
        if args.command in ("questions", "ask"):
            return question_command(args)
        if args.command == "validate":
            config = load_twin(args.twin)
            print(tr("{name}: valid configuration ({mode} mode).").format(name=config.name, mode=config.mode))
        elif args.command == "id":
            status = twin_status(args.twin)
            print(status.twin_id)
            print(f"{status.config.name}: {state_label(status)}")
        elif args.command == "report":
            from ..report import generate_report

            result = generate_report(args.twin, args.output)
            for skipped, reason in zip(result.skipped, result.reasons, strict=True):
                print(tr("Skipped: {question} ({model}): {reason}").format(
                    question=skipped["question"], model=skipped["model"], reason=render(reason, tr)))
            print(tr("Public report written: {folder} ({entries} entries, {figures} figures, {skipped} skipped).").format(
                folder=result.folder, entries=result.entries, figures=result.figures, skipped=len(result.skipped)))
        elif args.command == "lab":
            from .lab import build_point

            built = build_point(args.lab, args.point)
            for warning in built.warnings:
                print(tr("Warning:"), render(warning, tr), file=sys.stderr)
            manifest = built.manifest
            print(tr("Processed data written: {folder} ({rows} rows, from {start} to {end}, fingerprint {digest}).").format(
                folder=built.folder, rows=manifest["rows"], start=manifest["start"], end=manifest["end"], digest=manifest["cleaned_sha256"]))
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
    except (TwinConfigError, LabConfigError) as exc:
        print(tr("Invalid laboratory file:") if isinstance(exc, LabConfigError) else tr("Invalid configuration:"), file=sys.stderr)
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
    print(tr("Access level: {level}").format(level=args.level), file=sys.stderr)
    try:
        # L'appareil de calcul est lu par un worker (qui importe JAX) : l'API, elle, ne charge pas le moteur.
        device = runner.submit("diag.device", {}, key=task_key("diag.device", {}))
        uvicorn.run(create_app(runner, folder, device_task=device.id, level=args.level), host=args.host, port=args.port)
    finally:
        runner.close()
    return 0


# Questions ---------------------------------------------------------------------------------------


def add_question_commands(sub) -> None:
    """Les commandes `questions` et `ask` de l'analyseur."""
    questions_parser = sub.add_parser("questions")
    questions_parser.add_argument("twin", nargs="?", default=None, help="twin_xx folder or twin.yaml (only the questions of its mode)")
    ask_parser = sub.add_parser("ask")
    ask_parser.add_argument("twin", help="twin_xx folder or twin.yaml")
    ask_parser.add_argument("question", help="question identifier (water_flux) or reference (HN-10)")
    ask_parser.add_argument("--params", default=None, help="parameters of the question, as the JSON body of the POST request")
    for parser in (questions_parser, ask_parser):
        parser.add_argument("--level", choices=LEVELS, default="user", help="access level (default: user)")
        parser.add_argument("--json", action="store_true", help="write the JSON of the /v1 API (errors too, on the standard output)")


def question_command(args) -> int:
    """`questions` et `ask`. Avec --json, une erreur attendue est écrite comme celle de /v1, sur la sortie standard."""
    try:
        return ask(args) if args.command == "ask" else list_questions(args)
    except UserError as error:
        if not args.json:
            raise
        from ..tasks.model import error_dict

        print(json.dumps(error_dict(error), ensure_ascii=False))
        return 1


def leveled_api(args, twin: str | None):
    """(API au niveau demandé, référence du twin) : le dossier parent du twin est le dossier de twins."""
    from ..api import LeveledAPI, TwinAPI

    if twin is None:
        return LeveledAPI(TwinAPI(Path.cwd()), args.level), None
    path = Path(twin).resolve()
    folder = path.parent if path.is_file() else path
    return LeveledAPI(TwinAPI(folder.parent), args.level), folder.name


def list_questions(args) -> int:
    api, ref = leveled_api(args, args.twin)
    questions = api.questions(ref)
    if args.json:
        from ..contract.catalog import questions_out

        print(json.dumps(questions_out(questions).model_dump(mode="json"), ensure_ascii=False))
        return 0
    print(tr("Questions of the {level} level:").format(level=args.level))
    for question in questions:
        print(f"  {question.id:20s} {question.ref:6s} {' '.join(question.modes):4s} {tr(question.label)}")
    return 0


def read_params(text: str | None) -> dict | None:
    """Le corps du POST, donné à --params : un objet JSON (E_REQUEST_INVALID, comme l'API, sinon)."""
    if text is None:
        return None
    try:
        params = json.loads(text)
    except json.JSONDecodeError as error:
        raise UserError("E_REQUEST_INVALID", issues=[Issue("params", "E_PARAM_INVALID", prefixed=True, message=f"JSON: {error}")]) from None
    if not isinstance(params, dict):
        raise UserError("E_REQUEST_INVALID", issues=[Issue("params", "E_PARAM_INVALID", prefixed=True, message="Input should be a JSON object")])
    return params


def ask(args) -> int:
    params = read_params(args.params)
    api, ref = leveled_api(args, args.twin)
    answer = api.ask(ref, args.question, params)
    if args.json:
        from ..contract.codec import answer_to_json

        print(json.dumps(answer_to_json(answer), ensure_ascii=False))
        return 0
    question = api.question(args.question)
    print(f"{tr(question.label)} ({question.id}, {question.ref})")
    for name, quantity in answer.data.items():
        print(f"  {name}: {format_quantity(quantity)}")
    parameters = (answer.provenance.get("parameters") or {}).items()
    if parameters:
        print(tr("Parameters:"), ", ".join(f"{name} = {value}" for name, value in parameters))
    if answer.warnings:
        print(tr("Warnings:"))
        for warning in answer.warnings:
            print(f"  - {render(warning, tr)}")
    return 0


def format_quantity(quantity) -> str:
    """« valeur unité », avec l'intervalle s'il y en a un ; un long tableau est résumé."""
    text = format_value(quantity.value)
    if quantity.interval is not None:
        text += f" ({tr('interval')} {format_value(quantity.interval[0])} – {format_value(quantity.interval[1])})"
    return text if quantity.unit in ("1", "datetime64[s]") else f"{text} {quantity.unit}"


def format_value(value) -> str:
    """Un scalaire tel quel (6 chiffres significatifs), un court tableau en liste, un long tableau en résumé :
    nombre de valeurs, minimum et maximum (les valeurs absentes, NaN et NaT, ne comptent pas)."""
    import numpy as np

    array = np.asarray(value)
    if array.ndim == 0:
        return show_scalar(array[()])
    flat = array.ravel()
    if flat.size <= SHORT_ARRAY:
        return "[" + ", ".join(show_scalar(x) for x in flat) + "]"
    present = flat[~np.isnat(flat)] if flat.dtype.kind == "M" else flat[np.isfinite(flat)] if flat.dtype.kind == "f" else flat
    if flat.dtype.kind in "fiuM" and present.size:
        return tr("{count} values, from {low} to {high}").format(count=flat.size, low=show_scalar(present.min()), high=show_scalar(present.max()))
    return tr("{count} values").format(count=flat.size)


def show_scalar(x) -> str:
    import numpy as np

    if isinstance(x, np.floating | float):
        return f"{x:.6g}"
    if isinstance(x, np.datetime64):
        return "—" if np.isnat(x) else str(x.astype("datetime64[s]"))
    return str(x)
