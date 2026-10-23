"""Ligne de commande des twins : langue choisie par MOLONARI_LANG (ou son ancien nom MOLONAVIZ_LANG ; fr par
défaut), sans Qt."""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from molonari_twin.i18n import load_catalog

EXAMPLES = Path(__file__).resolve().parents[2] / "examples" / "twins"
FR = load_catalog("fr")


def cli(*args, lang=None, old=None):
    """Lance la CLI avec MOLONARI_LANG=lang et MOLONAVIZ_LANG=old (None : variable absente)."""
    env = {k: v for k, v in os.environ.items() if k not in ("MOLONARI_LANG", "MOLONAVIZ_LANG")}
    if lang is not None:
        env["MOLONARI_LANG"] = lang
    if old is not None:
        env["MOLONAVIZ_LANG"] = old
    return subprocess.run([sys.executable, "-m", "molonari_twin", *map(str, args)], capture_output=True, text=True, timeout=300, env=env, check=False)


@pytest.fixture
def twin_01(tmp_path):
    return shutil.copytree(EXAMPLES / "twin_01", tmp_path / "twin_01")


ENGLISH = "twin_01: valid configuration (direct mode)."
FRENCH = FR["{name}: valid configuration ({mode} mode)."].format(name="twin_01", mode="direct")


@pytest.mark.parametrize(
    "lang, old, expected",
    [
        ("en", None, ENGLISH),
        (None, "en", ENGLISH),  # ancien nom de la variable, encore lu
        ("fr", "en", FRENCH),  # le nouveau nom l'emporte
        (None, None, FRENCH),
        ("de", None, FRENCH),
    ],
)
def test_validate_speaks_the_requested_language(twin_01, lang, old, expected):
    completed = cli("validate", twin_01, lang=lang, old=old)
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == expected


def test_invalid_configuration_in_french(twin_01):
    path = twin_01 / "twin.yaml"
    path.write_text(path.read_text(encoding="utf-8").replace("offset: 0", "ofset: 0"), encoding="utf-8")
    completed = cli("validate", twin_01, lang="fr")
    assert completed.returncode == 1
    lines = completed.stderr.splitlines()
    assert lines[0] == FR["Invalid configuration:"]
    unknown = FR["unknown key (allowed: {keys})."].split("{keys}")[0]
    assert any(line.startswith(f"  - sensors.ofset: {unknown}") for line in lines), completed.stderr


def test_missing_twin_in_english(tmp_path):
    completed = cli("validate", tmp_path / "absent", lang="en")
    assert completed.returncode == 1
    assert completed.stderr.splitlines() == ["Invalid configuration:", f"  - File not found: {tmp_path / 'absent'}"]


def test_usage_error_exit_code():
    assert cli("frobnicate").returncode == 2


# serve -------------------------------------------------------------------------------------------


def test_serve_starts_a_local_api_with_a_local_runner(monkeypatch, tmp_path, capsys):
    uvicorn = pytest.importorskip("uvicorn")
    from fastapi import FastAPI
    from molonari_twin.tasks.local import LocalTaskRunner
    from molonari_twin.twin.cli import main

    calls, closed = [], []
    monkeypatch.setattr(uvicorn, "run", lambda app, **options: calls.append((app, options)))
    monkeypatch.setattr(LocalTaskRunner, "close", lambda self: closed.append(self))
    monkeypatch.setenv("MOLONARI_LANG", "en")
    db = tmp_path / "tasks.sqlite"
    assert main(["serve", "--local", "--twins-folder", str(EXAMPLES), "--tasks-db", str(db), "--port", "9001"]) == 0
    [(app, options)] = calls
    assert isinstance(app, FastAPI)
    assert options == {"host": "127.0.0.1", "port": 9001}
    assert len(closed) == 1 and closed[0].store.path == db and closed[0].max_workers == 1
    assert capsys.readouterr().out.strip() == (
        f"Local server: http://127.0.0.1:9001 (twins: {EXAMPLES.resolve()}, tasks: {db})")


@pytest.mark.parametrize("host", ["0.0.0.0", "192.168.1.10"])
def test_serve_refuses_a_non_local_address(host, capsys, monkeypatch):
    from molonari_twin.twin.cli import main

    monkeypatch.setenv("MOLONARI_LANG", "en")
    with pytest.raises(SystemExit) as info:
        main(["serve", "--host", host])
    assert info.value.code == 2
    assert "only a local address is allowed until authentication exists" in capsys.readouterr().err


def test_serve_explains_the_missing_extra(monkeypatch, capsys):
    from molonari_twin.twin.cli import main

    monkeypatch.setitem(sys.modules, "uvicorn", None)  # import uvicorn → ImportError
    monkeypatch.setenv("MOLONARI_LANG", "fr")
    assert main(["serve", "--local"]) == 1
    assert capsys.readouterr().err.strip() == FR["The server needs optional dependencies: install molonari-twin[server]."]


def test_other_commands_do_not_import_the_server(twin_01):
    code = (
        "import sys\n"
        "from molonari_twin.twin.cli import main\n"
        f"main(['validate', {str(twin_01)!r}])\n"
        "print(sorted({n.split('.')[0] for n in sys.modules} & {'fastapi', 'uvicorn', 'starlette'}))"
    )
    completed = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300, check=False)
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip().splitlines()[-1] == "[]"
