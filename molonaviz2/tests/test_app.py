"""Traductions, point d'entrée, auto-test, tâches de fond."""

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest
from PySide6 import QtCore

from molonaviz2.i18n import LANGUAGES, load_catalog

TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))
from extract_strings import extract  # noqa: E402

PLACEHOLDER = re.compile(r"\{[^{}]*\}")


def test_every_interface_string_has_a_french_translation():
    catalog = load_catalog("fr")
    missing = sorted(set(extract()) - set(catalog))
    assert missing == [], f"{len(missing)} chaîne(s) sans traduction : {missing[:10]}"


def test_translations_keep_placeholders_and_markup():
    for source, translation in load_catalog("fr").items():
        assert sorted(PLACEHOLDER.findall(source)) == sorted(PLACEHOLDER.findall(translation)), source
        assert source.count("<b>") == translation.count("<b>"), source
        assert translation.strip(), source


def test_no_unused_translations():
    unused = sorted(set(load_catalog("fr")) - set(extract()))
    assert unused == [], unused


def test_languages_offered():
    assert set(LANGUAGES) == {"fr", "en"}
    assert load_catalog("en") == {}


def test_translator_is_used_by_qt(qapp):
    from molonaviz2.i18n import install_language, t

    install_language(qapp, "fr")
    assert t("Cancel") == "Annuler"
    assert t("A string that is not in the catalog") == "A string that is not in the catalog"
    install_language(qapp, "en")
    assert t("Cancel") == "Cancel"


def test_selftest_command_line():
    completed = subprocess.run([sys.executable, "-m", "molonaviz2.app", "--selftest"], capture_output=True, text=True, timeout=300)
    assert completed.returncode == 0, completed.stderr
    assert "self-test: OK" in completed.stdout


def test_task_manager_runs_one_computation_at_a_time(qtbot):
    import threading

    from molonaviz2.workers.tasks import Task, TaskManager

    manager = TaskManager()
    release = threading.Event()
    results = []
    task = Task("long", lambda ctx: release.wait(5) and 42)
    task.signals.succeeded.connect(results.append)
    manager.start(task)
    with pytest.raises(RuntimeError):
        manager.start(Task("second", lambda ctx: 1))
    release.set()
    qtbot.waitUntil(lambda: not manager.busy, timeout=5000)
    assert results == [42]


def test_task_cancellation_and_failure_signals(qtbot):
    from molonaviz2.workers.tasks import Task, TaskManager

    manager = TaskManager()
    seen = []

    def cooperative(ctx):
        while not ctx.cancelled():
            QtCore.QThread.msleep(5)
        return "ignored"

    task = Task("cancel me", cooperative)
    task.signals.cancelled.connect(lambda: seen.append("cancelled"))
    task.signals.succeeded.connect(lambda r: seen.append("succeeded"))
    manager.start(task)
    manager.cancel()
    qtbot.waitUntil(lambda: not manager.busy, timeout=5000)
    assert seen == ["cancelled"]

    failing = Task("fail", lambda ctx: 1 / 0)
    failing.signals.failed.connect(lambda exc: seen.append(type(exc).__name__))
    manager.start(failing)
    qtbot.waitUntil(lambda: not manager.busy, timeout=5000)
    assert seen[-1] == "ZeroDivisionError"


def test_catalog_file_is_valid_json():
    path = Path(__file__).resolve().parents[1] / "src" / "molonaviz2" / "i18n" / "fr.json"
    assert isinstance(json.loads(path.read_text(encoding="utf-8")), dict)
