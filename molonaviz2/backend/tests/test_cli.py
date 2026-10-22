"""Ligne de commande des twins : langue choisie par MOLONAVIZ_LANG (fr par défaut), sans Qt."""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from molonari_twin.i18n import load_catalog

EXAMPLES = Path(__file__).resolve().parents[2] / "examples" / "twins"
FR = load_catalog("fr")


def cli(*args, lang=None):
    env = {k: v for k, v in os.environ.items() if k != "MOLONAVIZ_LANG"}
    if lang is not None:
        env["MOLONAVIZ_LANG"] = lang
    return subprocess.run([sys.executable, "-m", "molonari_twin", *map(str, args)], capture_output=True, text=True, timeout=300, env=env, check=False)


@pytest.fixture
def twin_01(tmp_path):
    return shutil.copytree(EXAMPLES / "twin_01", tmp_path / "twin_01")


@pytest.mark.parametrize(
    "lang, expected",
    [
        ("en", "twin_01: valid configuration (direct mode)."),
        (None, FR["{name}: valid configuration ({mode} mode)."].format(name="twin_01", mode="direct")),
        ("de", FR["{name}: valid configuration ({mode} mode)."].format(name="twin_01", mode="direct")),
    ],
)
def test_validate_speaks_the_requested_language(twin_01, lang, expected):
    completed = cli("validate", twin_01, lang=lang)
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
