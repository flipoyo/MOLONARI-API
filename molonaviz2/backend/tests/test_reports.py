"""T5 : `reports list`, `show`, `chains` (niveau user, lecture seule, sans JAX).

Un point fait à la main : `public_reports/` avec `chains.npz` et un `report.json` qui cite le SHA-256 du `posterior.npz`.
"""

import hashlib
import json
import os
import subprocess
import sys

import numpy as np
import pytest
from molonari_twin.twin.cli import main as cli_main

PARAMETERS = ("neg_log10_k", "porosity", "lambda_s", "rhos_cs")
POSTERIOR_SHA = "ab" * 32
SEED_TEXT = "1234567"


@pytest.fixture(autouse=True)
def english(monkeypatch):
    monkeypatch.setenv("MOLONARI_LANG", "en")


def make_point(root, name="Point999", index=True, chains=True, cited_sha=None):
    reports = root / name / "public_reports"
    reports.mkdir(parents=True)
    content = b""
    if chains:
        rng = np.random.default_rng(0)
        arrays = {"iteration": np.array([5, 10, 15]), "chain": np.arange(4), "energy": rng.uniform(1, 2, (3, 4)),
                  "acceptance": np.full(4, 0.3), "resets": np.zeros(4, np.int64)}
        for param in PARAMETERS:
            arrays[f"samples.{param}"] = rng.uniform(0, 1, (3, 4, 2))
        np.savez(reports / "chains.npz", **arrays)
        content = (reports / "chains.npz").read_bytes()
    if index:
        document = {
            "report_format": "1.0.0", "point": name, "generated_at": "2026-11-01T10:00:00", "generator_version": "molonari-twin 0.9.0",
            "space": {"forward": {"twin": "twin_direct", "recipe_sha256": "00", "commit": "abc1234"},
                      "inversion": {"twin": "twin_mcmc", "recipe_sha256": "11", "commit": "def5678"}},
            "platform": {"python": "3.12.1"},
            "entries": [{"file": "inversion-water_flux.json", "question": "water_flux", "model": "inversion", "uncertainty": "posterior",
                         "parameters": {}, "sha256": "cd" * 32}],
            "skipped": [{"question": "fit_quality", "model": "forward", "code": "E_X", "params": {}}],
            "figures": [{"file": "figures/inversion-water_flux.svg", "from_entry": "inversion-water_flux.json", "sha256": "ef" * 32}],
            "chains": {"file": "chains.npz", "publication_thin": 5, "posterior_sha256": POSTERIOR_SHA,
                       "sha256": cited_sha or hashlib.sha256(content).hexdigest()},
        }
        (reports / "report.json").write_text(json.dumps(document), encoding="utf-8")
    return root / name


def run_json(capsys, *argv):
    status = cli_main([*argv, "--json"])
    return status, json.loads(capsys.readouterr().out)


def test_list_finds_the_points_that_have_public_reports(tmp_path, capsys):
    make_point(tmp_path, "Point034")
    make_point(tmp_path, "Point035", index=False)
    (tmp_path / "virtual_lab").mkdir()
    (tmp_path / "Point036" / "twins").mkdir(parents=True)  # un point sans public_reports/ n'est pas listé
    status, document = run_json(capsys, "reports", "list", str(tmp_path))
    assert status == 0
    assert document == {"points": [{"point": "Point034", "report": True, "chains": True},
                                   {"point": "Point035", "report": False, "chains": True}]}
    assert tmp_path.name not in json.dumps(document)  # des noms de points, jamais de chemin machine


def test_list_defaults_to_the_current_folder_and_accepts_a_point(tmp_path, capsys, monkeypatch):
    make_point(tmp_path)
    monkeypatch.chdir(tmp_path)
    assert cli_main(["reports", "list"]) == 0
    assert "Point999" in capsys.readouterr().out
    assert run_json(capsys, "reports", "list", str(tmp_path / "Point999"))[1]["points"][0]["point"] == "Point999"


def test_show_prints_the_report_without_paths_or_seed(tmp_path, capsys):
    point = make_point(tmp_path)
    assert cli_main(["reports", "show", str(point)]) == 0
    text = capsys.readouterr().out
    assert "Point999" in text and "inversion-water_flux.json" in text and "twin_mcmc" in text
    assert "fit_quality" in text and "figures/inversion-water_flux.svg" in text
    assert str(tmp_path) not in text and "seed" not in text.lower()
    status, document = run_json(capsys, "reports", "show", str(point))
    assert status == 0 and document["entries"][0]["question"] == "water_flux" and document["space"]["forward"]["commit"] == "abc1234"
    assert str(tmp_path) not in json.dumps(document) and "recipe_sha256" not in json.dumps(document)


def test_chains_summarizes_the_published_file(tmp_path, capsys):
    point = make_point(tmp_path)
    assert cli_main(["reports", "chains", str(point)]) == 0
    text = capsys.readouterr().out
    assert "4 chains, 3 samples per chain" in text and "publication_thin: 5" in text and POSTERIOR_SHA in text
    status, document = run_json(capsys, "reports", "chains", str(point))
    assert status == 0
    assert (document["chains"], document["samples_per_chain"], document["layers"]) == (4, 3, 2)
    assert (document["first_iteration"], document["last_iteration"]) == (5, 15)
    assert document["publication_thin"] == 5 and document["posterior_sha256"] == POSTERIOR_SHA
    assert document["sha256"] == hashlib.sha256((point / "public_reports" / "chains.npz").read_bytes()).hexdigest()
    assert document["sha256_matches"] is True and str(tmp_path) not in json.dumps(document)


def test_chains_reports_a_file_that_differs_from_the_index(tmp_path, capsys):
    point = make_point(tmp_path, cited_sha="00" * 32)
    assert run_json(capsys, "reports", "chains", str(point))[1]["sha256_matches"] is False
    assert cli_main(["reports", "chains", str(point)]) == 0
    assert "differs from the report" in capsys.readouterr().out


def test_chains_without_index_still_summarizes_the_file(tmp_path, capsys):
    point = make_point(tmp_path, index=False)
    status, document = run_json(capsys, "reports", "chains", str(point))
    assert status == 0 and document["chains"] == 4
    assert document["publication_thin"] is None and document["posterior_sha256"] is None and document["sha256_matches"] is None


@pytest.mark.parametrize(("command", "kwargs", "code"), [
    ("show", {"index": False}, "E_REPORTS_NO_REPORT"),
    ("chains", {"chains": False}, "E_REPORTS_NO_CHAINS"),
])
def test_a_point_without_report_or_chains_is_a_structured_error(tmp_path, capsys, command, kwargs, code):
    point = make_point(tmp_path, **kwargs)
    status, error = run_json(capsys, "reports", command, str(point))
    assert status == 1 and error["code"] == code and error["params"]["point"] == "Point999"
    assert str(tmp_path) not in json.dumps(error)
    assert cli_main(["reports", command, str(point)]) == 1  # sans --json : le message sur la sortie d'erreur
    assert "Point999" in capsys.readouterr().err


def test_a_point_without_public_reports_or_a_missing_folder(tmp_path, capsys):
    (tmp_path / "Point000").mkdir()
    assert run_json(capsys, "reports", "show", str(tmp_path / "Point000"))[1]["code"] == "E_REPORTS_NO_REPORT"
    assert run_json(capsys, "reports", "chains", str(tmp_path / "Point000"))[1]["code"] == "E_REPORTS_NO_CHAINS"
    assert run_json(capsys, "reports", "list", str(tmp_path / "absent"))[1]["code"] == "E_FOLDER_NOT_FOUND"


def test_unreadable_files_are_structured_errors(tmp_path, capsys):
    point = make_point(tmp_path)
    (point / "public_reports" / "report.json").write_text("{pas du json", encoding="utf-8")
    (point / "public_reports" / "chains.npz").write_bytes(b"not an npz")
    assert run_json(capsys, "reports", "show", str(point))[1]["code"] == "E_REPORTS_INDEX_INVALID"
    assert run_json(capsys, "reports", "chains", str(point))[1]["code"] == "E_REPORTS_CHAINS_INVALID"


def test_reports_do_not_load_jax(tmp_path):
    point = make_point(tmp_path)
    code = ("import sys\nfrom molonari_twin.twin.cli import main\n"
            f"statuses = [main(['reports', 'list', {str(tmp_path)!r}]), main(['reports', 'show', {str(point)!r}, '--json']),"
            f" main(['reports', 'chains', {str(point)!r}, '--json'])]\n"
            "assert statuses == [0, 0, 0], statuses\n"
            "assert 'jax' not in sys.modules, sorted(m for m in sys.modules if m.startswith('jax'))")
    completed = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False, timeout=120,
                               env={**os.environ, "MOLONARI_LANG": "en"})
    assert completed.returncode == 0, completed.stderr
