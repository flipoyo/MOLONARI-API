"""T10 : `reproduce --export` et `reproduce --compare` : export d'un twin calculé et comparaison de deux exports.

Les tests de comparaison travaillent sur des dossiers fabriqués ; ceux de l'export calculent de vrais twins (slow).
"""

import json
import re
import shutil

import numpy as np
import pytest
from molonari_twin.twin import cli
from molonari_twin.twin.reproduce import compare_exports, export_twin
from molonari_twin.twin.run import run_twin, twin_status

from conftest import EXAMPLE_TWINS


def fake_export(folder, values=None, fingerprint=None):
    """Un export minimal : results/ (metadata.json, table.csv, arrays.npz) et fingerprint.json."""
    results = folder / "results"
    results.mkdir(parents=True)
    (results / "metadata.json").write_text(json.dumps({"twin_id": "x", "created_at": "t", "duration_s": 1.0}), encoding="utf-8")
    (results / "table.csv").write_text("a,b\n1,2\n", encoding="utf-8")
    np.savez(results / "arrays.npz", x=np.array([1.0, np.nan, 3.0]) if values is None else values, n=np.arange(3))
    np.savez(results / "chain_state.npz", state=np.arange(4.0), burn_in_duration_s=np.array(1.0))
    data = {"twin_id": "id1", "engine_version": "1", "jax": "0.4", "jaxlib": "0.4", "numpy": "2", "cleaned_sha256": None,
            "device": "cpu", "platform": {"machine": "arm64"}}
    data.update(fingerprint or {})
    (folder / "fingerprint.json").write_text(json.dumps(data), encoding="utf-8")
    return folder


def test_compare_same_export_is_identical(tmp_path):
    a = fake_export(tmp_path / "A")
    shutil.copytree(a, tmp_path / "B")
    comparison = compare_exports(a, tmp_path / "B")
    assert comparison.identical and all(f.match for f in comparison.files)
    assert cli.main(["reproduce", "--compare", str(a), str(tmp_path / "B")]) == 0


def test_compare_ignores_machine_fields_and_burn_in_duration(tmp_path):
    a = fake_export(tmp_path / "A")
    b = fake_export(tmp_path / "B", fingerprint={"device": "gpu", "platform": {"machine": "x86_64"}})
    np.savez(b / "results" / "chain_state.npz", state=np.arange(4.0), burn_in_duration_s=np.array(99.0))
    assert compare_exports(a, b).identical


def test_compare_detects_one_changed_value_in_npz(tmp_path, capsys):
    a = fake_export(tmp_path / "A")
    b = fake_export(tmp_path / "B", values=np.array([1.0, np.nan, 3.0 + 2.0**-40]))
    comparison = compare_exports(a, b)
    assert not comparison.identical
    bad = [f for f in comparison.files if not f.match]
    assert [f.path for f in bad] == ["arrays.npz"]
    assert bad[0].arrays[0]["name"] == "x" and bad[0].arrays[0]["different"] == 1 and bad[0].arrays[0]["max_abs_diff"] > 0
    assert cli.main(["reproduce", "--compare", str(a), str(b)]) == 1
    assert "arrays.npz" in capsys.readouterr().out


def test_compare_detects_missing_file(tmp_path):
    a = fake_export(tmp_path / "A")
    b = fake_export(tmp_path / "B")
    (b / "results" / "table.csv").unlink()
    comparison = compare_exports(a, b)
    assert not comparison.identical
    assert [(f.path, f.detail) for f in comparison.files if not f.match] == [("table.csv", "absent de B")]


def test_compare_flags_different_recipe(tmp_path):
    a = fake_export(tmp_path / "A")
    b = fake_export(tmp_path / "B", fingerprint={"jax": "0.5"})
    comparison = compare_exports(a, b)
    assert not comparison.identical and comparison.recipe_differences == ["jax"]


def test_compare_cli_json_and_exit_code(tmp_path, capsys):
    a = fake_export(tmp_path / "A")
    b = fake_export(tmp_path / "B", values=np.array([9.0, np.nan, 3.0]))
    assert cli.main(["reproduce", "--compare", str(a), str(b), "--json"]) == 1
    out = json.loads(capsys.readouterr().out)
    assert out["identical"] is False and out["files"][0]["path"] in {f["path"] for f in out["files"]}
    shutil.copytree(a, tmp_path / "C")
    assert cli.main(["reproduce", "--compare", str(a), str(tmp_path / "C"), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["identical"] is True


def test_reproduce_option_errors(tmp_path):
    for argv in (["reproduce"], ["reproduce", "--json"], ["reproduce", "--compare", "a", "b", "--output", "o"],
                 ["reproduce", "t", "--export", "e", "--compare", "a", "b"]):
        with pytest.raises(SystemExit) as stop:
            cli.main(argv)
        assert stop.value.code == 2


@pytest.mark.slow
def test_export_writes_results_fingerprint_and_values(computed_twins, tmp_path):
    for name in ("twin_01", "twin_03"):
        folder = export_twin(computed_twins / name, tmp_path / name)
        results = twin_status(computed_twins / name).results_dir
        assert sorted(p.name for p in (folder / "results").iterdir()) == sorted(p.name for p in results.iterdir() if p.is_file())
        fingerprint = json.loads((folder / "fingerprint.json").read_text(encoding="utf-8"))
        for key in ("files", "comparable_metadata_sha256", "cleaned_sha256", "twin_id", "engine_version", "device", "platform",
                    "jax", "jaxlib", "numpy", "cgs_state_sha256"):
            assert key in fingerprint
        assert set(fingerprint["files"]) == {p.name for p in (folder / "results").iterdir()}
        assert set(fingerprint["platform"]) == {"system", "release", "machine", "processor"}
        lines = {line.split(" = ")[0]: line.split(" = ")[1] for line in (folder / "resultats.txt").read_text(encoding="utf-8").splitlines()
                 if not line.startswith("#")}
        assert set(lines) == {"water_flux_mean", "heat_total", "heat_advective", "heat_conductive", "rmse_t1", "rmse_t2", "rmse_t3", "rmse_total"}
        for key in ("water_flux_mean", "rmse_total"):
            shown, hexa = lines[key].split(" | ")
            assert float.fromhex(hexa) == float(shown)
        assert compare_exports(folder, folder).identical
    assert cli.main(["reproduce", str(computed_twins / "twin_03"), "--export", str(tmp_path / "cli")]) == 0


@pytest.mark.slow
def test_export_has_no_seed(tmp_path):
    """Une graine distinctive (731942) n'apparaît dans aucun fichier exporté, ni la clé « seed »."""
    twin = tmp_path / "twin_seed"
    shutil.copytree(EXAMPLE_TWINS / "twin_03", twin)
    yaml = twin / "twin.yaml"
    text = yaml.read_text(encoding="utf-8")
    for old, new in (("nb_iter: 300", "nb_iter: 40"), ("burn_in_max: 300", "burn_in_max: 20"), ("n_field_samples: 40", "n_field_samples: 16"),
                     ("seed: 1\n", "seed: 731942\n")):
        assert old in text
        text = text.replace(old, new)
    yaml.write_text(text, encoding="utf-8")
    assert "731942" in yaml.read_text(encoding="utf-8")
    run_twin(twin)
    folder = export_twin(twin, tmp_path / "export")
    checked = 0
    for path in sorted(folder.rglob("*")):
        if not path.is_file():
            continue
        checked += 1
        if path.suffix == ".npz":
            with np.load(path, allow_pickle=False) as archive:
                assert "seed" not in archive.keys()
                assert all("731942" not in key for key in archive.keys())
                for key in archive.keys():
                    assert not (archive[key].dtype.kind in "iu" and 731942 in archive[key])
        else:
            content = path.read_text(encoding="utf-8")
            # la graine comme nombre entier isolé : les suites de chiffres d'un flottant ne comptent pas
            assert not re.search(r"(?<![\w.])731942(?![\w.])", content), path
            assert '"seed"' not in content and "seed:" not in content, path
    assert checked > 5
