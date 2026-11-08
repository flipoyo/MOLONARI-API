"""T10 : `reproduce --export` et `reproduce --compare` : export d'un twin calculé et comparaison de deux exports.

Les tests de comparaison travaillent sur des dossiers fabriqués ; ceux de l'export calculent de vrais twins (slow).
"""

import json
import re
import shutil
from pathlib import Path

import numpy as np
import pytest
from conftest import EXAMPLE_TWINS
from molonari_twin.twin import cli
from molonari_twin.twin.reproduce import compare_exports, export_twin
from molonari_twin.twin.run import run_twin, twin_status


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


NUMBER = re.compile(r"(?<![\w.])[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?(?![\w.])")
SEED_KEY = re.compile(r'"seed"|\bseed\s*[:=]')


def _text_forms(content, seed):
    """Les formes de la graine dans un texte : un nombre isolé de même valeur (731942, 731942.0, 7.31942e5...), ou la clé « seed »."""
    forms = [f"nombre {token}" for token in NUMBER.findall(content) if float(token) == float(seed)]
    if SEED_KEY.search(content):
        forms.append("clé seed")
    return forms


def seed_occurrences(folder: Path, seed: int) -> list[str]:
    """Une description « chemin relatif: forme » par occurrence de la graine (ou de la clé « seed ») dans un export ; vide s'il n'y en a pas."""
    found = []
    for path in sorted(folder.rglob("*")):
        if not path.is_file():
            continue
        name = path.relative_to(folder).as_posix()
        if path.suffix != ".npz":
            found += [f"{name}: {form}" for form in _text_forms(path.read_text(encoding="utf-8"), seed)]
            continue
        with np.load(path, allow_pickle=False) as archive:
            for key in archive.files:
                if "seed" in key or str(seed) in key:
                    found.append(f"{name}: tableau nommé {key}")
                array = archive[key]
                if array.dtype.kind in "iuf" and np.any(array == seed):
                    found.append(f"{name}[{key}]: valeur numérique")
                elif array.dtype.kind in "US":
                    found += [f"{name}[{key}]: {form}" for item in array.ravel() for form in _text_forms(str(item), seed)]
    return found


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


def _write(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _metadata_with(folder, extra):
    meta = folder / "results" / "metadata.json"
    _write(meta, json.dumps({**json.loads(meta.read_text(encoding="utf-8")), **extra}))


INJECTIONS = {
    "entier dans csv": lambda f: _write(f / "results" / "table.csv", "a,b\n731942,2\n"),
    "flottant 731942.0 dans metadata": lambda f: _write(
        f / "results" / "metadata.json", '{"twin_id": "x", "x": 731942.0}'),
    "notation 7.31942e5": lambda f: _write(f / "results" / "table.csv", "a,b\n7.31942e5,2\n"),
    "notation 7.319420E+05": lambda f: _write(f / "results" / "table.csv", "a,b\n7.319420E+05,2\n"),
    "npz flottant": lambda f: np.savez(f / "results" / "extra.npz", v=np.array([1.0, 731942.0])),
    "npz entier": lambda f: np.savez(f / "results" / "extra.npz", v=np.array([731942])),
    "npz chaînes": lambda f: np.savez(f / "results" / "extra.npz", v=np.array(["a", "731942"])),
    "npz nommé seed": lambda f: np.savez(f / "results" / "extra.npz", seed=np.array(3)),
    "clé json": lambda f: _metadata_with(f, {"seed": 3}),
    "clé yaml": lambda f: _write(f / "results" / "recipe.yaml", "seed: 3\n"),
    "clé yaml en ligne": lambda f: _write(f / "results" / "recipe.yaml", "{seed: 3}\n"),
    "clé avec égal": lambda f: _write(f / "results" / "recipe.yaml", "seed=3\n"),
}


def test_seed_detector_finds_every_form(tmp_path):
    assert seed_occurrences(fake_export(tmp_path / "clean"), 731942) == []
    for i, (case, inject) in enumerate(INJECTIONS.items()):
        folder = fake_export(tmp_path / f"case{i}")
        inject(folder)
        assert seed_occurrences(folder, 731942), f"graine non vue : {case}"
    for i, neighbour in enumerate(("0.731942", "7319420", "1731942", "731942.5", "a731942", "731942b")):
        folder = fake_export(tmp_path / f"neg{i}")
        _write(folder / "results" / "table.csv", f"a,b\n{neighbour},2\n")
        assert seed_occurrences(folder, 731942) == [], f"faux positif : {neighbour}"


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
    assert sum(1 for p in folder.rglob("*") if p.is_file()) > 5
    assert seed_occurrences(folder, 731942) == []
