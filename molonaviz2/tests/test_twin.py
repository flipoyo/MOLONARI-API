"""Twins : validation (C1), identifiant (C2), registre et doublons (C3), exécution et résultats (C4)."""

import json
import shutil
import subprocess
import sys

import pytest
from conftest import FIXTURES

from molonaviz2.domain.compute import ForwardConfig, InversionConfig, LayerSpec
from molonaviz2.errors import UserError
from molonaviz2.io.lab_import import import_lab_directory
from molonaviz2.io.point_import import read_pressure_file, read_temperature_file
from molonaviz2.processing.cleanup import CleanupSettings, clean
from molonaviz2.twin.config import TwinConfigError, load_twin
from molonaviz2.twin.create import create_twin
from molonaviz2.twin.registry import read_registry
from molonaviz2.twin.results import load_results
from molonaviz2.twin.run import list_twins, next_twin_folder, run_twin, twin_status

DEPTHS = (0.1, 0.2, 0.3, 0.4)


@pytest.fixture(scope="module")
def point034():
    lab = import_lab_directory(FIXTURES / "lab").lab
    sensor = next(p for p in lab.pressure_sensors if p.name == "P508")
    raw_t = read_temperature_file(FIXTURES / "point034" / "point034_T_measures.csv").frame
    raw_p = read_pressure_file(FIXTURES / "point034" / "point034_P_measures.csv").frame
    return clean(raw_t, raw_p, sensor, CleanupSettings()).cleaned.iloc[:400]


@pytest.fixture
def twins(tmp_path):
    folder = tmp_path / "demo" / "twins"
    folder.mkdir(parents=True)
    return folder


def make_twin(twins, measurements, config=None, sigma_t=0.05):
    config = config or ForwardConfig([LayerSpec("couche 1", 0.4)], nb_cells=40)
    return create_twin(twins, measurements, "Point034", "test", DEPTHS, 0.0, sigma_t, 0.01, config)


def edit(folder, old, new):
    path = folder / "twin.yaml"
    text = path.read_text()
    assert old in text, old
    path.write_text(text.replace(old, new))


# C1 : validation ---------------------------------------------------------------------------


def test_created_twin_is_valid_and_human_readable(twins, point034):
    folder = make_twin(twins, point034)
    assert folder.name == "twin_01"
    text = (folder / "twin.yaml").read_text()
    assert "schema: molonaviz-twin/1" in text and "depths: [0.1, 0.2, 0.3, 0.4]" in text
    config = load_twin(folder)
    assert (config.name, config.mode, config.point) == ("twin_01", "direct", "Point034")
    assert config.layers[0].neg_log10_k == 12 and config.column_depth == 0.4


def test_validation_lists_every_problem_with_its_key(twins):
    folder = twins / "twin_09"
    folder.mkdir()
    (folder / "twin.yaml").write_text(
        "schema: molonaviz-twin/1\nname: twin_09\nmode: direkt\ndata: {measurements: absent.csv}\n"
        "sensors: {depths: [0.1, 0.2, 0.3, 0.4], ofset: 0}\nlayers: [{z_low: 0.3, porosity: 2}]\n"
    )
    with pytest.raises(TwinConfigError) as error:
        load_twin(folder)
    issues = "\n".join(error.value.issues)
    for key in ("mode:", "data.measurements:", "sensors.ofset: unknown key", "layers[0].porosity:", "layers[0].z_low:"):
        assert key in issues, key


@pytest.mark.parametrize(
    "old, new, key",
    [
        ("nb_cells: 40", "nb_cells: 40.5", "numerics.nb_cells"),
        ("initial_condition: linear", "initial_condition: steady", "numerics.initial_condition"),
        ("mode: direct", "mode: mcmc\nmcmc: {nb_chains: 3}", "nb_chains"),
        ("measurements: measurements.csv", "measurements: measurements.csv\n  start: 2016-07-12 10:00\n  end: 2016-07-12 10:15", "data"),
        ("name: twin_01", "name: twin_01\ncode_version: molonaviz2 0.0.1", "code_version"),
        ("depths: [0.1, 0.2, 0.3, 0.4]", "depths: [0.1, 0.3, 0.2, 0.4]", "sensors.depths"),
    ],
)
def test_invalid_values_are_reported(twins, point034, old, new, key):
    folder = make_twin(twins, point034)
    edit(folder, old, new)
    with pytest.raises(TwinConfigError) as error:
        load_twin(folder)
    assert any(key in issue for issue in error.value.issues), error.value.issues


def test_yaml_syntax_error_is_reported(twins):
    folder = twins / "twin_01"
    folder.mkdir()
    (folder / "twin.yaml").write_text("name: [unterminated\n")
    with pytest.raises(TwinConfigError, match="YAML"):
        load_twin(folder)


# C2 : identifiant --------------------------------------------------------------------------


def test_twin_id_ignores_labels_and_formatting(twins, point034):
    folder = make_twin(twins, point034)
    reference = load_twin(folder).twin_id("v1")
    text = (folder / "twin.yaml").read_text()
    reformatted = "# un commentaire\n" + text.replace("name: twin_01", "name: autre_nom").replace("point: Point034", "point: P34") \
        .replace("description: test", "description: autre").replace("name: couche 1", "name: graviers")
    (folder / "twin.yaml").write_text(reformatted)
    assert load_twin(folder).twin_id("v1") == reference


@pytest.mark.parametrize("old, new", [("nb_cells: 40", "nb_cells: 41"), ("porosity: 0.15", "porosity: 0.16"), ("offset: 0", "offset: 0.01")])
def test_twin_id_changes_with_anything_that_changes_the_result(twins, point034, old, new):
    folder = make_twin(twins, point034)
    reference = load_twin(folder).twin_id("v1")
    if old == "offset: 0":
        edit(folder, "z_low: 0.4", "z_low: 0.41")
    edit(folder, old, new)
    assert load_twin(folder).twin_id("v1") != reference


def test_twin_id_depends_on_data_and_code_version(twins, point034):
    folder = make_twin(twins, point034)
    config = load_twin(folder)
    reference = config.twin_id("v1")
    assert config.twin_id("v2") != reference
    data = folder / "measurements.csv"
    data.write_text(data.read_text().replace("\n", "\n", 1) + "\n")
    assert load_twin(folder).twin_id("v1") != reference


def test_lateral_exchange_keys_are_refused(twins, point034):
    """L'échange latéral (q_s, t_lateral) n'est pas pris en charge dans cette version."""
    folder = make_twin(twins, point034)
    edit(folder, "rhos_cs: 5000000", "rhos_cs: 5000000, q_s: 1.0e-7, t_lateral: 12")
    with pytest.raises(TwinConfigError) as error:
        load_twin(folder)
    assert any("q_s" in issue for issue in error.value.issues), error.value.issues
    assert any("t_lateral" in issue for issue in error.value.issues), error.value.issues


# C3 + C4 : registre, exécution, doublons ---------------------------------------------------


def test_full_lifecycle(twins, point034, monkeypatch):
    folder = make_twin(twins, point034)
    assert twin_status(folder).state == "new"
    run = run_twin(folder)
    assert run.computed and run.status.state == "computed"
    results = run.status.results_dir
    assert results.parent == twins.parent / "results" and results.name == run.status.twin_id
    assert {p.name for p in results.iterdir()} == {"temperature.csv", "water_flux.csv", "residuals.csv", "metadata.json"}
    meta = json.loads((results / "metadata.json").read_text())
    assert meta["twin_id"] == run.status.twin_id and meta["mode"] == "direct" and not meta["experimental"]
    assert meta["data"]["rows"] == 400 and len(meta["data"]["sha256"]) == 64
    entries = read_registry(twins)
    assert [(e.name, e.status) for e in entries] == [("twin_01", "computed")]

    # Relance : aucun recalcul, résultat existant.
    from molonaviz2.engine import api

    monkeypatch.setattr(api, "run_forward", lambda *a, **k: pytest.fail("ne doit pas recalculer"))
    again = run_twin(folder)
    assert not again.computed and again.status.state == "computed"

    # Copie renommée : doublon de twin_01.
    copy = twins / "twin_02"
    shutil.copytree(folder, copy)
    edit(copy, "name: twin_01", "name: twin_02")
    duplicate = run_twin(copy)
    assert not duplicate.computed and duplicate.status.state == "duplicate" and duplicate.status.source.name == "twin_01"
    assert duplicate.status.results_dir == results
    assert [(e.name, e.status, e.duplicate_of) for e in read_registry(twins)][-1] == ("twin_02", "duplicate", "twin_01")

    # Config figée : modifier twin_01 après exécution est refusé.
    edit(folder, "porosity: 0.15", "porosity: 0.2")
    assert twin_status(folder).state == "modified"
    with pytest.raises(UserError, match="frozen"):
        run_twin(folder)


def test_results_are_identical_bit_for_bit(tmp_path, point034):
    folders = []
    for name in ("a", "b"):
        twins = tmp_path / name / "twins"
        twins.mkdir(parents=True)
        folder = make_twin(twins, point034)
        folders.append(run_twin(folder).status.results_dir)
    for file in ("temperature.csv", "water_flux.csv", "residuals.csv"):
        assert (folders[0] / file).read_bytes() == (folders[1] / file).read_bytes(), file


@pytest.mark.parametrize("head, direction", [(0.05, "infiltration"), (-0.05, "exfiltration")])
def test_flux_sign_convention(twins, point034, head, direction):
    data = point034.copy()
    data["pressure"] = head  # rivière plus haute que la nappe : l'eau descend
    run = run_twin(make_twin(twins, data))
    flux = load_results(run.status.results_dir).flux
    assert (flux["direction"].iloc[10:] == direction).all()


def test_mcmc_twin_is_marked_experimental(twins, point034):
    config = InversionConfig(layers=[LayerSpec("couche 1", 0.4)], nb_cells=20, nb_chains=7, nb_iter=10, burn_in_max=10, n_field_samples=7)
    folder = make_twin(twins, point034.iloc[:150], config, sigma_t=0.2)
    assert "mcmc:" in (folder / "twin.yaml").read_text()
    assert load_twin(folder).name == "twin_01"  # régression : le nom était écrasé par la boucle des lois a priori
    run = run_twin(folder)
    results = load_results(run.status.results_dir)
    assert results.metadata["name"] == "twin_01" and read_registry(twins)[0].name == "twin_01"
    assert results.metadata["experimental"] is True
    assert {"burn_in_iterations", "acceptance_mean", "rhat", "best_parameters"} <= set(results.metadata["mcmc"])
    assert (run.status.results_dir / "posterior_summary.csv").is_file()
    assert any(c.startswith("q_riverbed_q") for c in results.flux.columns)


def test_listing_and_next_twin_name(twins):
    for name in ("twin_2", "twin_10", "twin_1", "notes"):
        (twins / name).mkdir()
        (twins / name / "twin.yaml").write_text("name: x\n")
    (twins / "twin_99").mkdir()  # sans twin.yaml : ignoré
    assert [p.name for p in list_twins(twins)] == ["twin_1", "twin_2", "twin_10"]
    assert next_twin_folder(twins).name == "twin_100"


def test_command_line(twins, point034):
    folder = make_twin(twins, point034)

    def cli(*args):
        return subprocess.run([sys.executable, "-m", "molonaviz2", "twin", *args], capture_output=True, text=True, timeout=300)

    assert cli("validate", str(folder)).returncode == 0
    identity = cli("id", str(folder))
    assert identity.returncode == 0 and identity.stdout.split()[0] == load_twin(folder).twin_id(__import__("molonaviz2.engine.api", fromlist=["x"]).engine_version())
    assert cli("run", str(folder)).returncode == 0
    assert cli("run", str(folder)).returncode == 0  # doublon : succès, sans recalcul
    assert [e.status for e in read_registry(twins)] == ["computed"]
    edit(folder, "porosity: 0.15", "porosity: 0.3")
    assert cli("run", str(folder)).returncode == 1  # config figée
    assert cli("validate", str(twins / "absent")).returncode == 1
    assert "twin_01" in cli("list", str(twins)).stdout
