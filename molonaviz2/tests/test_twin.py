"""Twins : validation (C1), identifiant (C2), registre et doublons (C3), exécution et résultats (C4)."""

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from conftest import FIXTURES, git_run

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
# Champs du format 2.0.0, avec la première cellule (profondeur|unité).
FIELD_FILES = {"water_flux_field.csv": "depth_m|m_per_s", "heat_flux_advective.csv": "depth_m|W_per_m2", "heat_flux_conductive.csv": "depth_m|W_per_m2"}


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
    assert "schema: molonari-twin/1.1.0" in text and "depths: [0.1, 0.2, 0.3, 0.4]" in text
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


# Schéma ------------------------------------------------------------------------------------

EXAMPLES = Path(__file__).resolve().parents[1] / "examples" / "twins"
EXAMPLE_IDS_V1 = {"twin_01": "aac866e95333d10f", "twin_02": "aac866e95333d10f", "twin_03": "864cad617b96bb5d", "twin_04": "d20973170dbd571a"}


def copy_of_twin_03(twins, schema="molonari-twin/1.1.0", extra=""):
    folder = twins / "twin_03"
    shutil.copytree(EXAMPLES / "twin_03", folder)
    edit(folder, "schema: molonaviz-twin/1", f"schema: {schema}")
    if extra:
        edit(folder, "  time_step: 1\n", f"  time_step: 1\n{extra}")
    return folder


@pytest.mark.parametrize("name", sorted(EXAMPLE_IDS_V1))
def test_examples_keep_their_twin_id(name):
    assert load_twin(EXAMPLES / name).twin_id("v1") == EXAMPLE_IDS_V1[name]


@pytest.mark.parametrize("schema", ["molonari-twin/1.0.0", "molonari-twin/1.1.0"])
def test_new_schema_names_are_read(twins, schema):
    assert load_twin(copy_of_twin_03(twins, schema)).twin_id("v1") == EXAMPLE_IDS_V1["twin_03"]


@pytest.mark.parametrize("schema, extra, key, code", [
    ("molonari-twin/2.0.0", "", "schema", "E_TWIN_SCHEMA"),
    ("molonaviz-twin/1", "  thin: 2\n", "mcmc.thin", "E_TWIN_UNKNOWN_KEY"),
    ("molonari-twin/1.0.0", "  store_fields: full\n", "mcmc.store_fields", "E_TWIN_UNKNOWN_KEY"),
    ("molonari-twin/1.1.0", "  fields_per_chain: 5\n", "mcmc.fields_per_chain", "E_MCMC_FIELDS_BOTH"),
    ("molonari-twin/1.1.0", "  store_fields: tout\n", "mcmc.store_fields", "E_MCMC_STORE_FIELDS"),
])
def test_schema_and_keys_of_1_1_0_are_checked(twins, schema, extra, key, code):
    with pytest.raises(TwinConfigError) as error:
        load_twin(copy_of_twin_03(twins, schema, extra))
    assert [(issue.field, issue.code) for issue in error.value.issues] == [(key, code)]


def test_keys_of_1_1_0_are_read(twins):
    folder = copy_of_twin_03(twins, extra="  thin: 2\n  store_fields: full\n")
    edit(folder, "  n_field_samples: 40\n", "  fields_per_chain: 5\n")
    mcmc = load_twin(folder).inversion_config()
    assert (mcmc.thin, mcmc.fields_per_chain, mcmc.store_fields) == (2, 5, "full")


def test_twin_id_ignores_keys_of_1_1_0_at_their_default(twins):
    folder = copy_of_twin_03(twins, extra="  store_fields: sensors\n  thin: 1\n")
    assert load_twin(folder).twin_id("v1") == EXAMPLE_IDS_V1["twin_03"]
    edit(folder, "  thin: 1\n", "  thin: 2\n")
    assert load_twin(folder).twin_id("v1") != EXAMPLE_IDS_V1["twin_03"]


def test_created_mcmc_twin_declares_1_1_0_without_default_keys(twins, point034):
    config = InversionConfig(layers=[LayerSpec("couche 1", 0.4)], nb_cells=20, nb_chains=7, nb_iter=10, burn_in_max=10, n_field_samples=7)
    text = (make_twin(twins, point034, config) / "twin.yaml").read_text()
    assert "schema: molonari-twin/1.1.0" in text and "n_field_samples: 7" in text
    assert not any(key in text for key in ("thin", "fields_per_chain", "store_fields"))
    config.thin, config.fields_per_chain, config.store_fields = 2, 3, "none"
    mcmc = load_twin(make_twin(twins, point034, config)).inversion_config()
    assert (mcmc.thin, mcmc.fields_per_chain, mcmc.store_fields, mcmc.n_field_samples) == (2, 3, "none", 200)


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
    assert {p.name for p in results.iterdir()} == {"temperature.csv", "water_flux.csv", "residuals.csv", "metadata.json", *FIELD_FILES}
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


def test_direct_results_2_0_0_fields_and_metadata(twins, point034):
    from molonaviz2.engine import api

    folder = make_twin(twins, point034)
    results_dir = run_twin(folder).status.results_dir
    config = load_twin(folder)
    forward = api.run_forward(config.column_inputs(), config.forward_config())
    expected = {"water_flux_field.csv": forward.water_flux, "heat_flux_advective.csv": forward.advective_flux,
                "heat_flux_conductive.csv": forward.conductive_flux_down}
    for name, index_name in FIELD_FILES.items():
        frame = pd.read_csv(results_dir / name, index_col=0, float_precision="round_trip")
        assert frame.index.name == index_name
        np.testing.assert_array_equal(frame.index.to_numpy(float), forward.depths)
        np.testing.assert_array_equal(frame.to_numpy(float), expected[name])
    metadata = json.loads((results_dir / "metadata.json").read_text())
    assert metadata["results_format"] == "2.0.0"
    assert metadata["instruments"] == {"sigma_temperature_C": 0.05, "sigma_pressure_m": 0.01}
    assert metadata["conventions"]["advective_reference_C"] == 0.0
    assert {"z", "q", "heat_flux"} <= set(metadata["conventions"]) and "downwards" in metadata["conventions"]["heat_flux"]
    assert not any(results_dir.glob("*.npz"))  # tableaux .npz : mode mcmc seulement


def test_results_2_0_0_are_identical_bit_for_bit(tmp_path, point034):
    folders = []
    for name in ("a", "b"):
        twins = tmp_path / name / "twins"
        twins.mkdir(parents=True)
        folders.append(run_twin(make_twin(twins, point034)).status.results_dir)
    for file in FIELD_FILES:
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
    assert {p.name for p in run.status.results_dir.glob("*.npz")} == {"quantiles.npz", "posterior.npz", "fields_temperature.npz"}
    assert results.results_format == "2.0.0" and "W_FIELD_UNAVAILABLE" not in [w.code for w in results.warnings]
    assert results.posterior["samples"].shape == (10, 7, 1, 4) and set(results.fields) == {"temperature"}
    assert results.fields["temperature"]["values"].shape[1] == 3  # aux 3 capteurs (store_fields par défaut)
    assert results.quantiles["temperature_C"].shape[0] == 3 and results.heat_flux_conductive.shape == results.temperature.shape
    assert any(c.startswith("q_riverbed_q") for c in results.flux.columns)


def as_format_1(results_dir):
    """Ramène un dossier de résultats direct au format 1.x : sans les fichiers ni les clés de 2.0.0."""
    for name in FIELD_FILES:
        (results_dir / name).unlink()
    metadata = json.loads((results_dir / "metadata.json").read_text())
    for key in ("results_format", "instruments", "conventions"):
        del metadata[key]
    (results_dir / "metadata.json").write_text(json.dumps(metadata))
    return results_dir


def test_results_of_format_1_are_read_with_a_warning(twins, point034):
    results_dir = run_twin(make_twin(twins, point034)).status.results_dir
    current = load_results(results_dir)
    assert current.results_format == "2.0.0" and current.quantiles is None and current.fields == {}
    assert "W_FIELD_UNAVAILABLE" not in [w.code for w in current.warnings]  # tmp_path hors dépôt : seulement W_PROVENANCE_*
    written = pd.read_csv(results_dir / "heat_flux_conductive.csv", index_col=0, float_precision="round_trip")
    np.testing.assert_array_equal(current.heat_flux_conductive, written.to_numpy(float))
    old = as_format_1(shutil.copytree(results_dir, twins.parent / "old"))
    previous = load_results(old)
    assert previous.results_format == "1" and previous.warnings[0].code == "W_FIELD_UNAVAILABLE"
    assert previous.heat_flux_conductive is None and previous.water_flux_field is None and previous.posterior is None
    np.testing.assert_array_equal(previous.temperature, current.temperature)
    pd.testing.assert_frame_equal(previous.flux, current.flux)


def small_inversion_config(**overrides):
    values = {"layers": [LayerSpec("couche 1", 0.4)], "nb_cells": 20, "nb_chains": 7, "nb_iter": 10, "burn_in_max": 10, "n_field_samples": 7}
    return InversionConfig(**{**values, **overrides})


def load_npz(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


@pytest.mark.parametrize("store_fields, files", [
    ("sensors", {"fields_temperature.npz"}),
    ("full", {"fields_temperature.npz", "fields_water_flux.npz"}),
    ("none", set()),
])
def test_mcmc_arrays_2_0_0(tmp_path, point034, store_fields, files):
    from molonaviz2.engine import api
    from molonaviz2.twin.results import write_results

    twins = tmp_path / "twins"
    twins.mkdir()
    config = load_twin(make_twin(twins, point034.iloc[:150], small_inversion_config(thin=2, store_fields=store_fields), sigma_t=0.2))
    inversion = api.run_inversion(config.column_inputs(), config.inversion_config())
    folder = tmp_path / "results"
    write_results(folder, inversion.best, inversion, ["couche 1"], {})
    assert {p.name for p in folder.glob("*.npz")} == {"quantiles.npz", "posterior.npz", *files}

    quantiles = load_npz(folder / "quantiles.npz")
    np.testing.assert_array_equal(quantiles["quantile"], [0.05, 0.5, 0.95])
    assert quantiles["time"].dtype == np.dtype("datetime64[s]") and quantiles["depth_m"].shape == inversion.field_depths.shape
    np.testing.assert_array_equal(quantiles["temperature_C"], np.stack([q.temperature for q in inversion.quantiles]))
    np.testing.assert_array_equal(quantiles["water_flux_m_per_s"], np.stack([q.water_flux for q in inversion.quantiles]))

    posterior = load_npz(folder / "posterior.npz")
    assert posterior["samples"].shape == (5, 7, 1, 4)  # 10 itérations, thin = 2
    np.testing.assert_array_equal(posterior["samples"], inversion.samples)
    np.testing.assert_array_equal(posterior["energy"], inversion.energies)
    np.testing.assert_array_equal(posterior["acceptance"], inversion.acceptance)
    np.testing.assert_array_equal(posterior["resets"], inversion.resets)
    np.testing.assert_array_equal(posterior["iteration"], [2, 4, 6, 8, 10])
    assert list(posterior["parameter"]) == ["neg_log10_k", "porosity", "lambda_s", "rhos_cs"] and list(posterior["layer"]) == ["couche 1"]

    if "fields_temperature.npz" in files:
        fields = load_npz(folder / "fields_temperature.npz")
        np.testing.assert_array_equal(fields["chain"], inversion.field_chain)
        np.testing.assert_array_equal(fields["iteration"], inversion.field_iteration)
        expected = inversion.sensor_temperature if store_fields == "sensors" else inversion.field_temperature
        np.testing.assert_array_equal(fields["values"], expected)
        assert fields["values"].dtype == expected.dtype
        depths = inversion.best.sensor_depths if store_fields == "sensors" else inversion.field_depths
        np.testing.assert_array_equal(fields["depth_m"], depths)
        assert fields["values"].shape == (len(fields["chain"]), len(depths), len(fields["time"]))
    if "fields_water_flux.npz" in files:
        np.testing.assert_array_equal(load_npz(folder / "fields_water_flux.npz")["values"], inversion.field_water_flux)


# Provenance --------------------------------------------------------------------------------


def test_provenance_of_a_twin_in_a_git_repository(twins, point034):
    import os
    import time

    folder = make_twin(twins, point034)
    workspace = twins.parent  # contient twins/ (dépôt du twin) et .cgitsync/ (espace cgs)
    state = workspace / ".cgitsync" / "state"
    state.mkdir(parents=True)
    (state / "ancien.gts").write_text("ancien")
    (state / "recent.gts").write_text("récent")
    os.utime(state / "ancien.gts", (time.time() - 3600, time.time() - 3600))
    git_run(twins, "init", "-q")
    git_run(twins, "remote", "add", "origin", "https://user:ghp_secret@github.com/org/twins.git")
    git_run(twins, "add", "twin_01")
    git_run(twins, "commit", "-q", "-m", "twin_01")
    git_run(twins, "tag", "campagne-2016")

    run = run_twin(folder)
    text = (run.status.results_dir / "metadata.json").read_text()
    assert "ghp_secret" not in text
    provenance = json.loads(text)["provenance"]
    twin = provenance["twin"]
    assert twin["remote"] == "https://github.com/org/twins.git" and twin["commit"] == git_run(twins, "rev-parse", "HEAD")
    assert twin["tags"] == ["campagne-2016"] and twin["branch"] == "main" and twin["clean"] is True and twin["path"] == "twin_01"
    assert provenance["code"]["commit"] is not None and provenance["code"]["path"].endswith("molonaviz2")
    environment = provenance["environment"]
    assert all(environment[name] for name in ("python", "platform", "processor", "device", "jax", "jaxlib", "numpy"))
    assert provenance["cgs_state_sha256"] == hashlib.sha256("récent".encode()).hexdigest()
    assert provenance["warnings"] == [] and run.warnings == []


def test_provenance_outside_a_repository_is_null_with_a_warning(twins, point034):
    folder = make_twin(twins, point034)  # tmp_path n'est dans aucun dépôt
    run = run_twin(folder)
    provenance = json.loads((run.status.results_dir / "metadata.json").read_text())["provenance"]
    assert provenance["twin"]["commit"] is None and provenance["twin"]["remote"] is None
    assert provenance["warnings"] == [{"code": "W_PROVENANCE_NOT_A_REPOSITORY", "params": {"target": "twin"}}]
    assert [(w.code, w.params) for w in run.warnings] == [("W_PROVENANCE_NOT_A_REPOSITORY", {"target": "twin"})]
    assert "W_PROVENANCE_NOT_A_REPOSITORY" in [w.code for w in load_results(run.status.results_dir).warnings]


def test_provenance_without_remote_or_git(twins, point034, monkeypatch):
    from molonaviz2.twin import provenance as provenance_module

    folder = make_twin(twins, point034)
    git_run(twins, "init", "-q")
    entry = provenance_module.twin_provenance(folder, "CPU")
    assert entry["twin"]["remote"] is None and entry["twin"]["clean"] is False  # twin_01 non suivi
    assert {"code": "W_PROVENANCE_NO_REMOTE", "params": {"target": "twin"}} in entry["warnings"]
    monkeypatch.setenv("PATH", "")
    entry = provenance_module.twin_provenance(folder, "CPU")
    assert entry["code"]["commit"] is None and entry["twin"]["commit"] is None
    assert entry["warnings"] == [{"code": "W_PROVENANCE_NO_GIT", "params": {}}]


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
        return subprocess.run([sys.executable, "-m", "molonaviz2", "twin", *args], capture_output=True, text=True, timeout=300, check=False)

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


# Interface ---------------------------------------------------------------------------------


def test_welcome_page_lists_twins_and_window_runs_them(qtbot, tmp_path, twins, point034, monkeypatch):
    from PySide6 import QtCore

    from molonaviz2.settings import Settings
    from molonaviz2.ui import errors as ui_errors
    from molonaviz2.ui.main_window import MainWindow

    shown = []
    previous = ui_errors.set_presenter(lambda *args: shown.append(args))
    folder = make_twin(twins, point034)
    settings = Settings(QtCore.QSettings(str(tmp_path / "s.ini"), QtCore.QSettings.Format.IniFormat))
    settings.twin_folder = twins
    window = MainWindow(settings)
    qtbot.addWidget(window)
    assert window.welcome.twins.count() == 1
    assert "not run yet" in window.welcome.twins.item(0).text()

    twin_window = window.open_twin(folder)
    assert twin_window.run_button.isEnabled() and not twin_window.tabs.isVisible()
    twin_window.run()
    qtbot.waitUntil(lambda: not window.tasks.busy, timeout=120_000)
    qtbot.waitUntil(lambda: twin_window.results is not None, timeout=10_000)
    assert twin_window.status.state == "computed"
    assert not twin_window.run_button.isEnabled() and "Already computed" in twin_window.run_button.toolTip()
    assert "Infiltration" in twin_window.flux_summary.text()
    from molonaviz2.ui.units import measurement_level

    sigma = twin_window.status.config.sigma_temperature
    dashed = {round(abs(float(line.get_ydata()[0])), 9) for line in twin_window.residuals.figure.axes[0].lines if line.get_linestyle() == "--"}
    assert dashed == {round(measurement_level.z * sigma, 9)}  # résidus : lignes à ± z·σT (90 % par défaut)
    window.welcome.refresh_twins(twins)
    assert "computed" in window.welcome.twins.item(0).text()

    copy = twins / "twin_02"
    shutil.copytree(folder, copy)
    edit(copy, "name: twin_01", "name: twin_02")
    duplicate_window = window.open_twin(copy)
    assert "Duplicate of twin_01" in duplicate_window.state.text() and duplicate_window.results is not None
    ui_errors.set_presenter(previous)
    assert shown == []


def test_twin_window_warns_about_results_of_format_1(qtbot, tmp_path, twins, point034):
    from PySide6 import QtCore

    from molonaviz2.settings import Settings
    from molonaviz2.ui.main_window import MainWindow

    folder = make_twin(twins, point034)
    as_format_1(run_twin(folder).status.results_dir)
    settings = Settings(QtCore.QSettings(str(tmp_path / "s.ini"), QtCore.QSettings.Format.IniFormat))
    window = MainWindow(settings)
    qtbot.addWidget(window)
    twin_window = window.open_twin(folder)
    assert twin_window.results is not None and not twin_window.result_warnings.isHidden()
    assert "before format 2.0.0" in twin_window.result_warnings.text()
    run_twin_window = window.open_twin(run_twin(make_twin(twins, point034.iloc[:200])).status.config.folder)
    assert "before format 2.0.0" not in run_twin_window.result_warnings.text()


def test_create_twin_from_a_computation_in_the_app(qtbot, tmp_path, twins, point034, monkeypatch):
    from PySide6 import QtCore

    from molonaviz2.domain.models import SamplingPoint, Study
    from molonaviz2.settings import Settings
    from molonaviz2.storage.project import Project
    from molonaviz2.ui.main_window import MainWindow

    project = Project.create(tmp_path / "p.molo")
    project.add_lab(import_lab_directory(FIXTURES / "lab", name="Lab").lab)
    project.add_study(Study("E", "Lab"))
    raw_t = read_temperature_file(FIXTURES / "point034" / "point034_T_measures.csv").frame.iloc[:200]
    raw_p = read_pressure_file(FIXTURES / "point034" / "point034_P_measures.csv").frame.iloc[:200]
    project.add_point(SamplingPoint("Point034", "E", "P508", "S007", river_bed=0.505), raw_t, raw_p)
    settings = Settings(QtCore.QSettings(str(tmp_path / "s.ini"), QtCore.QSettings.Format.IniFormat))
    settings.twin_folder = twins
    window = MainWindow(settings, project)
    qtbot.addWidget(window)
    window.open_point("E", "Point034")
    view = window.point_tabs.currentWidget()
    view.cleanup_tab.apply()
    view.forward_tab.nb_cells.set_value(30)
    view.forward_tab.run()
    qtbot.waitUntil(lambda: not window.tasks.busy, timeout=120_000)
    view.results_tab.refresh()
    monkeypatch.setattr("molonaviz2.ui.main_window.ask", lambda *a, **k: False)
    created = window.create_twin_from_computation(view.context, view.results_tab.info)
    assert created == twins / "twin_01"
    config = load_twin(created)
    assert config.point == "Point034" and config.nb_cells == 30 and config.depths == (0.1, 0.2, 0.3, 0.4)
    data = pd.read_csv(created / "measurements.csv")
    assert len(data) == len(project.cleaned("E", "Point034"))
    # Même calcul dans l'application et dans le twin.
    run = run_twin(created)
    app_result = view.results_tab.forward
    twin_temperature = load_results(run.status.results_dir).temperature
    assert np.max(np.abs(twin_temperature - app_result.temperature)) < 1e-4  # mesures relues depuis le CSV (6 chiffres)
