"""Générateur du rapport public (P2c.3) : entrées obligatoires et par défaut, étiquettes de modèle, visibilité, index,
fichiers, déterminisme, commande et, en `slow`, un space de deux instances 2.0.0 sans la graine."""

import dataclasses
import json
import shutil
import time
from pathlib import Path

import numpy as np
import pytest
from conftest import FIXTURES
from molonari_twin.api import TwinAPI
from molonari_twin.contract.codec import answer_from_json, answer_from_npz
from molonari_twin.errors import UserError
from molonari_twin.questions import Visibility
from molonari_twin.report import (
    FOLDER_NAME,
    REPORT_FORMAT,
    Space,
    build_report,
    generate_report,
    resolve_space,
)
from molonari_twin.report.generator import effective_visibility, plan
from molonari_twin.report.writer import dump_json, encode_entry, sha256, write_folder
from molonari_twin.twin.cli import main as cli_main
from molonari_twin.twin.config import ReportEntry, default_report, load_twin
from molonari_twin.twin.run import run_twin


def space_of(folder: Path, refs: dict, entries=None, **visibility) -> Space:
    """Space des twins 1.x du dossier de test (twin_01 direct, twin_03 mcmc), avec le rapport par défaut (pas de clé
    `report` dans un twin 1.x). `visibility` remplace des éléments déclarés dans chaque twin (2.0.0 seulement en vrai)."""
    configs = {role: load_twin(folder / ref) for role, ref in refs.items()}
    if visibility:
        configs = {role: dataclasses.replace(c, visibility={**c.visibility, **visibility}) for role, c in configs.items()}
    return Space(next(reversed(refs.values())), refs, configs, tuple(entries) if entries is not None else default_report().entries, False)


BOTH = {"forward": "twin_01", "inversion": "twin_03"}
FORWARD = {"forward": "twin_01"}
INVERSION = {"inversion": "twin_03"}


@pytest.fixture(scope="module")
def reports(computed_twins):
    api = TwinAPI(computed_twins)
    return {name: build_report(space_of(computed_twins, refs), api) for name, refs in
            (("both", BOTH), ("forward", FORWARD), ("inversion", INVERSION))}


def entries_of(report, **match):
    return [e for e in report.index["entries"] if all(e[k] == v for k, v in match.items())]


def answer(report, file):
    content = report.files[file]
    return answer_from_npz(content) if file.endswith(".npz") else answer_from_json(content)


# P2c.3c : étiquettes de modèle ---------------------------------------------------------------------------------


def test_direct_twin_is_publishable_and_labelled_without_uncertainty(reports):
    report = reports["forward"]
    assert report.index["space"]["inversion"] is None and report.index["space"]["forward"]["twin"] == "twin_01"
    assert {e["model"] for e in report.index["entries"]} == {"forward"}
    assert {e["uncertainty"] for e in report.index["entries"]} == {"none"}
    for entry in report.index["entries"]:
        loaded = answer(report, entry["file"])
        assert "W_NO_UNCERTAINTY" in [w.code for w in loaded.warnings], entry["file"]
        assert all(q.interval is None for q in loaded.data.values()), entry["file"]
    assert {e["question"] for e in report.index["entries"]} == {"describe", "provenance", "water_flux", "fit_quality"}
    assert not any(e["question"] == "compare" for e in report.index["entries"])


def test_mcmc_twin_gives_the_flux_with_a_90_percent_interval_rmse_and_convergence(reports):
    report = reports["inversion"]
    assert report.index["space"]["forward"] is None
    assert {e["model"] for e in report.index["entries"]} == {"inversion"}
    assert {e["uncertainty"] for e in report.index["entries"]} == {"posterior"}
    flux = answer(report, "inversion-water_flux.json")
    assert flux.data["mean"].interval is not None and flux.data["level"].value == 0.9
    assert "W_NO_UNCERTAINTY" not in [w.code for w in flux.warnings]
    fit = answer(report, "inversion-fit_quality.json")
    assert np.asarray(fit.data["rmse"].value).shape == (3,) and np.asarray(fit.data["bias"].value).shape == (3,)
    assert fit.data["rmse_total"].value > 0 and "median_rmse" in fit.data
    convergence = answer(report, "inversion-public_convergence.json")
    assert {"rhat_max", "ess_min", "acceptance_mean"} <= set(convergence.data)


def test_two_models_add_the_direct_comparison_and_both_sets(reports):
    report = reports["both"]
    assert [e["model"] for e in report.index["entries"]].count("both") == 1
    compare = answer(report, "compare.json")
    assert {"mean_difference", "rmse", "rmse_total"} <= set(compare.data)
    assert entries_of(report, file="compare.json")[0]["uncertainty"] == "none" and entries_of(report, file="compare.json")[0]["ref"] is None
    assert len(entries_of(report, question="water_flux")) == 2 and len(entries_of(report, question="fit_quality")) == 2
    assert report.index["space"]["forward"]["twin"] == "twin_01" and report.index["space"]["inversion"]["twin"] == "twin_03"


# P2c.3d, P2c.3e : JSON, npz ---------------------------------------------------------------------------------


def test_posterior_samples_and_chain_data_are_npz_and_read_without_pickle(reports):
    report = reports["inversion"]
    for question in ("posterior_samples", "chain_data"):
        (entry,) = entries_of(report, question=question)
        assert entry["file"].endswith(".npz") and entry["format"] == "npz"
        loaded = answer_from_npz(report.files[entry["file"]])
        assert loaded.provenance["question"] == question
    samples = answer(report, "inversion-posterior_samples.npz").data["samples"].value
    assert samples.ndim == 4 and samples.shape[-1] == 4
    assert entries_of(report, question="soil_properties")  # le résumé de la PDF, en JSON
    assert entries_of(report, question="soil_properties")[0]["format"] == "json"


def test_an_entry_switches_to_npz_beyond_the_threshold_and_keeps_its_values(computed_twins):
    api = TwinAPI(computed_twins)
    space = space_of(computed_twins, INVERSION)
    json_report = build_report(space, api)
    npz_report = build_report(space, api, npz_threshold=500)
    assert entries_of(json_report, question="water_flux")[0]["format"] == "json"
    flux = entries_of(npz_report, question="water_flux")[0]
    assert flux["format"] == "npz" and flux["file"] == "inversion-water_flux.npz"
    assert answer(npz_report, "inversion-water_flux.npz").data["mean"] == answer(json_report, "inversion-water_flux.json").data["mean"]
    extension, _ = encode_entry("water_flux", answer(json_report, "inversion-water_flux.json"), threshold=1 << 20)
    assert extension == "json"


def test_no_nan_in_a_json_entry(reports):
    for report in reports.values():
        for name, content in report.files.items():
            if name.endswith(".json"):
                json.loads(content, parse_constant=lambda c, name=name: pytest.fail(f"{c} in {name}"))


# P2c.3f : figures -------------------------------------------------------------------------------------------


def test_figures_are_derived_from_entries_with_their_fingerprint(reports):
    report = reports["both"]
    figures = report.index["figures"]
    assert {f["file"] for f in figures} == {"figures/forward-water_flux.svg", "figures/inversion-water_flux.svg",
                                              "figures/forward-fit_quality.svg", "figures/inversion-fit_quality.svg"}
    files = {e["file"] for e in report.index["entries"]}
    for figure in figures:
        assert figure["from_entry"] in files and figure["sha256"] == sha256(report.files[figure["file"]])
        assert report.files[figure["file"]].startswith(b"<svg")
    assert b"no confidence interval" in report.files["figures/forward-water_flux.svg"]
    assert b"90 % posterior interval" in report.files["figures/inversion-water_flux.svg"]


# P2c.3b : index ---------------------------------------------------------------------------------------------


def test_index_fields(reports, computed_twins):
    report = reports["both"]
    index = report.index
    assert index["report_format"] == REPORT_FORMAT == "1.0.0"
    assert {"space", "twin_semver", "generated_at", "generator_version", "platform", "visibility", "entries", "figures", "skipped",
            "signature", "license"} <= set(index)
    assert index["generator_version"].startswith("molonari-twin ") and index["twin_semver"] is None
    from molonari_twin.twin.run import twin_status

    created = [json.loads((twin_status(computed_twins / ref).results_dir / "metadata.json").read_text())["created_at"] for ref in BOTH.values()]
    assert index["generated_at"] == max(created)  # une date de calcul, jamais l'horloge
    assert set(index["visibility"]) == set(Visibility().to_dict())
    for entry in index["entries"]:
        assert {"file", "format", "question", "ref", "model", "uncertainty", "parameters", "sha256"} <= set(entry)
    assert json.loads(report.files["report.json"]) == json.loads(json.dumps(index))


def test_recipe_fingerprint_is_the_dev_recipe_without_the_seed(reports, computed_twins):
    from molonari_twin.twin.config import without_seed
    from molonari_twin.twin.run import twin_status

    status = twin_status(computed_twins / "twin_03")
    stored = json.loads((status.results_dir / "metadata.json").read_text())["config"]
    recipe = json.dumps(without_seed(stored), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    assert reports["inversion"].index["space"]["inversion"]["recipe_sha256"] == sha256(recipe.encode())
    assert "seed" not in without_seed(stored)


# P2c.3g : refus et sauts -----------------------------------------------------------------------------------


def test_posterior_entries_are_skipped_when_the_pdf_is_private(computed_twins):
    report = build_report(space_of(computed_twins, BOTH, posterior=False), TwinAPI(computed_twins))
    skipped = {(s["question"], s["model"]): s for s in report.index["skipped"]}
    assert set(skipped) == {("soil_properties", "forward"), ("soil_properties", "inversion"), ("posterior_samples", "inversion"),
                            ("chain_data", "inversion")}
    assert all(s["code"] == "E_REPORT_PRIVATE" for s in skipped.values())
    assert "posterior" in skipped[("posterior_samples", "inversion")]["params"]["elements"]
    assert not [name for name in report.files if "posterior" in name or "chain_data" in name or "soil" in name]
    assert entries_of(report, question="water_flux", model="inversion") and report.index["visibility"]["posterior"] is False


def test_forward_parameters_are_published_only_when_public(computed_twins):
    api = TwinAPI(computed_twins)
    private = build_report(space_of(computed_twins, FORWARD), api)
    public = build_report(space_of(computed_twins, FORWARD, forward_parameters=True), api)
    assert ("soil_properties", "forward") in {(s["question"], s["model"]) for s in private.index["skipped"]}
    assert "forward-soil_properties.json" in public.files and "forward-soil_properties.json" not in private.files


def test_a_declared_entry_that_is_private_fails_and_nothing_is_built(computed_twins):
    entries = (*default_report().entries, ReportEntry("temperature", {"z": 0.2}))
    with pytest.raises(UserError) as error:
        build_report(space_of(computed_twins, BOTH, entries), TwinAPI(computed_twins))
    assert error.value.code == "E_REPORT_PRIVATE" and "states" in error.value.params["elements"]
    with pytest.raises(UserError) as error:  # publique, mais sans projection : la grille de requêtes n'est pas décidée (T3)
        build_report(space_of(computed_twins, BOTH, entries, states=True), TwinAPI(computed_twins))
    assert error.value.code == "E_REPORT_NO_PROJECTION"


def test_a_developer_only_declared_entry_is_refused(computed_twins):
    entries = (*default_report().entries, ReportEntry("residuals", {}))
    with pytest.raises(UserError) as error:
        build_report(space_of(computed_twins, FORWARD, entries), TwinAPI(computed_twins))
    assert error.value.code == "E_REPORT_DEV_ONLY"


def test_a_required_entry_with_a_private_element_fails_the_report(computed_twins):
    with pytest.raises(UserError) as error:  # describe expose les réglages d'échantillonnage, que la PDF publique révèle
        build_report(space_of(computed_twins, INVERSION, inversion_settings_public=False), TwinAPI(computed_twins))
    assert error.value.code == "E_REPORT_PRIVATE"


def test_a_question_that_does_not_apply_to_a_model_is_skipped(computed_twins):
    entries = (*default_report().entries, ReportEntry("exchange_direction", {}), ReportEntry("public_convergence", {}))
    report = build_report(space_of(computed_twins, BOTH, entries), TwinAPI(computed_twins))
    assert entries_of(report, question="exchange_direction", model="forward") and entries_of(report, question="exchange_direction", model="inversion")
    assert [s["code"] for s in report.index["skipped"] if s["question"] == "public_convergence"] == ["E_QUESTION_NOT_AVAILABLE"]
    assert len(entries_of(report, question="public_convergence")) == 1  # déclarée et obligatoire : produite une fois


def test_a_report_with_no_entry_is_refused(computed_twins):
    with pytest.raises(UserError) as error:
        build_report(Space("x", {}, {}, (), False), TwinAPI(computed_twins))
    assert error.value.code == "E_REPORT_EMPTY"


def test_a_twin_without_results_is_refused(computed_twins):
    refs = {"forward": "twin_09"}
    with pytest.raises(UserError) as error:
        build_report(space_of(computed_twins, refs), TwinAPI(computed_twins))
    assert error.value.code == "E_TWIN_NOT_COMPUTED"


def test_plan_order_is_fixed(computed_twins):
    items = plan(space_of(computed_twins, BOTH))
    assert [(i.role, i.question) for i in items] == [
        ("forward", "describe"), ("forward", "provenance"), ("forward", "water_flux"), ("forward", "fit_quality"),
        ("inversion", "describe"), ("inversion", "provenance"), ("inversion", "water_flux"), ("inversion", "fit_quality"),
        ("inversion", "public_convergence"), ("both", "compare"),
        ("forward", "soil_properties"), ("inversion", "soil_properties"), ("inversion", "posterior_samples"), ("inversion", "chain_data")]
    assert [i.required for i in items] == [True] * 10 + [False] * 4


def test_effective_visibility_is_the_strictest_of_the_twins_it_applies_to(computed_twins):
    forward, inversion = load_twin(computed_twins / "twin_01"), load_twin(computed_twins / "twin_03")
    forward = dataclasses.replace(forward, visibility={**forward.visibility, "forward_parameters": True, "posterior": False})
    inversion = dataclasses.replace(inversion, visibility={**inversion.visibility, "states": True, "posterior": True})
    visibility = effective_visibility([forward, inversion])
    assert visibility.forward_parameters is True  # l'élément n'existe que dans le twin direct : le twin mcmc ne le contredit pas
    assert visibility.posterior is True  # la PDF n'existe que dans le twin mcmc
    assert visibility.states is False  # le twin direct ne le déclare pas public
    assert effective_visibility([inversion, dataclasses.replace(inversion, visibility={**inversion.visibility, "posterior": False})]).posterior is False


# Aucune fuite --------------------------------------------------------------------------------------------------


def test_nothing_private_or_seed_related_is_written(computed_twins, reports):
    config = load_twin(computed_twins / "twin_03")
    private = build_report(space_of(computed_twins, BOTH, posterior=False), TwinAPI(computed_twins))
    for report in (*reports.values(), private):
        everything = b"\n".join(report.files.values())
        for forbidden in (config.description, "data_sha256", "data_file", "reproduce", "sigma_temperature",
                          "store_fields", "burn_in_max", "orx", '"seed"'):
            assert forbidden.encode() not in everything, forbidden
        for name, content in report.files.items():
            if name.endswith(".json"):
                assert str(computed_twins).encode() not in content, name  # aucun chemin
    assert not any(name.endswith(".npz") for name in private.files)


# P2c.3a/h : écriture, fichiers listés, déterminisme ----------------------------------------------------------------


def listed(index):
    return {"report.json", *(e["file"] for e in index["entries"]), *(f["file"] for f in index["figures"])}


def on_disk(folder: Path):
    return {p.relative_to(folder).as_posix() for p in folder.rglob("*") if p.is_file()}


def test_report_json_lists_exactly_the_files_of_the_folder_with_their_fingerprints(computed_twins, tmp_path):
    api = TwinAPI(computed_twins)
    report = build_report(space_of(computed_twins, BOTH), api)
    target = tmp_path / FOLDER_NAME
    write_folder(target, report.files)
    index = json.loads((target / "report.json").read_text(encoding="utf-8"))
    assert on_disk(target) == listed(index)
    for item in (*index["entries"], *index["figures"]):
        assert sha256((target / item["file"]).read_bytes()) == item["sha256"]
    assert not (tmp_path / f"{FOLDER_NAME}.partial").exists() and not (tmp_path / f"{FOLDER_NAME}.old").exists()


def test_two_generations_give_identical_bytes(computed_twins):
    api = TwinAPI(computed_twins)
    first = build_report(space_of(computed_twins, BOTH), api)
    time.sleep(2.1)  # l'horloge change : les octets, non
    second = build_report(space_of(computed_twins, BOTH), api)
    assert first.files == second.files


def test_regenerating_removes_a_foreign_file(computed_twins, tmp_path):
    report = build_report(space_of(computed_twins, FORWARD), TwinAPI(computed_twins))
    target = tmp_path / FOLDER_NAME
    write_folder(target, report.files)
    (target / "intrus.txt").write_text("à la main")
    write_folder(target, report.files)
    assert on_disk(target) == set(report.files)


def test_a_failure_during_writing_keeps_the_old_report_and_leaves_no_partial_folder(tmp_path, monkeypatch):
    target = tmp_path / FOLDER_NAME
    write_folder(target, {"report.json": b"ancien\n", "a.json": b"a\n"})
    calls = []
    real = Path.write_bytes

    def failing(self, content):
        calls.append(self.name)
        if len(calls) == 2:
            raise OSError("disque plein")
        return real(self, content)

    monkeypatch.setattr(Path, "write_bytes", failing)
    with pytest.raises(OSError, match="disque plein"):
        write_folder(target, {"report.json": b"nouveau\n", "b.json": b"b\n"})
    monkeypatch.undo()
    assert on_disk(target) == {"report.json", "a.json"} and (target / "report.json").read_bytes() == b"ancien\n"
    assert [p.name for p in tmp_path.iterdir()] == [FOLDER_NAME]


def test_generate_report_writes_the_folder_of_the_twin(computed_twins, tmp_path):
    result = generate_report(computed_twins / "twin_03", tmp_path / "out")
    assert result.folder == tmp_path / "out" and result.entries == len(json.loads((result.folder / "report.json").read_text())["entries"])
    assert result.figures == 2 and result.skipped == []
    assert on_disk(result.folder) == listed(json.loads((result.folder / "report.json").read_text()))
    index = json.loads((result.folder / "report.json").read_text())
    assert index["space"]["forward"] is None and index["space"]["inversion"]["twin"] == "twin_03"  # un twin 1.x : son seul mode


def test_dump_json_is_deterministic():
    assert dump_json({"b": 1, "a": [1.5, None]}) == b'{\n  "a": [\n    1.5,\n    null\n  ],\n  "b": 1\n}\n'


# P2c.3h : la commande ---------------------------------------------------------------------------------------


def test_command_report_in_french(computed_twins, tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("MOLONARI_LANG", "fr")
    assert cli_main(["report", str(computed_twins / "twin_01"), "--output", str(tmp_path / "rapport")]) == 0
    out = capsys.readouterr()
    assert "Rapport public écrit" in out.out and "sautées" in out.out
    assert "Sautée : soil_properties (forward)" in out.out and "forward_parameters" in out.out
    assert (tmp_path / "rapport" / "report.json").is_file()


def test_command_report_in_english_and_errors(computed_twins, tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("MOLONARI_LANG", "en")
    assert cli_main(["report", str(computed_twins / "twin_03"), "--output", str(tmp_path / "r")]) == 0
    assert "Public report written" in capsys.readouterr().out
    assert cli_main(["report", str(computed_twins / "twin_09"), "--output", str(tmp_path / "r2")]) == 1
    assert "has no results yet" in capsys.readouterr().err and not (tmp_path / "r2").exists()


# Résolution du space (twins 2.0.0, sans calcul) --------------------------------------------------------------


RAW = """schema: molonari-twin/2.0.0
name: {name}
mode: {mode}
data: {{lab: Laboratory, point: SamplingPoints/Point034}}
layers:
  - {{name: couche 1, z_low: 0.4{soil}}}
{extra}"""
SOIL = ", neg_log10_k: 12, porosity: 0.15, lambda_s: 3.4, rhos_cs: 5000000"
MCMC = "mcmc:\n  nb_chains: 7\n  nb_iter: 10\n  burn_in_max: 10\n  n_field_samples: 7\n"
REPORT = """report:
  models: {{forward: {forward}, inversion: {inversion}}}
  entries:
    - {{question: water_flux, parameters: {{level: 0.9}}}}
    - {{question: fit_quality}}
"""


@pytest.fixture(scope="module")
def data_repo(tmp_path_factory):
    data = tmp_path_factory.mktemp("data")
    shutil.copytree(FIXTURES / "lab", data / "Laboratory")
    shutil.copytree(FIXTURES / "point034", data / "SamplingPoints" / "Point034")
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("MOLONARI_DATA", str(data))
        yield data


def write_space(root: Path, forward_mode="direct", seed=4242, report=True):
    """twin_01 (direct) et twin_02 (mcmc, avec sa couche orx) ; la clé `report` est dans le twin mcmc."""
    twins = root / "twins"
    for name, mode in (("twin_01", forward_mode), ("twin_02", "mcmc")):
        folder = twins / name
        folder.mkdir(parents=True)
        extra = MCMC if mode == "mcmc" else "numerics: {nb_cells: 40}\n"
        if name == "twin_02":
            extra += "description: secret description\nsite_label: Site A\n" + (REPORT.format(forward="twin_01", inversion="twin_02") if report else "")
        (folder / "twin.yaml").write_text(RAW.format(name=name, mode=mode, soil=SOIL if mode == "direct" else "", extra=extra), encoding="utf-8")
    (twins / "twin_02" / "orx.yaml").write_text(f"mcmc:\n  seed: {seed}\n", encoding="utf-8")
    return twins


def test_space_is_resolved_from_the_report_key(tmp_path, data_repo):
    twins = write_space(tmp_path)
    space = resolve_space(twins / "twin_02")
    assert space.refs == {"forward": "twin_01", "inversion": "twin_02"} and space.declared
    assert space.carrier == "twin_02" and [e.question for e in space.entries] == ["water_flux", "fit_quality"]
    assert space.configs["inversion"].site_label == "Site A"


def test_the_twins_of_a_report_describe_the_same_point(tmp_path, data_repo, monkeypatch):
    twins = write_space(tmp_path)

    def load_other_point(folder):
        config = load_twin(folder)
        return dataclasses.replace(config, point="Point035") if Path(folder).name == "twin_01" else config

    monkeypatch.setattr("molonari_twin.report.generator.load_twin", load_other_point)
    with pytest.raises(UserError) as error:
        resolve_space(twins / "twin_02")
    assert error.value.code == "E_REPORT_POINT_MISMATCH"


def test_a_twin_without_report_key_uses_the_default_report_alone(tmp_path, data_repo):
    twins = write_space(tmp_path, report=False)
    space = resolve_space(twins / "twin_02")
    assert space.refs == {"inversion": "twin_02"} and not space.declared
    assert [e.question for e in space.entries] == ["water_flux", "fit_quality"]
    assert resolve_space(twins / "twin_01").refs == {"forward": "twin_01"}


def test_a_model_of_the_wrong_mode_is_refused(tmp_path, data_repo):
    twins = write_space(tmp_path, forward_mode="mcmc")
    with pytest.raises(UserError) as error:
        resolve_space(twins / "twin_02")
    assert error.value.code == "E_REPORT_MODEL_MODE" and error.value.params["role"] == "forward"


def test_an_unknown_model_is_refused(tmp_path, data_repo):
    twins = write_space(tmp_path)
    shutil.rmtree(twins / "twin_01")
    with pytest.raises(UserError) as error:
        resolve_space(twins / "twin_02")
    assert error.value.code == "E_TWIN_NOT_FOUND"


# De bout en bout, twins 2.0.0 calculés (slow) --------------------------------------------------------------------


@pytest.mark.slow
def test_end_to_end_two_instances_without_the_seed(tmp_path, data_repo):
    twins = write_space(tmp_path, seed=4242)
    run_twin(twins / "twin_01")
    run_twin(twins / "twin_02")
    (twins / "twin_02" / "orx.yaml").unlink()  # le dev n'a pas la couche orx
    result = generate_report(twins / "twin_02")
    folder = twins / "twin_02" / FOLDER_NAME
    assert result.folder == folder
    index = json.loads((folder / "report.json").read_text(encoding="utf-8"))
    assert on_disk(folder) == listed(index)
    assert index["space"]["forward"]["twin"] == "twin_01" and index["space"]["inversion"]["twin"] == "twin_02"
    flux = answer_from_json((folder / "inversion-water_flux.json").read_bytes())
    assert flux.data["mean"].interval is not None
    assert answer_from_json((folder / "forward-water_flux.json").read_bytes()).data["mean"].interval is None
    assert (folder / "compare.json").is_file() and (folder / "inversion-public_convergence.json").is_file()
    site = answer_from_json((folder / "inversion-describe.json").read_bytes())
    assert site.data["site_label"].value == "Site A"
    assert site.data["point"].value == "Point034" and index["point"] == "Point034"  # le rapport appartient à un point réel
    everything = b"\n".join(p.read_bytes() for p in folder.rglob("*.json")) + b"\n".join(p.read_bytes() for p in folder.rglob("*.svg"))
    for forbidden in (b"4242", b"secret description", b"seed", b"orx", str(tmp_path).encode()):
        assert forbidden not in everything, forbidden
    again = folder.parent / "again"
    generate_report(twins / "twin_02", again)
    assert {p.relative_to(again).as_posix(): p.read_bytes() for p in again.rglob("*") if p.is_file()} == \
        {p.relative_to(folder).as_posix(): p.read_bytes() for p in folder.rglob("*") if p.is_file()}
