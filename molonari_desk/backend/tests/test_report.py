"""Générateur du rapport public (P2c.3) : entrées obligatoires et par défaut, étiquettes de modèle, visibilité, index,
fichiers, déterminisme, commande et, en `slow`, un space de deux instances 2.0.0 sans la graine."""

import dataclasses
import json
import re
import shutil
import time
from pathlib import Path
from xml.etree import ElementTree

import numpy as np
import pytest
from conftest import FIXTURES
from leaks import seed_occurrences, value_occurrences
from molonari_twin.api import TwinAPI
from molonari_twin.contract.codec import answer_from_json, answer_from_npz
from molonari_twin.errors import MESSAGES, UserError
from molonari_twin.i18n import load_catalog
from molonari_twin.questions import Visibility
from molonari_twin.report import (
    FOLDER_NAME,
    REPORT_FORMAT,
    Space,
    build_report,
    generate_report,
    resolve_space,
)
from molonari_twin.report.figures import build_figures
from molonari_twin.report.generator import decode_entries, effective_visibility, plan
from molonari_twin.report.interpretation import damping, findings
from molonari_twin.report.readme import readme
from molonari_twin.report.writer import dump_json, encode_entry, sha256, write_folder
from molonari_twin.twin.cli import main as cli_main
from molonari_twin.twin.config import (
    ReportEntry,
    TwinConfigError,
    default_report,
    load_twin,
)
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


FIGURES_BOTH = {
    "figures/water_flux.svg": ["forward-water_flux.json", "inversion-water_flux.json"],
    "figures/temperature_field-forward.svg": ["forward-fields.npz", "forward-describe.json"],
    "figures/temperature_field-inversion.svg": ["inversion-fields.npz", "inversion-describe.json"],
    "figures/temperature_depths-forward.svg": ["forward-fields.npz"],
    "figures/temperature_depths-inversion.svg": ["inversion-fields.npz"],
    "figures/water_flux_field-forward.svg": ["forward-fields.npz"],
    "figures/water_flux_field-inversion.svg": ["inversion-fields.npz"],
    "figures/posterior.svg": ["inversion-posterior_samples.npz", "inversion-soil_properties.json"],
    "figures/chains.svg": ["chains.npz", "inversion-public_convergence.json"],
    "figures/fit_quality.svg": ["forward-fit_quality.json", "inversion-fit_quality.json"],
    "figures/compare.svg": ["compare.json"],
}
"""Les onze figures d'un rapport à deux modèles dont les états sont publics, et leurs sources (F1)."""


@pytest.fixture(scope="module")
def with_states(computed_twins):
    """Les mêmes rapports, états publics : les champs T et q sont publiés (F1)."""
    api = TwinAPI(computed_twins)
    return {name: build_report(space_of(computed_twins, refs, states=True), api) for name, refs in
            (("both", BOTH), ("forward", FORWARD), ("inversion", INVERSION))}


def test_figures_are_derived_from_entries_with_their_fingerprint(with_states):
    report = with_states["both"]
    figures = report.index["figures"]
    assert [f["file"] for f in figures] == list(FIGURES_BOTH)
    for figure in figures:
        sources = [figure["from_entry"]] if "from_entry" in figure else figure["from_entries"]
        assert ("from_entry" in figure) != ("from_entries" in figure)
        assert sources == FIGURES_BOTH[figure["file"]] and all(source in report.files for source in sources)
        assert figure["sha256"] == sha256(report.files[figure["file"]])
        assert report.files[figure["file"]].startswith(b"<svg")
    assert b"forward model \xe2\x80\x94 no interval" in report.files["figures/water_flux.svg"]
    assert b"90 % posterior interval" in report.files["figures/water_flux.svg"]


# P2c.3b : index ---------------------------------------------------------------------------------------------


def test_index_fields(reports, computed_twins):
    report = reports["both"]
    index = report.index
    assert index["report_format"] == REPORT_FORMAT == "1.1.0"
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
                            ("chain_data", "inversion"), ("fields", "forward"), ("fields", "inversion")}  # les états sont privés par défaut
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
        ("forward", "fields"), ("forward", "soil_properties"), ("inversion", "fields"), ("inversion", "soil_properties"),
        ("inversion", "posterior_samples"), ("inversion", "chain_data")]
    assert [i.required for i in items] == [True] * 10 + [False] * 6
    assert [i.parameters for i in items if i.question == "fields"] == [{"variables": ["temperature", "water_flux"], "time_step": 4}] * 2


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


def test_nothing_private_or_seed_related_is_written(computed_twins, reports, with_states):
    config = load_twin(computed_twins / "twin_03")
    private = build_report(space_of(computed_twins, BOTH, posterior=False), TwinAPI(computed_twins))
    for report in (*reports.values(), private, *with_states.values()):
        everything = b"\n".join(report.files.values())
        for forbidden in (config.description, "data_sha256", "data_file", "reproduce", "sigma_temperature",
                          "store_fields", "burn_in_max", "orx", '"seed"'):
            assert forbidden.encode() not in everything, forbidden
        for name, content in report.files.items():
            if not name.endswith(".npz"):  # JSON, SVG, README : aucun chemin
                assert str(computed_twins).encode() not in content, name
    assert not any(name.endswith(".npz") for name in private.files)


# P2c.3a/h : écriture, fichiers listés, déterminisme ----------------------------------------------------------------


def listed(index):
    files = {"report.json", index["readme"]["file"], *(e["file"] for e in index["entries"]), *(f["file"] for f in index["figures"])}
    if index.get("chains") is not None:
        files.add(index["chains"]["file"])
    return files


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
    assert result.figures == 4 and [s["question"] for s in result.skipped] == ["fields"]  # twin_03 seul : flux, RMSE, PDF (posterior, chaînes) ; états privés
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

SEED = 731942  # graine distinctive (déjà celle de T10b)
SIGMA = 0.073129  # σ distinctif du twin mcmc
PRIORS = {"neg_log10_k": [10.8713, 14.6329], "porosity": [0.013727, 0.296381], "lambda_s": [1.17329, 4.87113], "rhos_cs": [1137291, 9871137]}
"""Bornes a priori distinctives (au moins quatre chiffres significatifs hors des deux décimales des coordonnées SVG)."""
DISTINCTIVE = [SEED, SIGMA, *(bound for bounds in PRIORS.values() for bound in bounds)]


@pytest.fixture(scope="module")
def data_repo(tmp_path_factory):
    data = tmp_path_factory.mktemp("data")
    shutil.copytree(FIXTURES / "lab", data / "Laboratory")
    shutil.copytree(FIXTURES / "point034", data / "SamplingPoints" / "Point034")
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("MOLONARI_DATA", str(data))
        yield data


def write_space(root: Path, forward_mode="direct", seed=4242, report=True, distinctive=False, states=False, fields_time_step=None):
    """twin_01 (direct) et twin_02 (mcmc, avec sa couche orx) ; la clé `report` est dans le twin mcmc.
    `distinctive` : graine, σ et bornes a priori distinctifs (test de fuite) ; `states` : `visibility: {states: true}` dans les
    deux twins ; `fields_time_step` : réglage des champs publics."""
    twins = root / "twins"
    seed = SEED if distinctive else seed
    for name, mode in (("twin_01", forward_mode), ("twin_02", "mcmc")):
        folder = twins / name
        folder.mkdir(parents=True)
        extra = MCMC if mode == "mcmc" else "numerics: {nb_cells: 40}\n"
        if mode == "mcmc" and distinctive:
            extra += "  priors:\n  - " + "\n    ".join(f"{key}: {value}" for key, value in PRIORS.items()) + "\n"
            extra += f"sensors: {{sigma_temperature: {SIGMA}}}\n"
        if states:
            extra += "visibility: {states: true}\n"
        if name == "twin_02":
            extra += "description: secret description\nsite_label: Site A\n" + (REPORT.format(forward="twin_01", inversion="twin_02") if report else "")
            if report and fields_time_step is not None:
                extra += f"  fields_time_step: {fields_time_step}\n"
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


def folder_texts(folder: Path):
    """(nom, texte) de chaque fichier du rapport : les textes tels quels, et pour un npz son `meta` et ses tableaux de chaînes."""
    for path in sorted(folder.rglob("*")):
        if not path.is_file():
            continue
        name = path.relative_to(folder).as_posix()
        if path.suffix != ".npz":
            yield name, path.read_text(encoding="utf-8")
            continue
        with np.load(path, allow_pickle=False) as archive:
            for key in archive.files:
                yield f"{name}[{key}]", key
                if archive[key].dtype.kind in "US":
                    yield f"{name}[{key}]", "\n".join(str(item) for item in archive[key].ravel())


def assert_no_measured_series(folder: Path, measurements, step: int):
    """Aucun tableau numérique d'un npz publié n'est une série mesurée (ni sous-échantillonnée au pas des champs)."""
    series = {name: measurements[name].to_numpy(float) for name in ("temp_bed", "t1", "t2", "t3", "t4", "pressure")}
    checked = 0
    for path in sorted(folder.rglob("*.npz")):
        with np.load(path, allow_pickle=False) as archive:
            for key in archive.files:
                array = archive[key]
                if array.dtype.kind not in "fiu" or array.ndim == 0:
                    continue
                for axis, size in enumerate(array.shape):
                    rows = np.moveaxis(array, axis, -1).reshape(-1, size)
                    for name, values in series.items():
                        for candidate in (values, values[::step]):
                            if len(candidate) == size:
                                checked += 1
                                assert not any(np.array_equal(row, candidate) for row in rows), f"{path.name}[{key}] contient la série {name}"
    assert checked  # les tailles comparables existent bien (les champs sont publiés)


@pytest.mark.slow
def test_end_to_end_two_instances_without_the_seed(tmp_path, data_repo):
    twins = write_space(tmp_path, distinctive=True, states=True)
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
    # les états sont publics : les champs, les figures qui en dérivent et la page lisible sont là (un mcmc sans `store_fields:
    # full` donne les champs du meilleur ajustement, relus de temperature.csv et water_flux_field.csv)
    for role in ("forward", "inversion"):
        fields = answer_from_npz((folder / f"{role}-fields.npz").read_bytes())
        assert set(fields.data) == {"depth", "time", "temperature", "water_flux"}
    assert (folder / "README.md").is_file() and (folder / "figures" / "temperature_field-inversion.svg").is_file()
    # aucune valeur distinctive (graine, σ, huit bornes a priori) ni interdit dans aucun fichier : JSON, SVG, README, npz et leurs meta
    for value in DISTINCTIVE:
        assert value_occurrences(folder, value) == [], value
    assert seed_occurrences(folder, SEED) == []
    for name, text in folder_texts(folder):
        for forbidden in ("secret description", "orx", "seed", str(tmp_path)):
            assert forbidden not in text, (name, forbidden)
    config = load_twin(twins / "twin_02")
    assert_no_measured_series(folder, config.load_measurements(), config.report.fields_time_step)
    again = folder.parent / "again"
    generate_report(twins / "twin_02", again)
    assert {p.relative_to(again).as_posix(): p.read_bytes() for p in again.rglob("*") if p.is_file()} == \
        {p.relative_to(folder).as_posix(): p.read_bytes() for p in folder.rglob("*") if p.is_file()}


# chains.npz (T2) : sous-échantillon public des chaînes -------------------------------------------------------


def _load_chains(report):
    """Charge chains.npz du rapport en dict de ndarrays, sans pickle."""
    import io
    buf = io.BytesIO(report.files["chains.npz"])
    with np.load(buf, allow_pickle=False) as data:
        return {k: data[k] for k in data.files}


@pytest.mark.slow
def test_chains_npz_is_generated_and_loadable_without_pickle(computed_twins):
    """chains.npz est dans le rapport, chargeable sans pickle, et contient les bonnes clés."""
    api = TwinAPI(computed_twins)
    space = space_of(computed_twins, INVERSION)
    report = build_report(space, api)
    assert "chains.npz" in report.files
    chains = _load_chains(report)
    expected = {"iteration", "chain", "energy", "acceptance", "resets",
                "samples.neg_log10_k", "samples.porosity", "samples.lambda_s", "samples.rhos_cs"}
    assert expected <= set(chains.keys())


@pytest.mark.slow
def test_chains_npz_no_leak(computed_twins):
    """Aucune clé ne contient seed, T, temperature ni flux ; aucun chemin de la machine n'apparaît."""
    api = TwinAPI(computed_twins)
    report = build_report(space_of(computed_twins, INVERSION), api)
    chains = _load_chains(report)
    for forbidden in ("seed", "temperature", "flux"):
        assert not any(forbidden in k for k in chains), f"key containing {forbidden!r} found"
    # Aucun chemin n'apparaît dans les valeurs stringifiées
    raw = report.files["chains.npz"]
    import os
    assert os.path.expanduser("~").encode() not in raw


@pytest.mark.slow
def test_chains_npz_derives_from_posterior(computed_twins):
    """chains.npz dérive exactement de posterior.npz (mêmes échantillons, sous-échantillonnés)."""
    from molonari_twin.twin.run import twin_status
    api = TwinAPI(computed_twins)
    space = space_of(computed_twins, INVERSION)
    report = build_report(space, api)
    chains = _load_chains(report)
    status = twin_status(computed_twins / "twin_03")
    with np.load(status.results_dir / "posterior.npz") as post:
        pub_thin = space.publication_thin
        config = load_twin(computed_twins / "twin_03")
        ratio = pub_thin // config.mcmc.thin
        rows = np.arange(post["samples"].shape[0])[ratio - 1 :: ratio]
        np.testing.assert_array_equal(chains["energy"], post["energy"][rows])
        for k in ("neg_log10_k", "porosity", "lambda_s", "rhos_cs"):
            idx = ("neg_log10_k", "porosity", "lambda_s", "rhos_cs").index(k)
            np.testing.assert_array_equal(chains[f"samples.{k}"], post["samples"][rows][..., idx])


@pytest.mark.slow
def test_chains_npz_report_cites_posterior_sha256(computed_twins):
    """L'index du rapport cite le SHA-256 du posterior.npz et publication_thin."""
    api = TwinAPI(computed_twins)
    report = build_report(space_of(computed_twins, INVERSION), api)
    chains_info = report.index.get("chains")
    assert chains_info is not None
    assert chains_info["publication_thin"] == 10
    assert chains_info["posterior_sha256"] is not None
    assert len(chains_info["posterior_sha256"]) == 64  # hex sha256


@pytest.mark.slow
def test_publication_thin_not_multiple_of_thin_is_rejected(tmp_path):
    """publication_thin qui n'est pas un multiple du thin de calcul donne une erreur."""
    yaml_text = """schema: molonari-twin/2.0.0
name: twin_x
mode: mcmc
data: {lab: Laboratory, point: SamplingPoints/Point034}
layers:
  - {name: couche, z_low: 0.4}
mcmc:
  nb_chains: 7
  nb_iter: 30
  thin: 3
  burn_in_max: 10
  n_field_samples: 7
report:
  publication_thin: 7
  entries:
    - {question: water_flux, parameters: {level: 0.9}}
    - {question: fit_quality}
"""
    from molonari_twin.twin.config import TwinConfigError, load_twin
    folder = tmp_path / "twin_x"
    folder.mkdir()
    (folder / "twin.yaml").write_text(yaml_text, encoding="utf-8")
    with pytest.raises(TwinConfigError) as exc_info:
        load_twin(folder, check_data=False)
    assert any(i.code == "E_TWIN_PUBLICATION_THIN_MULTIPLE" for i in exc_info.value.issues)


@pytest.mark.slow
def test_publication_thin_does_not_change_twin_id(computed_twins):
    """Changer publication_thin ne change pas le twin_id (hors canonical)."""
    config = load_twin(computed_twins / "twin_03")
    id_a = config.twin_id("test-version")
    config2 = dataclasses.replace(config, report=dataclasses.replace(config.report, publication_thin=99))
    id_b = config2.twin_id("test-version")
    assert id_a == id_b


# F1 : champs publics, figures, constats, page lisible ------------------------------------------------------------


def test_fields_are_published_only_when_states_are_public(reports, with_states):
    public = with_states["both"]
    assert {"forward-fields.npz", "inversion-fields.npz"} <= set(public.files)
    assert [e["parameters"] for e in entries_of(public, question="fields")] == [{"variables": ["temperature", "water_flux"], "time_step": 4}] * 2
    private = reports["both"]
    assert [(s["question"], s["model"], s["code"]) for s in private.index["skipped"] if s["question"] == "fields"] == \
        [("fields", "forward", "E_REPORT_PRIVATE"), ("fields", "inversion", "E_REPORT_PRIVATE")]
    assert "states" in next(s for s in private.index["skipped"] if s["question"] == "fields")["params"]["elements"]
    assert not any("fields" in name for name in private.files)


def test_fields_time_step_subsamples_and_is_npz(computed_twins):
    api = TwinAPI(computed_twins)
    space = space_of(computed_twins, INVERSION, states=True)
    every_date = build_report(dataclasses.replace(space, fields_time_step=1), api, npz_threshold=1 << 40)
    fourth = build_report(space, api, npz_threshold=1 << 40)  # seuil énorme : l'entrée reste en npz
    entry = entries_of(fourth, question="fields")[0]
    assert entry["format"] == "npz" and entry["file"] == "inversion-fields.npz"
    full, thin = answer(every_date, "inversion-fields.npz"), answer(fourth, "inversion-fields.npz")
    count = len(full.data["time"].value)
    assert len(thin.data["time"].value) == -(-count // 4) and thin.data["temperature"].value.shape == (len(thin.data["depth"].value), -(-count // 4))
    np.testing.assert_array_equal(thin.data["temperature"].value, full.data["temperature"].value[:, ::4])
    assert thin.provenance["parameters"]["time_step"] == 4 and full.provenance["parameters"]["time_step"] == 1


def test_public_fields_have_only_depth_time_temperature_and_water_flux(with_states):
    for role in ("forward", "inversion"):
        fields = answer(with_states["both"], f"{role}-fields.npz")
        assert set(fields.data) == {"depth", "time", "temperature", "water_flux"}
        assert not any(name.startswith("heat_flux") for name in fields.data)
        assert set(fields.provenance) <= {"question", "ref", "parameters", "instance"} and "sigma" not in fields.provenance["parameters"]
        assert set(fields.provenance["parameters"]) == {"variables", "time_step"}  # ni période, ni plage de profondeurs, ni chemin


def test_no_field_anywhere_when_states_are_private(reports):
    report = reports["both"]
    assert not [name for name in report.files if name.endswith("-fields.npz") or re.search(r"temperature_field|temperature_depths|water_flux_field", name)]
    assert not [f for f in report.index["findings"] if f["code"] in ("I_FLUX_INFILTRATION_SHARE", "I_DAILY_DAMPING")]
    assert not [f for f in report.index["figures"] if "fields" in json.dumps(f)]
    text = report.files["README.md"].decode()
    assert "Limites" in text and "ne sont pas publiés : le développeur garde les états privés" in text
    assert "temperature_field" not in text and "amortissement" not in text


def test_expected_figures_and_their_sources(with_states):
    both = with_states["both"]
    assert len(both.index["figures"]) == 11  # 1, 2 × 2, 3 × 2, 4 × 2, 5, 6, 7, 8
    assert {f["file"]: f.get("from_entries") or [f["from_entry"]] for f in both.index["figures"]} == FIGURES_BOTH
    assert b"best-fit" in both.files["figures/temperature_field-inversion.svg"]  # le champ de l'inversion est celui du meilleur ajustement
    assert b"best-fit" in both.files["figures/water_flux_field-inversion.svg"]
    forward = with_states["forward"]
    assert [f["file"] for f in forward.index["figures"]] == ["figures/water_flux.svg", "figures/temperature_field-forward.svg",
                                                             "figures/temperature_depths-forward.svg", "figures/water_flux_field-forward.svg",
                                                             "figures/fit_quality.svg"]
    assert forward.index["figures"][0]["from_entry"] == "forward-water_flux.json"
    inversion = with_states["inversion"]
    assert "figures/compare.svg" not in inversion.files and "figures/posterior.svg" in inversion.files
    assert "figures/chains.svg" in inversion.files


def read_folder(folder: Path) -> dict[str, bytes]:
    return {p.relative_to(folder).as_posix(): p.read_bytes() for p in folder.rglob("*") if p.is_file()}


def test_figures_are_rebuilt_from_the_published_files_alone(with_states, tmp_path):
    for name in ("both", "forward", "inversion"):
        target = tmp_path / name
        write_folder(target, with_states[name].files)
        files = read_folder(target)
        index = json.loads(files["report.json"])
        published, chains = decode_entries(files, index["entries"])  # relus par le codec, comme le fait un lecteur
        rebuilt = build_figures(published, chains)
        assert [path for path, _, _ in rebuilt] == [f["file"] for f in index["figures"]]
        for path, content, sources in rebuilt:
            assert files[path] == content and all(source in files for source in sources), path
        found = findings(published, chains)
        assert found == index["findings"]
        assert readme(index, found, published) == files["README.md"]


def svg_texts(report):
    for name, content in report.files.items():
        if name.endswith(".svg"):
            root = ElementTree.fromstring(content)
            yield name, root, ["".join(t.itertext()) for t in root.iter("{http://www.w3.org/2000/svg}text")]


def test_svg_root_has_viewbox_title_and_desc(with_states):
    for name, root, _ in svg_texts(with_states["both"]):
        assert re.fullmatch(r"0 0 \d+\.\d\d \d+\.\d\d", root.get("viewBox")), name
        assert root.get("width") and root.get("height") is None, name
        assert root[0].tag.endswith("title") and root[0].text and root[1].tag.endswith("desc") and len(root[1].text) > 40, name


def text_extents(root):
    """(texte, gauche, droite) estimés de chaque `<text>` d'un SVG, dans le repère de la racine : largeur = nombre de
    caractères × CHAR_WIDTH (×13/12 pour un titre), ancrage start / middle / end, `translate` des panneaux cumulés."""
    from molonari_twin.report.svgplot import CHAR_WIDTH

    svg = "{http://www.w3.org/2000/svg}"

    def walk(node, dx):
        if node.tag == f"{svg}g" and node.get("transform", "").startswith("translate("):
            dx += float(node.get("transform")[len("translate("):-1].split()[0])
        if node.tag == f"{svg}text":
            text = "".join(node.itertext())
            small = sum(len(t.text or "") for t in node.iter(f"{svg}tspan"))  # exposants : 9 px au lieu de 12
            width = (len(text) - small * 0.25) * CHAR_WIDTH * (13 / 12 if node.get("font-weight") == "bold" else 1)
            x = float(node.get("x")) + dx
            anchor = node.get("text-anchor", "start")
            left = {"start": x, "middle": x - width / 2, "end": x - width}[anchor]
            yield text, left, left + width
        for child in node:
            yield from walk(child, dx)

    return list(walk(root, 0.0))


def test_no_text_overflows_the_viewbox_of_any_figure(with_states):
    for name, root, _ in svg_texts(with_states["both"]):
        width = float(root.get("viewBox").split()[2])
        for text, left, right in text_extents(root):
            assert left >= -1 and right <= width + 1, f"{name}: « {text} » déborde ({left:.0f} à {right:.0f} sur {width:.0f})"
    compare = with_states["both"].files["figures/compare.svg"].decode()
    assert "total RMSE" in compare and f"{float(answer(with_states['both'], 'compare.json').data['rmse_total'].value):.3f}" in compare


def test_no_scientific_tick_label_in_any_figure(with_states):
    for name, _, texts in svg_texts(with_states["both"]):
        assert not [t for t in texts if re.fullmatch(r"-?\d\.\d+e[-+]\d+", t)], name
        assert not [t for t in texts if re.search(r"\d[eE][-+]\d", t)], name
    flux = with_states["both"].files["figures/water_flux.svg"].decode()
    assert "\u00d7 10<tspan" in flux and "m\u00b7s<tspan" in flux  # le facteur commun et l'unité, en tspan


def test_findings_codes_are_declared_and_translated(with_states):
    import string

    catalog = load_catalog("fr")
    codes = set()
    for finding in with_states["both"].index["findings"]:
        template = MESSAGES[finding["code"]]
        assert re.fullmatch(r"I_[A-Z0-9_]+", finding["code"]) and template in catalog, finding["code"]
        wanted = {name for _, name, _, _ in string.Formatter().parse(template) if name}
        assert wanted <= set(finding["params"]), finding["code"]
        assert finding["figure"] in with_states["both"].files
        codes.add(finding["code"])
    assert codes >= {"I_FLUX_MEAN_EXFILTRATION", "I_FLUX_INTERVAL_EXCLUDES_ZERO", "I_FLUX_INFILTRATION_SHARE", "I_DAILY_DAMPING",
                     "I_MODEL_DIFFERENCE", "I_FIT_RMSE", "I_CONVERGENCE"}


def test_findings_values(with_states, computed_twins):
    report = with_states["both"]
    by_code: dict[str, list[dict]] = {}
    for finding in report.index["findings"]:
        by_code.setdefault(finding["code"], []).append(finding)
    for finding in (*by_code.get("I_FLUX_MEAN_INFILTRATION", []), *by_code.get("I_FLUX_MEAN_EXFILTRATION", [])):
        model = finding["params"]["model"]
        mean = float(answer(report, f"{model}-water_flux.json").data["mean"].value)
        assert (finding["code"] == "I_FLUX_MEAN_INFILTRATION") == (mean >= 0)
        assert finding["params"]["mean"] == float(f"{mean:.4g}") and finding["params"]["exponent"] == int(np.floor(np.log10(abs(mean))))
    (interval,) = [*by_code.get("I_FLUX_INTERVAL_EXCLUDES_ZERO", []), *by_code.get("I_FLUX_INTERVAL_CONTAINS_ZERO", [])]
    low, high = answer(report, "inversion-water_flux.json").data["mean"].interval
    assert (interval["code"] == "I_FLUX_INTERVAL_CONTAINS_ZERO") == (low <= 0 <= high) and interval["params"]["level"] == 90.0
    for finding in by_code["I_FIT_RMSE"]:
        total = float(answer(report, f"{finding['params']['model']}-fit_quality.json").data["rmse_total"].value)
        assert finding["params"]["rmse"] == float(f"{total:.4g}")
    (compare,) = by_code["I_MODEL_DIFFERENCE"]
    assert compare["params"]["rmse"] == float(f"{float(answer(report, 'compare.json').data['rmse_total'].value):.4g}")
    (convergence,) = by_code["I_CONVERGENCE"]
    assert convergence["params"]["chains"] == 8 and convergence["params"]["burn_in"] == 20
    shares = {f["params"]["model"]: f["params"] for f in by_code["I_FLUX_INFILTRATION_SHARE"]}
    assert set(shares) == {"forward", "inversion"} and all(0 <= p["share"] <= 100 and p["step"] == 1.0 for p in shares.values())
    for role, share in shares.items():  # part du temps où q >= 0 au fond, pondérée par les intervalles de temps
        fields = answer(report, f"{role}-fields.npz")
        times = fields.data["time"].value.astype("datetime64[s]").astype(np.int64).astype(float)
        weights = np.diff(times, prepend=times[0], append=times[-1])  # demi-intervalles de chaque côté (trapèzes), comme flux.py
        weights = (weights[:-1] + weights[1:]) / 2
        first = fields.data["water_flux"].value[0]
        assert share["share"] == pytest.approx(100 * weights[first >= 0].sum() / weights.sum(), abs=0.01)
        assert share["depth"] == float(f"{fields.data['depth'].value[0]:.4g}")


def test_a_finding_with_a_non_finite_number_is_left_out_and_the_page_is_still_written(with_states):
    from molonari_twin.questions import Answer, Quantity

    published, chains = decode_entries(with_states["both"].files, with_states["both"].index["entries"])
    convergence = published["inversion-public_convergence.json"]
    broken = {**convergence.data, "rhat_max": Quantity(float("inf"), "1")}  # chaînes trop courtes : R̂ infini
    published["inversion-public_convergence.json"] = Answer(broken, convergence.provenance)
    found = findings(published, chains)
    assert "I_CONVERGENCE" not in [f["code"] for f in found] and "I_FIT_RMSE" in [f["code"] for f in found]
    dump_json({"findings": found})  # JSON sans NaN ni infini
    chains_svg = {path: content for path, content, _ in build_figures(published, chains)}["figures/chains.svg"]
    assert "max = inf" in chains_svg.decode()  # la figure écrit la valeur
    assert readme(with_states["both"].index, found, published)


def test_daily_damping_recovers_a_synthetic_wave():
    seconds = np.arange(0, 10 * 86400, 3600.0)
    omega = 2 * np.pi / 86400
    depth = np.array([0.1, 0.25, 0.4])
    top = 15 + 2.0 * np.cos(omega * (seconds - 14 * 3600)) + 0.3 * seconds / 86400
    bottom = 14 + 0.5 * np.cos(omega * (seconds - 20 * 3600)) + 0.1 * seconds / 86400  # 6 h plus tard, quatre fois plus faible
    middle = (top + bottom) / 2
    result = damping(seconds, depth, np.stack([top, middle, bottom]))
    assert result["amplitude_top"] == pytest.approx(2.0, abs=1e-6) and result["amplitude_bottom"] == pytest.approx(0.5, abs=1e-6)
    assert result["ratio"] == pytest.approx(0.25, abs=1e-6) and result["lag"] == pytest.approx(6.0, abs=1e-6)
    assert result["depth_bottom"] == 0.4
    later = damping(seconds, depth, np.stack([top, middle, 14 + 0.5 * np.cos(omega * (seconds - 10 * 3600))]))
    assert later["lag"] == pytest.approx(20.0, abs=1e-6)  # ramené dans [0, 24[


def test_daily_damping_is_left_out_when_fields_are_too_far_apart(computed_twins):
    report = build_report(dataclasses.replace(space_of(computed_twins, BOTH, states=True), fields_time_step=24), TwinAPI(computed_twins))
    assert not [f for f in report.index["findings"] if f["code"] == "I_DAILY_DAMPING"]
    assert [f["code"] for f in report.index["findings"]].count("I_FLUX_INFILTRATION_SHARE") == 2
    assert "n'est pas calculé" in report.files["README.md"].decode()  # le README le dit


def test_report_never_reads_or_shows_thresholds(with_states):
    source = Path(__file__).resolve().parents[1] / "src" / "molonari_twin" / "report"
    for name in ("figures.py", "interpretation.py", "readme.py"):
        text = (source / name).read_text(encoding="utf-8")
        for forbidden in ("rhat_green", "rhat_orange", "ess_green", "ess_orange", "thresholds_validated", "_identifiability"):
            assert forbidden not in text, (name, forbidden)
    for report in with_states.values():
        page = report.files["README.md"].decode().lower()
        for forbidden in ("seuil", "bien contraint", "mal contraint", "accord", "identifiab"):
            assert forbidden not in page, forbidden
        for name, _, texts in svg_texts(report):
            assert not [t for t in texts if re.search(r"threshold|identifiab|well.constrained|agree", t, re.IGNORECASE)], name


def test_readme_is_french_lists_figures_and_limits(with_states, tmp_path):
    report = with_states["both"]
    page = report.files["README.md"].decode()
    assert report.index["readme"] == {"file": "README.md", "sha256": sha256(report.files["README.md"])}
    assert page.startswith("# Rapport public \u2014 Point034")
    write_folder(tmp_path / "out", report.files)
    for figure in report.index["figures"]:
        assert f"]({figure['file']})" in page and (tmp_path / "out" / figure["file"]).is_file()
    assert "## Limites" in page and "Le modèle direct n'a pas d'incertitude" in page and "Les mesures ne sont pas publiées" in page
    assert "Les champs sont ceux du meilleur ajustement" in page
    french = load_catalog("fr")[MESSAGES["I_FIT_RMSE"]]
    assert french.split("{model}")[0] in page and "RMSE totale" in page
    assert str(tmp_path) not in page and "/Users" not in page


def test_report_keys_and_visibility_do_not_change_twin_id(tmp_path, data_repo):
    def twin_id(root, **options):
        twins = write_space(root, **options)
        return load_twin(twins / "twin_02").twin_id("test-version"), load_twin(twins / "twin_01").twin_id("test-version")

    base = twin_id(tmp_path / "base")
    assert twin_id(tmp_path / "states", states=True) == base
    assert twin_id(tmp_path / "step", fields_time_step=8) == base
    assert twin_id(tmp_path / "both", states=True, fields_time_step=8) == base
    assert load_twin(tmp_path / "both" / "twins" / "twin_02").report.fields_time_step == 8
    assert load_twin(tmp_path / "base" / "twins" / "twin_02").report.fields_time_step == 4
    assert load_twin(tmp_path / "states" / "twins" / "twin_02").visibility["states"] is True


@pytest.mark.parametrize(("value", "code"), [("0", "E_VALUE_OUT_OF_RANGE"), ("97", "E_VALUE_OUT_OF_RANGE"), ('"x"', "E_TWIN_EXPECTED_INTEGER"),
                                             ("2.5", "E_TWIN_EXPECTED_INTEGER")])
def test_fields_time_step_is_validated(tmp_path, value, code):
    text = f"""schema: molonari-twin/2.0.0
name: twin_x
mode: direct
data: {{lab: Laboratory, point: SamplingPoints/Point034}}
layers:
  - {{name: couche, z_low: 0.4, neg_log10_k: 12, porosity: 0.15, lambda_s: 3.4, rhos_cs: 5000000}}
report:
  entries:
    - {{question: water_flux, parameters: {{level: 0.9}}}}
    - {{question: fit_quality}}
  fields_time_step: {value}
"""
    (tmp_path / "twin.yaml").write_text(text, encoding="utf-8")
    with pytest.raises(TwinConfigError) as error:
        load_twin(tmp_path, check_data=False)
    assert [i.code for i in error.value.issues if i.field == "report.fields_time_step"] == [code]


# Détecteur de fuite : il trouve ce qu'on y met ---------------------------------------------------------------------


def test_leak_detector_finds_injected_values(tmp_path):
    def folder(name):
        target = tmp_path / name
        (target / "figures").mkdir(parents=True)
        (target / "figures" / "a.svg").write_text('<svg><text x="12.34" y="56.78">142.42 and 7319420</text></svg>', encoding="utf-8")
        (target / "README.md").write_text("# Rapport\n\nvaleur 0,5 et 10,87 m\n", encoding="utf-8")
        (target / "report.json").write_text('{"a": 1.5}\n', encoding="utf-8")
        np.savez(target / "x.npz", depth=np.linspace(0, 1, 5), meta=np.array('{"a": 1}'))
        return target

    values = [SEED, SIGMA, 10.8713]
    clean = folder("clean")
    assert all(value_occurrences(clean, value) == [] for value in values) and seed_occurrences(clean, SEED) == []
    cases = {
        "graine dans un <text>": (lambda f: _inject(f / "figures" / "a.svg", '<text>seed value 731942</text>'), SEED),
        "graine en flottant dans le README": (lambda f: _inject(f / "README.md", "731942.0\n"), SEED),
        "graine en notation scientifique": (lambda f: _inject(f / "README.md", "7.31942E+05\n"), SEED),
        "σ dans le README": (lambda f: _inject(f / "README.md", "sigma 0.073129\n"), SIGMA),
        "σ dans un <text>": (lambda f: _inject(f / "figures" / "a.svg", "<text>0.073129</text>"), SIGMA),
        "borne dans un <text>": (lambda f: _inject(f / "figures" / "a.svg", "<text>10.8713</text>"), 10.8713),
        "borne dans le JSON": (lambda f: _inject(f / "report.json", '{"prior": 10.8713}\n'), 10.8713),
        "σ dans un tableau npz": (lambda f: np.savez(f / "y.npz", values=np.array([0.1, SIGMA, 0.3])), SIGMA),
        "borne dans un tableau npz": (lambda f: np.savez(f / "y.npz", values=np.array([[1.0, 10.8713]])), 10.8713),
        "graine dans un tableau entier": (lambda f: np.savez(f / "y.npz", values=np.array([SEED])), SEED),
    }
    for i, (case, (inject, value)) in enumerate(cases.items()):
        target = folder(f"case{i}")
        inject(target)
        assert value_occurrences(target, value), f"valeur non vue : {case}"
    for i, neighbour in enumerate(("0,073129", "0.0731290001", "x0.073129", "a0.073129b", "0.0731")):  # pas la valeur : pas de fuite
        target = folder(f"neg{i}")
        _inject(target / "README.md", f"{neighbour}\n")
        assert value_occurrences(target, SIGMA) == [], neighbour
    measured = np.sin(np.arange(40.0))
    target = folder("measured")
    np.savez(target / "z.npz", temperature=np.stack([measured, measured + 1]))
    with pytest.raises(AssertionError):
        assert_no_measured_series(target, {"temp_bed": _Series(measured), "t1": _Series(measured), "t2": _Series(measured),
                                           "t3": _Series(measured), "t4": _Series(measured), "pressure": _Series(measured)}, 1)
    clean_target = folder("unmeasured")
    np.savez(clean_target / "z.npz", temperature=np.stack([measured + 0.5, measured + 1]))
    assert_no_measured_series(clean_target, {name: _Series(measured) for name in ("temp_bed", "t1", "t2", "t3", "t4", "pressure")}, 1)


def _inject(path: Path, text: str) -> None:
    path.write_text(path.read_text(encoding="utf-8") + text, encoding="utf-8")


class _Series:
    """Une colonne de mesures réduite à `to_numpy` (la forme que `assert_no_measured_series` lit d'un DataFrame)."""

    def __init__(self, values):
        self.values = values

    def to_numpy(self, dtype=float):
        return np.asarray(self.values, dtype)
