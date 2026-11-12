"""`molonari-twin plot temperature|fields` (F2) : les figures de l'admin, aux niveaux dev et opx, hors des publications."""

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from xml.etree import ElementTree

import numpy as np
import pytest
from conftest import EXAMPLE_TWINS
from molonari_twin.api import TwinAPI
from molonari_twin.errors import MESSAGES
from molonari_twin.report import plots
from molonari_twin.report.plots import ADMIN_MAX_POINTS
from molonari_twin.twin.cli import main
from molonari_twin.twin.run import run_twin
from svgcheck import parse, text_extents

SVG = "{http://www.w3.org/2000/svg}"


@pytest.fixture
def run(capsys, monkeypatch):
    """run(*args, lang="en") → (code de sortie, sortie standard, sortie d'erreur) de la ligne de commande, dans ce processus."""

    def call(*args, lang="en"):
        monkeypatch.setenv("MOLONARI_LANG", lang)
        code = main([str(a) for a in args])
        captured = capsys.readouterr()
        return code, captured.out, captured.err

    return call


@pytest.fixture
def plot(run, computed_twins, tmp_path):
    """plot(action, twin, *options) → (code, sortie standard, sortie d'erreur, dossier de sortie) au niveau dev, dans tmp_path."""

    calls = []

    def call(action, twin, *options, level="dev"):
        calls.append(None)
        output = tmp_path / f"{len(calls)}-{twin}-{action}-{level}"
        code, out, err = run("plot", action, computed_twins / twin, "--level", level, "--output", output, *options)
        return code, out, err, output

    return call


def svg(folder, name="temperature.svg"):
    return ElementTree.fromstring((folder / name).read_bytes())


def texts(root):
    return ["".join(t.itertext()) for t in root.iter(f"{SVG}text")]


def panels(root):
    return [g for g in root.iter(f"{SVG}g") if g.get("transform", "").startswith("translate")]


def paths_with_opacity(root):
    return [p for p in root.iter(f"{SVG}path") if p.get("fill-opacity") is not None]


def date_ticks(root):
    return [t for t in texts(root) if re.fullmatch(r"\d\d-\d\d", t)]


@pytest.fixture(scope="module")
def mcmc_without_fields(tmp_path_factory):
    """Dossier de twins avec un mcmc raccourci aux champs non stockés (`store_fields: none`) : pas de bandes."""
    folder = tmp_path_factory.mktemp("sans-champs") / "twins"
    folder.mkdir()
    shutil.copytree(EXAMPLE_TWINS / "twin_03", folder / "twin_03")
    yaml = folder / "twin_03" / "twin.yaml"
    text = yaml.read_text(encoding="utf-8")
    for old, new in (("nb_iter: 300", "nb_iter: 40"), ("burn_in_max: 300", "burn_in_max: 20"), ("n_field_samples: 40", "n_field_samples: 16"),
                     ("schema: molonaviz-twin/1", "schema: molonari-twin/1.1.0"), ("  time_step: 1", "  time_step: 1\n  store_fields: none")):
        assert old in text
        text = text.replace(old, new)
    yaml.write_text(text, encoding="utf-8")
    run_twin(folder / "twin_03")
    return folder


def test_temperature_plot_has_one_panel_per_thermometer_and_residuals(plot):
    code, out, err, output = plot("temperature", "twin_01")
    assert code == 0, err
    assert [p.name for p in output.iterdir()] == ["temperature.svg"]
    root = svg(output)
    assert len(panels(root)) == 4
    shown = texts(root)
    for label in ("T1", "T2", "T3"):
        assert any(t.startswith(label) for t in shown), label
    assert "residual (°C)" in shown and any(t.startswith("Residuals") for t in shown)
    measured = [p for p in root.iter(f"{SVG}polyline") if p.get("stroke") == "#666666"]
    assert len(measured) == 3
    for line in measured:
        assert 2 < len(line.get("points").split()) <= ADMIN_MAX_POINTS
    assert "Plot written: " in out and str(output / "temperature.svg") in out


def test_inversion_plot_draws_the_90_percent_band(plot):
    code, _, err, output = plot("temperature", "twin_03")
    assert code == 0, err
    root = svg(output)
    assert "90 % credible band" in texts(root) and "90 % predictive band" in texts(root)
    assert "best-fit sample" in texts(root)
    bands = paths_with_opacity(root)
    opacities = [p.get("fill-opacity") for p in bands]
    assert opacities.count("0.3") == 3 and opacities.count("0.15") == 3  # une bande crédible et une prédictive par thermomètre
    code, _, err, output = plot("temperature", "twin_01")
    assert code == 0, err
    root = svg(output)
    assert not paths_with_opacity(root) and "forward model" in texts(root)
    assert not any("credible" in t or "predictive" in t for t in texts(root))


def test_start_and_end_restrict_the_period(plot):
    code, _, _, full = plot("temperature", "twin_01")
    assert code == 0
    code, _, err, output = plot("temperature", "twin_01", "--start", "2016-07-01T00:00:00", "--end", "2016-07-06T00:00:00")
    assert code == 0, err
    ticks = date_ticks(svg(output))
    assert ticks[0] >= "07-01" and ticks[-1] <= "07-06" and len(ticks) >= 3, ticks
    assert (output / "temperature.svg").read_bytes() != (full / "temperature.svg").read_bytes()
    assert "2016-07-01 00:00 to 2016-07-06 00:00" in "".join(texts(svg(output)))
    # les champs sont coupés de même
    code, _, err, fields = plot("fields", "twin_01", "--start", "2016-07-01T00:00:00", "--end", "2016-07-06T00:00:00")
    assert code == 0, err
    assert all(t >= "07-01" and t <= "07-06" for t in date_ticks(svg(fields, "fields-temperature.svg")))


def test_period_outside_the_data_is_the_period_empty_error(run, computed_twins, tmp_path):
    code, out, _ = run("plot", "temperature", computed_twins / "twin_01", "--level", "dev", "--output", tmp_path / "x", "--start", "2020-01-01", "--json")
    assert code == 1 and json.loads(out)["code"] == "E_QUESTION_PERIOD_EMPTY"
    code, out, _ = run("plot", "fields", computed_twins / "twin_01", "--level", "dev", "--output", tmp_path / "x", "--start", "2020-01-01", "--json")
    assert code == 1 and json.loads(out)["code"] == "E_QUESTION_PERIOD_EMPTY"
    assert not (tmp_path / "x").exists()


def test_fields_plot_works_on_a_twin_with_private_states(plot, computed_twins):
    config = (computed_twins / "twin_01" / "twin.yaml").read_text(encoding="utf-8")
    assert "visibility" not in config  # états privés : le défaut
    code, _, err, output = plot("fields", "twin_01")
    assert code == 0, err
    assert sorted(p.name for p in output.iterdir()) == ["fields-temperature.svg", "fields-water_flux.svg"]
    for name in ("fields-temperature.svg", "fields-water_flux.svg"):
        assert svg(output, name).get("viewBox")
    code, _, err, output = plot("fields", "twin_03")
    assert code == 0, err
    assert any("inversion model" in t for t in texts(svg(output, "fields-temperature.svg")))


def test_user_level_is_refused_with_a_structured_code(run, computed_twins, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for action in ("temperature", "fields"):
        for options in ((), ("--level", "user")):
            code, out, err = run("plot", action, computed_twins / "twin_01", *options)
            assert code == 1 and out == ""
            assert f"\"plot {action}\" requires the dev or opx level" in err and "Traceback" not in err
            code, out, err = run("plot", action, computed_twins / "twin_01", *options, "--json")
            assert code == 1 and err == ""
            assert json.loads(out)["code"] == "E_ACCESS_USER_ONLY"
            assert list(tmp_path.iterdir()) == []


def test_invalid_dates_are_a_structured_error(run, computed_twins, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cases = (("--start", "garbage"), ("--end", "garbage"), ("--start", "2016-06-28T00:00:00+02:00"),
             ("--start", "2016-07-02T00:00:00", "--end", "2016-07-01T00:00:00"))
    for action in ("temperature", "fields"):
        for dates in cases:
            code, out, err = run("plot", action, computed_twins / "twin_01", "--level", "dev", *dates)
            assert code == 1 and out == "" and "Invalid request" in err and "Traceback" not in err, (dates, err)
            code, out, err = run("plot", action, computed_twins / "twin_01", "--level", "dev", *dates, "--json")
            assert code == 1 and json.loads(out)["code"] == "E_REQUEST_INVALID", dates
            assert list(tmp_path.iterdir()) == []


def test_mcmc_without_stored_fields_has_no_band_and_says_so(run, mcmc_without_fields, tmp_path):
    twin = mcmc_without_fields / "twin_03"
    output = tmp_path / "plots"
    code, out, err = run("plot", "temperature", twin, "--level", "dev", "--output", output)
    assert code == 0, err
    root = svg(output)
    assert not paths_with_opacity(root)
    assert any("no band: the fields were not stored" in t for t in texts(root))
    assert "Warning:" in err and "stored" in err  # l'avertissement de la réponse, sur la sortie d'erreur
    code, out, _ = run("plot", "temperature", twin, "--level", "dev", "--output", output, "--json")
    assert code == 0 and "E_QUESTION_NO_FIELDS" in json.loads(out)["warnings"]
    # les champs du meilleur ajustement existent toujours (E_QUESTION_NO_FIELDS ne vient que des champs des échantillons) : deux figures
    code, _, err = run("plot", "fields", twin, "--level", "dev", "--output", tmp_path / "fields")
    assert code == 0, err
    assert sorted(p.name for p in (tmp_path / "fields").iterdir()) == ["fields-temperature.svg", "fields-water_flux.svg"]


def test_a_twin_without_results_is_refused(run, computed_twins, tmp_path):
    output = tmp_path / "plots"
    for action in ("temperature", "fields"):
        code, out, err = run("plot", action, computed_twins / "twin_09", "--level", "dev", "--output", output, "--json")
        assert code == 1 and json.loads(out)["code"] == "E_TWIN_NOT_COMPUTED"
        code, out, err = run("plot", action, computed_twins / "twin_09", "--level", "dev", "--output", output)
        assert code == 1 and "Traceback" not in err and err.strip()
        assert not output.exists()
    code, out, _ = run("plot", "temperature", computed_twins / "twin_99", "--level", "dev", "--output", output, "--json")
    assert code == 1 and json.loads(out)["code"] == "E_TWIN_NOT_FOUND" and not output.exists()


def test_admin_writes_the_same_bytes_as_opx(plot):
    for action, names in (("temperature", ["temperature.svg"]), ("fields", ["fields-temperature.svg", "fields-water_flux.svg"])):
        _, _, _, admin = plot(action, "twin_03", level="admin")
        _, _, _, opx = plot(action, "twin_03", level="opx")
        for name in names:
            assert (admin / name).read_bytes() == (opx / name).read_bytes() and (admin / name).stat().st_size > 1000


def test_output_in_public_reports_or_results_is_refused(run, computed_twins, tmp_path):
    for output in (tmp_path / "Point" / "public_reports" / "x", computed_twins / "twin_01" / "results" / "x", tmp_path / "results"):
        for action in ("temperature", "fields"):
            code, out, err = run("plot", action, computed_twins / "twin_01", "--level", "dev", "--output", output)
            folder = "results" if "results" in output.parts else "public_reports"
            assert code == 1 and f"is inside {folder}/: an admin plot shows measurements and is never published." in err
            code, out, _ = run("plot", action, computed_twins / "twin_01", "--level", "dev", "--output", output, "--json")
            error = json.loads(out)
            assert code == 1 and error["code"] == "E_PLOT_OUTPUT_FORBIDDEN" and error["params"]["folder"] == folder
            assert not output.exists() and not (tmp_path / "Point").exists()
    assert not (computed_twins / "twin_01" / "results" / "x").exists()


def test_output_folder_is_created_and_leaves_no_partial_file(run, computed_twins, tmp_path):
    output = tmp_path / "a" / "b"
    code, _, err = run("plot", "fields", computed_twins / "twin_01", "--level", "dev", "--output", output)
    assert code == 0, err
    assert sorted(p.name for p in output.iterdir()) == ["fields-temperature.svg", "fields-water_flux.svg"]


def test_default_output_is_the_twin_name_in_the_current_folder(run, computed_twins, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    code, _, err = run("plot", "temperature", computed_twins / "twin_01" / "twin.yaml", "--level", "dev")
    assert code == 0, err
    assert (tmp_path / "twin_01_plots" / "temperature.svg").is_file()


def test_plot_does_not_load_jax(computed_twins, tmp_path):
    code = ("import sys\nfrom molonari_twin.twin.cli import main\n"
            f"twin = {str(computed_twins / 'twin_03')!r}\n"
            f"statuses = [main(['plot', 'temperature', twin, '--level', 'dev', '--output', {str(tmp_path / 'a')!r}]),"
            f" main(['plot', 'fields', twin, '--level', 'opx', '--output', {str(tmp_path / 'b')!r}])]\n"
            "assert statuses == [0, 0], statuses\n"
            "loaded = sorted(m for m in sys.modules if m.split('.')[0] in ('jax', 'jaxlib', 'arviz'))\n"
            "assert not loaded, loaded")
    completed = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False, timeout=180,
                               env={**os.environ, "MOLONARI_LANG": "en"})
    assert completed.returncode == 0, completed.stderr


def test_json_output_lists_the_files_and_errors_use_the_v1_error(run, computed_twins, tmp_path):
    output = tmp_path / "json"
    code, out, err = run("plot", "fields", computed_twins / "twin_03", "--level", "admin", "--output", output, "--json")
    assert code == 0, err
    document = json.loads(out)
    assert list(document) == ["command", "twin", "level", "files", "warnings"]
    assert document["command"] == "plot fields" and document["twin"] == "twin_03" and document["level"] == "opx"
    assert [f["file"] for f in document["files"]] == [str(output.resolve() / "fields-temperature.svg"), str(output.resolve() / "fields-water_flux.svg")]
    for entry in document["files"]:
        content = Path(entry["file"]).read_bytes()
        assert entry["sha256"] == hashlib.sha256(content).hexdigest() and entry["bytes"] == len(content)
    assert all(isinstance(w, str) and w for w in document["warnings"])
    code, out, _ = run("plot", "temperature", computed_twins / "twin_01", "--output", output / "no", "--json")
    error = json.loads(out)
    assert code == 1 and error["code"] == "E_ACCESS_USER_ONLY" and set(error) >= {"code", "field", "params", "message_en"}
    assert not (output / "no").exists()


def test_messages_in_french_and_english(run, computed_twins, tmp_path):
    code, out, _ = run("plot", "fields", computed_twins / "twin_01", "--level", "dev", "--output", tmp_path / "fr", lang="fr")
    assert code == 0 and out.count("Figure écrite : ") == 2
    code, out, _ = run("plot", "fields", computed_twins / "twin_01", "--level", "dev", "--output", tmp_path / "en", lang="en")
    assert code == 0 and out.count("Plot written: ") == 2
    code, _, err = run("plot", "temperature", computed_twins / "twin_01", "--level", "dev", "--output", tmp_path / "results" / "x", lang="fr")
    assert code == 1 and "une figure d'admin montre des mesures et n'est jamais publiée" in err
    code, _, err = run("plot", "temperature", computed_twins / "twin_01", lang="fr")
    assert code == 1 and "nécessite le niveau dev ou opx" in err
    assert "{path}" in MESSAGES["E_PLOT_OUTPUT_FORBIDDEN"] and "{folder}" in MESSAGES["E_PLOT_OUTPUT_FORBIDDEN"]


def test_plot_bytes_are_deterministic_and_text_stays_in_the_viewbox(computed_twins):
    api = TwinAPI(computed_twins)
    for ref, expected_bands in (("twin_01", 0), ("twin_03", 6)):
        describe = api.ask(ref, "describe")
        sensors = api.ask(ref, "sensor_intervals", {"level": 0.9})
        fields = api.ask(ref, "fields", {"variables": ["temperature", "water_flux"]})
        role = plots.role_of(describe)
        first = [("temperature.svg", plots.temperature_figure(sensors, describe)), *plots.fields_figures(fields, describe, role)]
        second = [("temperature.svg", plots.temperature_figure(sensors, describe)), *plots.fields_figures(fields, describe, role)]
        assert first == second and [name for name, _ in first] == ["temperature.svg", "fields-temperature.svg", "fields-water_flux.svg"]
        for name, content in first:
            root, shown = parse(content)
            assert re.fullmatch(r"0 0 \d+\.\d\d \d+\.\d\d", root.get("viewBox")) and root.get("height") is None, name
            assert root[0].tag.endswith("title") and root[0].text and root[1].tag.endswith("desc") and root[1].text, name
            width = float(root.get("viewBox").split()[2])
            for text, left, right in text_extents(root):
                assert left >= -1 and right <= width + 1, f"{name}: « {text} » déborde ({left:.0f} à {right:.0f} sur {width:.0f})"
            if name == "temperature.svg":
                assert len(paths_with_opacity(root)) == expected_bands
                assert "Admin figure with measurements: never published." in root[1].text
                assert "temperature (°C)" in shown and f"T1, depth {float(describe.data['sensor_depths'].value[0]) + float(describe.data['offset'].value):g} m" in shown


def test_the_polylines_of_the_report_keep_their_200_points():
    from molonari_twin.report.svgplot import MAX_POINTS, Plot, decimate

    assert MAX_POINTS == 200 and ADMIN_MAX_POINTS > MAX_POINTS
    xs = np.arange(5000.0)
    assert len(decimate(xs, xs)[0]) == MAX_POINTS and len(decimate(xs, xs, 1500)[0]) == 1500
    for kwargs, expected in (({}, MAX_POINTS), ({"max_points": 1500}, 1500)):
        plot = Plot("t", width=300)
        plot.x_scale(0, 5000)
        plot.y_scale(0, 5000)
        plot.polyline(xs, xs, "#000000", **kwargs)
        plot.band(xs, xs, xs + 1, "#000000", **kwargs)
        body = "\n".join(plot._body)
        assert len(re.search(r'<polyline points="([^"]*)"', body).group(1).split()) == expected
        assert len(re.search(r'<path d="M([^L]*)L', body).group(1).split()) == 2 * expected


def test_controls_run_in_order_level_then_output_then_reading(run, computed_twins, tmp_path):
    forbidden = tmp_path / "public_reports" / "x"
    code, out, _ = run("plot", "temperature", computed_twins / "twin_09", "--output", forbidden, "--start", "garbage", "--json")
    assert code == 1 and json.loads(out)["code"] == "E_ACCESS_USER_ONLY"  # le niveau d'abord
    code, out, _ = run("plot", "temperature", computed_twins / "twin_09", "--level", "dev", "--output", forbidden, "--start", "garbage", "--json")
    assert code == 1 and json.loads(out)["code"] == "E_PLOT_OUTPUT_FORBIDDEN"  # puis la sortie, avant toute lecture
    code, out, _ = run("plot", "temperature", computed_twins / "twin_09", "--level", "dev", "--output", tmp_path / "ok", "--start", "garbage", "--json")
    assert code == 1 and json.loads(out)["code"] == "E_REQUEST_INVALID" and not (tmp_path / "ok").exists()
