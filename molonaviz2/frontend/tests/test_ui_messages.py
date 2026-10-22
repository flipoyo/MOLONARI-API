"""Erreurs du cœur affichées par l'interface en français : traduites à partir de leur code, avec le même
texte qu'en 0.2.0, où le cœur appelait t(modèle).format(...) lui-même."""

import pytest
from molonari_twin.errors import UserError
from molonaviz2.i18n import install_language, load_catalog
from molonaviz2.ui import errors as ui_errors

FR = load_catalog("fr")


@pytest.fixture
def shown_errors(qapp):
    install_language(qapp, "fr")
    errors = []
    previous = ui_errors.set_presenter(lambda title, message, details: errors.append((title, message, details)))
    yield errors
    ui_errors.set_presenter(previous)
    install_language(qapp, "en")


def test_core_error_is_shown_in_french(shown_errors, tmp_path):
    from molonari_twin.storage.project import Project

    path = tmp_path / "absent.molo"
    with pytest.raises(UserError) as error:
        Project.open(path)
    ui_errors.report_exception(error.value)
    assert shown_errors == [(FR["Action not completed"], FR["File not found: {path}"].format(path=path), "")]


def test_grouped_settings_issues_are_shown_in_french(shown_errors):
    from molonari_twin.domain.compute import ForwardConfig, LayerSpec

    issues = ForwardConfig([LayerSpec("c1", z_low=0.4)], nb_cells=5).validate(0.4)
    ui_errors.report_exception(UserError("E_SETTINGS_INVALID", issues=issues))
    assert shown_errors[0][1] == FR["The number of cells must be between 10 and 2000."]


def test_twin_issue_keeps_its_key_in_french(tmp_path):
    from molonari_twin.twin.config import TwinConfigError, load_twin
    from molonaviz2.i18n import render, translate

    (tmp_path / "twin.yaml").write_text("schema: molonaviz-twin/1\nname: x\nmode: direct\nsensors: {ofset: 0}\n")
    with pytest.raises(TwinConfigError) as error:
        load_twin(tmp_path, check_data=False)
    rendered = [render(issue, lambda s: translate(s, "fr")) for issue in error.value.issues]
    unknown = next(r for r in rendered if r.startswith("sensors.ofset: "))
    assert unknown.startswith("sensors.ofset: " + FR["unknown key (allowed: {keys})."].split("{keys}")[0])
