"""Erreurs du cœur affichées par l'interface en français : traduites à partir de leur code, avec le même
texte qu'en 0.2.0, où le cœur appelait t(modèle).format(...) lui-même."""

import ast
import string
from pathlib import Path

import pytest
from molonari_twin.errors import MESSAGES, UserError
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


# Codes d'erreur créés par l'application ---------------------------------------------------------

SOURCE = Path(__file__).resolve().parents[1] / "src" / "molonaviz2"
CODE_ARGUMENT = {"UserError": 0, "Issue": 1}  # position de l'argument « code »


def application_calls():
    """(fichier:ligne, nœud) pour chaque appel UserError(...) / Issue(...) du code de l'application."""
    for path in sorted(SOURCE.rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Call):
                func = node.func
                name = func.id if isinstance(func, ast.Name) else func.attr if isinstance(func, ast.Attribute) else None
                if name in CODE_ARGUMENT:
                    position = CODE_ARGUMENT[name]
                    code = node.args[position] if len(node.args) > position else None
                    yield f"{path.relative_to(SOURCE)}:{node.lineno}", code, node


def template_fields(template: str) -> set[str]:
    return {name for _, name, _, _ in string.Formatter().parse(template) if name}


def test_application_codes_are_literal_known_and_complete():
    found = list(application_calls())
    assert found, "aucun appel relevé : le test ne vérifie plus rien"
    problems = []
    for where, code, node in found:
        if not (isinstance(code, ast.Constant) and isinstance(code.value, str)):
            problems.append(f"{where} code non littéral")
        elif code.value not in MESSAGES:
            problems.append(f"{where} code inconnu {code.value}")
        elif any(k.arg is None for k in node.keywords):
            problems.append(f"{where} {code.value} : paramètres passés par **, invérifiables")
        else:
            given = {k.arg for k in node.keywords}
            problems += [f"{where} {code.value} : paramètre {name} manquant" for name in sorted(template_fields(MESSAGES[code.value]) - given)]
    assert not problems, "\n".join(problems)
