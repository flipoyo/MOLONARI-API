"""Codes d'erreur et catalogue des messages : chaque code a un modèle, chaque modèle sa traduction,
et chaque appel à code littéral fournit les paramètres de son modèle (vérifié par AST, y compris
pour les chemins rarement exécutés)."""

import ast
import copy
import json
import pickle
import re
import string
from pathlib import Path

import pytest

from molonaviz2.errors import MESSAGES, Issue, UserError, message_template

SOURCE = Path(__file__).resolve().parents[1] / "src" / "molonaviz2"
CATALOG = json.loads((SOURCE / "i18n" / "fr.json").read_text(encoding="utf-8"))
CODE = re.compile(r"^[EW]_[A-Z0-9_]+$")

# Position de l'argument « code » des constructeurs.
CODE_ARGUMENT = {"UserError": 0, "Issue": 1}
# Fonctions qui relaient un code reçu en paramètre (code non littéral admis) : fichier → noms.
FORWARDERS = {"errors.py": {"__init__", "__new__", "_rebuild_error", "from_error"}}


def fields(template: str) -> set[str]:
    return {name for _, name, _, _ in string.Formatter().parse(template) if name}


def calls():
    """(fichier, ligne, fonction englobante, nom appelé, nœud) pour chaque appel UserError(...) / Issue(...)."""
    for path in sorted(SOURCE.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        relative = path.relative_to(SOURCE).as_posix()

        def visit(node, function, relative=relative):
            for child in ast.iter_child_nodes(node):
                if isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef):
                    yield from visit(child, child.name)
                    continue
                if isinstance(child, ast.Call):
                    func = child.func
                    name = func.id if isinstance(func, ast.Name) else func.attr if isinstance(func, ast.Attribute) else None
                    if name in CODE_ARGUMENT:
                        yield relative, child.lineno, function, name, child
                yield from visit(child, function)

        yield from visit(tree, None)


def code_argument(name, node):
    position = CODE_ARGUMENT[name]
    if len(node.args) > position:
        return node.args[position]
    return next((k.value for k in node.keywords if k.arg == "code"), None)


def test_codes_are_well_formed():
    bad = [code for code in MESSAGES if not CODE.match(code)]
    assert not bad, bad


def test_every_template_has_its_french_translation():
    # Un modèle sans aucune lettre hors de ses champs ("{issues}", "{point} : {error}") n'a rien à traduire.
    missing = [t for t in MESSAGES.values() if re.search(r"[^\W\d_]", re.sub(r"\{[^}]*\}", "", t)) and t not in CATALOG]
    assert not missing, missing


def test_translations_keep_the_template_fields():
    wrong = [t for t in MESSAGES.values() if t in CATALOG and fields(CATALOG[t]) != fields(t)]
    assert not wrong, wrong


def test_codes_are_literal_and_known():
    problems = []
    for file, line, function, name, node in calls():
        code = code_argument(name, node)
        where = f"{file}:{line}"
        if isinstance(code, ast.Constant) and isinstance(code.value, str):
            if CODE.match(code.value):
                if code.value not in MESSAGES:
                    problems.append(f"{where} code inconnu {code.value}")
            elif "E_LEGACY" not in MESSAGES:
                problems.append(f"{where} {code.value!r} n'est pas un code")
        elif function in FORWARDERS.get(file, set()):
            continue
        elif "E_LEGACY" in MESSAGES:
            continue  # migration en cours : texte libre encore admis
        else:
            problems.append(f"{where} code non littéral")
    assert not problems, "\n".join(problems)


def test_literal_codes_receive_every_template_field():
    problems = []
    for file, line, _, name, node in calls():
        code = code_argument(name, node)
        if not (isinstance(code, ast.Constant) and isinstance(code.value, str) and code.value in MESSAGES):
            continue
        if any(k.arg is None for k in node.keywords):
            problems.append(f"{file}:{line} {code.value} : paramètres passés par **, invérifiables")
            continue
        given = {k.arg for k in node.keywords}
        for missing in sorted(fields(MESSAGES[code.value]) - given):
            problems.append(f"{file}:{line} {code.value} : paramètre {missing} manquant")
    assert not problems, "\n".join(problems)


# Comportement de UserError et Issue --------------------------------------------------------------


@pytest.fixture
def template(monkeypatch):
    monkeypatch.setitem(MESSAGES, "E_TEST_FILE", "File not found: {path}")
    return "E_TEST_FILE"


def test_user_error_carries_code_params_and_english_text(template):
    error = UserError(template, "data.measurements", path="a.csv")
    assert (error.code, error.field, error.params) == (template, "data.measurements", {"path": "a.csv"})
    assert str(error) == "File not found: a.csv" == message_template(template).format(path="a.csv")
    copied = pickle.loads(pickle.dumps(error))
    assert (type(copied), copied.code, copied.params, str(copied)) == (UserError, template, {"path": "a.csv"}, str(error))


def test_issue_is_a_string_with_its_code(template):
    issue = Issue("data.measurements", template, path="a.csv")
    assert issue == "File not found: a.csv" == issue.message
    assert (issue.code, issue.field, issue.params) == (template, "data.measurements", {"path": "a.csv"})
    prefixed = Issue("data.measurements", template, prefixed=True, path="a.csv")
    assert prefixed == "data.measurements: File not found: a.csv" and prefixed.message == "File not found: a.csv"
    assert "measurements" in prefixed and f"{prefixed}\n{issue}".count("a.csv") == 2
    for copied in (pickle.loads(pickle.dumps(prefixed)), copy.deepcopy(prefixed)):
        assert (copied, copied.code, copied.field, copied.prefixed) == (prefixed, template, "data.measurements", True)


def test_user_error_grouping_issues(template, monkeypatch):
    monkeypatch.setitem(MESSAGES, "E_TEST_GROUP", "{issues}")
    issues = [Issue("a", template, path="x"), Issue("b", template, path="y")]
    error = UserError("E_TEST_GROUP", issues=issues)
    assert str(error) == "File not found: x\nFile not found: y" and error.params["issues"] == issues


def test_unknown_code_is_refused():
    with pytest.raises(KeyError):
        UserError("E_DOES_NOT_EXIST")


# Traduction au bord ------------------------------------------------------------------------------


def french(text: str) -> str:
    from molonaviz2.i18n import translate

    return translate(text, "fr")


def test_render_in_english_is_the_text(template, monkeypatch):
    from molonaviz2.i18n import render

    monkeypatch.setitem(MESSAGES, "E_TEST_GROUP", "{issues}")
    issue = Issue("data", template, prefixed=True, path="a.csv")
    for obj in (UserError(template, path="a.csv"), issue, Issue("data", template, path="a.csv"), UserError("E_TEST_GROUP", issues=[issue, issue])):
        assert render(obj, lambda s: s) == str(obj)


def test_render_in_french_uses_the_catalog(template, monkeypatch):
    from molonaviz2.i18n import render

    monkeypatch.setitem(MESSAGES, "E_TEST_GROUP", "{issues}")
    assert render(UserError(template, path="a.csv"), french) == "Fichier introuvable : a.csv"
    issue = Issue("data.measurements", template, prefixed=True, path="a.csv")
    assert render(issue, french) == "data.measurements: Fichier introuvable : a.csv"
    grouped = UserError("E_TEST_GROUP", issues=[issue, Issue("x", template, path="b.csv")])
    assert render(grouped, french) == "data.measurements: Fichier introuvable : a.csv\nFichier introuvable : b.csv"


def test_render_delegates_to_objects_that_localize_themselves():
    from molonaviz2.i18n import render

    class Located:
        def localized(self, tr):
            return tr("line {n}").format(n=3)

    assert render(Located(), french) == "ligne 3"
    assert render(ValueError("boom"), french) == "boom"


def test_a_message_parameter_may_be_called_field(monkeypatch):
    monkeypatch.setitem(MESSAGES, "W_TEST_FIELD", "Could not read {value} for {field}.")
    issue = Issue(None, "W_TEST_FIELD", value="x", field="offset")
    error = UserError("W_TEST_FIELD", None, value="x", field="offset")
    assert issue == str(error) == "Could not read x for offset." and issue.field is None and error.field is None


def test_issue_from_error_and_nested_errors_are_rendered(template, monkeypatch):
    from molonaviz2.i18n import render

    monkeypatch.setitem(MESSAGES, "W_TEST_SKIPPED", "{point} : {error}")
    error = UserError(template, path="a.csv")
    assert Issue.from_error("data", error) == "File not found: a.csv" and Issue.from_error("data", error).code == template
    skipped = Issue(None, "W_TEST_SKIPPED", point="P1", error=error)
    assert skipped == "P1 : File not found: a.csv"
    assert render(skipped, french) == "P1 : Fichier introuvable : a.csv"
