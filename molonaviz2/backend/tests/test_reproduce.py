"""T6 : `reproduce` — relance un twin et verifie le determinisme.

Tests : deux reproduce donnent « identique », un octet modifie donne « different »
avec le fichier en cause, le rapport est JSON et texte, sans graine.
"""

import json

import pytest
from molonari_twin.errors import UserError
from molonari_twin.twin.reproduce import reproduce
from molonari_twin.twin.run import twin_status


@pytest.mark.slow
def test_reproduce_identical_on_same_machine(computed_twins):
    """Deux reproduce du meme twin donnent des resultats identiques."""
    twin_folder = computed_twins / "twin_03"
    report = reproduce(twin_folder, strict=False)
    assert report.all_files_match
    assert report.comparable_metadata_match
    assert report.chain_state_match
    assert report.message == "IDENTIQUE"
    assert len(report.files_compared) > 0

    # Deuxieme reproduction : meme resultat
    report2 = reproduce(twin_folder, strict=False)
    assert report2.all_files_match
    assert report2.message == "IDENTIQUE"
    assert report.twin_id == report2.twin_id
    assert len(report.files_compared) == len(report2.files_compared)


@pytest.mark.slow
def test_reproduce_detects_modified_file(computed_twins):
    """Un octet modifie dans les resultats est detecte."""
    twin_folder = computed_twins / "twin_03"
    # S'assurer qu'il y a des resultats
    status = twin_status(twin_folder)
    assert status.has_results

    # Modifier un fichier de resultats
    posterior = status.results_dir / "posterior.npz"
    original = posterior.read_bytes()
    modified = bytearray(original)
    modified[42] ^= 0xFF
    posterior.write_bytes(modified)

    report = reproduce(twin_folder, strict=False)
    assert not report.all_files_match
    assert report.message == "DIFFERENCES DETECTEES"
    assert any(f.path == "posterior.npz" and not f.match for f in report.files_compared)


@pytest.mark.slow
def test_reproduce_report_format_json_and_text(computed_twins):
    """Le rapport a les champs JSON attendus et un texte lisible, sans graine."""
    twin_folder = computed_twins / "twin_03"
    report = reproduce(twin_folder, strict=False)

    d = report.to_dict()
    for key in ("machine", "platform", "python_version", "code_version", "device",
                "timestamp", "twin_id", "all_files_match", "files_compared",
                "comparable_metadata_match", "chain_state_match", "message"):
        assert key in d, key

    text = report.to_text()
    assert "Rapport de reproduction" in text
    assert report.twin_id in text

    # Pas de graine dans le rapport
    json_str = json.dumps(d)
    assert '"seed"' not in json_str


@pytest.mark.slow
def test_reproduce_no_reference_raises_error(computed_twins):
    """reproduce() leve une erreur si le twin n'a pas de resultats."""
    twin_folder = computed_twins / "twin_09"
    with pytest.raises(UserError) as exc:
        reproduce(twin_folder, strict=False)
    assert exc.value.code == "E_REPRODUCE_NO_REFERENCE"


@pytest.mark.slow
def test_reproduce_includes_chain_state(computed_twins):
    """chain_state.npz est compare tableau par tableau (T3)."""
    twin_folder = computed_twins / "twin_03"
    report = reproduce(twin_folder, strict=False)
    assert report.chain_state_match
    # chain_state n'est pas dans files_compared (traite a part)
    assert not any(f.path == "chain_state.npz" for f in report.files_compared)
