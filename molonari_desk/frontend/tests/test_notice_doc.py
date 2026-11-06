"""La table des paramètres de la notice scientifique (docs/NOTICE_SCIENTIFIQUE.md) est générée depuis la notice de
l'application, et à jour."""

import importlib.util
import re
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("gen_notice_tables", WORKSPACE / "tools" / "gen_notice_tables.py")
gen_notice_tables = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gen_notice_tables)
DOCS = WORKSPACE / "docs"


def test_parameter_table_is_up_to_date():
    text = gen_notice_tables.DOC.read_text(encoding="utf-8")
    assert text == gen_notice_tables.render(text), (
        "docs/NOTICE_SCIENTIFIQUE.md n'est pas à jour : python tools/gen_notice_tables.py (depuis molonari_desk/)")


def test_table_covers_every_computation_setting():
    from molonari_desk import notice

    table = gen_notice_tables.table()
    for entry in notice.ENTRIES:
        if entry.section in (notice.LAYERS, notice.DIRECT, notice.INVERSION, notice.ADVANCED) and entry.twin_key:
            assert f"`{entry.twin_key}`" in table, entry.key


def test_relative_links_of_the_notices_exist():
    for name in ("NOTICE_UTILISATEUR.md", "NOTICE_SCIENTIFIQUE.md"):
        for target in re.findall(r"\]\(([^)#]+)(?:#[^)]*)?\)", (DOCS / name).read_text(encoding="utf-8")):
            if not target.startswith("http"):
                assert (DOCS / target).exists(), (name, target)
