"""`molonari-twin report --output <point>/public_reports` (T10) : les octets écrits, npz compris, ne contiennent ni la graine
ni un chemin absolu. Twins 2.0.0 calculés, donc `slow`."""

import json
import zipfile
from pathlib import Path

import numpy as np
import pytest
from molonari_twin.twin.cli import main
from molonari_twin.twin.run import run_twin
from test_report import data_repo, listed, on_disk, write_space  # noqa: F401  (data_repo : fixture)

SEED = 7391825  # distinctif : 7 chiffres, absent des valeurs calculées


def every_byte(folder: Path) -> dict[str, bytes]:
    """Octets à balayer : chaque fichier tel qu'écrit, et, pour un .npz (zip compressé), chacune de ses entrées décompressées."""
    found = {}
    for path in sorted(p for p in folder.rglob("*") if p.is_file()):
        name = path.relative_to(folder).as_posix()
        found[name] = path.read_bytes()
        if path.suffix == ".npz":
            with zipfile.ZipFile(path) as archive:
                assert archive.testzip() is None
                for member in archive.namelist():
                    found[f"{name}!{member}"] = archive.read(member)
    return found


@pytest.mark.slow
def test_report_output_to_public_reports_has_neither_the_seed_nor_an_absolute_path(tmp_path, data_repo, capsys):  # noqa: F811
    point = tmp_path / "workspace" / "Point034"
    twins = write_space(point, seed=SEED)
    run_twin(twins / "twin_01")
    run_twin(twins / "twin_02")
    assert f"seed: {SEED}" in (twins / "twin_02" / "orx.yaml").read_text(encoding="utf-8")  # la graine est bien sur le disque
    output = point / "public_reports"
    capsys.readouterr()

    assert main(["report", str(twins / "twin_02"), "--output", str(output)]) == 0
    assert str(output) in capsys.readouterr().out  # le terminal dit où : pas le rapport
    assert not (twins / "twin_02" / "Public Report").exists()
    index = json.loads((output / "report.json").read_text(encoding="utf-8"))
    assert on_disk(output) == listed(index)
    npz = [e["file"] for e in index["entries"] if e["file"].endswith(".npz")]
    assert len(npz) >= 2, npz  # l'a posteriori et les chaînes (BN-32, BN-34) : le balayage couvre des npz

    scanned = every_byte(output)
    assert any("!" in name for name in scanned)  # les entrées des npz sont balayées
    seeds = (str(SEED).encode(), np.int64(SEED).tobytes(), np.int32(SEED).tobytes(), np.uint64(SEED).tobytes(), np.uint32(SEED).tobytes())
    absolute = (str(tmp_path).encode(), str(output).encode(), str(point).encode(), str(Path.home()).encode(), str(tmp_path.resolve()).encode(),
                str(data_repo).encode(), b"/private/", b"/var/folders", b"/Users/", b"/home/")
    words = (b"seed", b"orx", b"secret description", b"Point034")
    for name, content in scanned.items():
        # les mots courts (3 à 4 octets) ne se cherchent pas dans l'archive compressée, où ils pourraient apparaître par hasard :
        # ses entrées décompressées, balayées à part, les couvrent
        for forbidden in (*seeds, *absolute, *([] if name.endswith(".npz") else words)):
            assert forbidden not in content, (name, forbidden)
        assert str(tmp_path) not in name and "Point034" not in name

    # le npz se relit sans pickle, et aucun de ses tableaux n'est de type objet (donc aucun chemin caché dans un pickle)
    for name in npz:
        with np.load(output / name, allow_pickle=False) as archive:
            assert all(archive[key].dtype.kind != "O" for key in archive.files)
