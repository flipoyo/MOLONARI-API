import os
import shutil
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

REPO_ROOT = Path(__file__).resolve().parents[3]  # racine du dépôt MOLONARI-API (contient l'ancien Molonaviz)
WORKSPACE_ROOT = Path(__file__).resolve().parents[4]  # espace cgs MOLONARI1D
DATA_DIR = WORKSPACE_ROOT / "data"
LAB_DIR = DATA_DIR / "Laboratory"
POINT034_DIR = DATA_DIR / "SamplingPoints" / "Point034"
FIXTURES = Path(__file__).parent / "fixtures"
EXAMPLE_TWINS = Path(__file__).resolve().parents[2] / "examples" / "twins"


def synthetic_inputs(nb_times=300, depths=(0.1, 0.2, 0.3, 0.4), offset=0.0, step_minutes=15, sigma_t=0.1):
    """Colonne synthétique simple : rivière sinusoïdale journalière, aquifère constant."""
    from molonari_twin.engine.column import ColumnInputs

    i = np.arange(nb_times)
    times = (np.datetime64("2016-06-27T12:00") + i * np.timedelta64(step_minutes, "m")).astype("datetime64[s]")
    return ColumnInputs(
        times=times,
        head=0.05 + 0.01 * np.sin(i / 40),
        t_river=15 + 3 * np.sin(2 * np.pi * i / 96),
        t_sensors=np.stack([15 + 2 * np.sin(2 * np.pi * i / 96 - 0.5 * k) for k in range(1, 4)], axis=1),
        t_aquifer=np.full(nb_times, 13.0),
        sensor_depths=depths,
        offset=offset,
        sigma_pressure=0.01,
        sigma_temperature=sigma_t,
    )


@pytest.fixture
def inputs():
    return synthetic_inputs()


@pytest.fixture
def pyheatmy_float64():
    """pyheatmy avec ses calculs internes passés en float64.

    pyheatmy calcule en float32 (solveur et tableaux de résultats). Pour vérifier
    que le portage JAX est exact, on remplace temporairement ces éléments par des
    équivalents float64 ; les écarts restants doivent être au niveau de l'arrondi.
    """
    import types

    import pyheatmy.linear_system as ls
    from pyheatmy import core

    def solver(a, b, c, d):
        n = len(d)
        bc, dc = b.astype(float).copy(), d.astype(float).copy()
        for i in range(1, n):
            m = a[i - 1] / bc[i - 1]
            bc[i] -= m * c[i - 1]
            dc[i] -= m * dc[i - 1]
        x = np.zeros(n)
        x[-1] = dc[-1] / bc[-1]
        for i in range(n - 2, -1, -1):
            x[i] = (dc[i] - c[i] * x[i + 1]) / bc[i]
        return x

    def tri_product(a, b, c, d):
        r = b * d
        r[1:] += a * d[:-1]
        r[:-1] += c * d[1:]
        return r

    shim = types.SimpleNamespace(**{k: getattr(np, k) for k in dir(np) if not k.startswith("__")})
    shim.float32 = np.float64
    saved = (ls.solver, ls.tri_product, ls.float32, ls.np, core.np)
    ls.solver, ls.tri_product, ls.float32, ls.np, core.np = solver, tri_product, np.float64, shim, shim
    yield
    ls.solver, ls.tri_product, ls.float32, ls.np, core.np = saved


# Dépôts git temporaires : identité et options neutres, indépendantes de la configuration git de la machine.
GIT = ["git", "-c", "user.name=Test", "-c", "user.email=test@example.org", "-c", "commit.gpgsign=false", "-c", "tag.gpgsign=false",
       "-c", "core.hooksPath=/dev/null", "-c", "init.defaultBranch=main"]


def git_run(repo, *args):
    import subprocess

    return subprocess.run([*GIT, *args], cwd=repo, check=True, capture_output=True, text=True).stdout.strip()


def commit_file(repo, name, text):
    path = repo / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    git_run(repo, "add", name)
    git_run(repo, "commit", "-q", "-m", f"modifie {name}")


@pytest.fixture(scope="session")
def computed_twins(tmp_path_factory):
    """Dossier de twins : twin_01 (direct) et twin_03 (mcmc, raccourci, champs complets) calculés, twin_09 jamais calculé."""
    folder = tmp_path_factory.mktemp("workspace") / "twins"
    folder.mkdir()
    shutil.copytree(EXAMPLE_TWINS / "twin_01", folder / "twin_01")
    shutil.copytree(EXAMPLE_TWINS / "twin_03", folder / "twin_03")
    yaml = folder / "twin_03" / "twin.yaml"
    text = yaml.read_text(encoding="utf-8")
    for old, new in (("nb_iter: 300", "nb_iter: 40"), ("burn_in_max: 300", "burn_in_max: 20"), ("n_field_samples: 40", "n_field_samples: 16"),
                     # champs complets (T et q) : nécessaires à P(q > 0) et à BN-30, qui n'existent pas aux capteurs seuls
                     ("schema: molonaviz-twin/1", "schema: molonari-twin/1.1.0"), ("  time_step: 1", "  time_step: 1\n  store_fields: full")):
        assert old in text
        text = text.replace(old, new)
    yaml.write_text(text, encoding="utf-8")
    shutil.copytree(EXAMPLE_TWINS / "twin_01", folder / "twin_09")
    yaml = folder / "twin_09" / "twin.yaml"
    # Un autre réglage (le nom seul ne change pas le twin_id : ce serait un doublon de twin_01, déjà calculé).
    yaml.write_text(yaml.read_text(encoding="utf-8").replace("name: twin_01", "name: twin_09").replace("nb_cells: 100", "nb_cells: 80"),
                    encoding="utf-8")
    from molonari_twin.twin.run import run_twin

    run_twin(folder / "twin_01")
    run_twin(folder / "twin_03")
    return folder
