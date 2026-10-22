import os
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

REPO_ROOT = Path(__file__).resolve().parents[2]  # racine du dépôt MOLONARI-API (contient l'ancien Molonaviz)
WORKSPACE_ROOT = Path(__file__).resolve().parents[3]  # espace cgs MOLONARI1D
DATA_DIR = WORKSPACE_ROOT / "data"
LAB_DIR = DATA_DIR / "Laboratory"
POINT034_DIR = DATA_DIR / "SamplingPoints" / "Point034"
FIXTURES = Path(__file__).parent / "fixtures"


def synthetic_inputs(nb_times=300, depths=(0.1, 0.2, 0.3, 0.4), offset=0.0, step_minutes=15, sigma_t=0.1):
    """Colonne synthétique simple : rivière sinusoïdale journalière, aquifère constant."""
    from molonaviz2.engine.column import ColumnInputs

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
