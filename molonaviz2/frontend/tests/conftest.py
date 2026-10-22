import os
from pathlib import Path

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

WORKSPACE = Path(__file__).resolve().parents[2]  # espace uv molonaviz2/ (backend/, frontend/, examples/)
FIXTURES = WORKSPACE / "backend" / "tests" / "fixtures"  # mesures de test, partagées avec le backend


def synthetic_inputs(nb_times=300, depths=(0.1, 0.2, 0.3, 0.4), offset=0.0, step_minutes=15, sigma_t=0.1):
    """Colonne synthétique simple : rivière sinusoïdale journalière, aquifère constant."""
    from molonari_twin.facade import ColumnInputs

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
