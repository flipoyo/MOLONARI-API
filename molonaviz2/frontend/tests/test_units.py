"""Unités d'affichage : l'interface réexporte l'intervalle de mesure des questions du backend."""

from molonari_twin.questions import MEASUREMENT_LEVELS, measurement_interval, z_score
from molonaviz2.ui import units


def test_measurement_interval_is_reexported_by_the_interface():
    assert units.z_score is z_score and units.measurement_interval is measurement_interval
    assert units.MEASUREMENT_LEVELS is MEASUREMENT_LEVELS
