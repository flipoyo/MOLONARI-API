"""Figures de l'admin (F2) : les températures mesurées en face des simulées, et les champs d'un twin. **Jamais publiées.**

Ces figures montrent les mesures des thermomètres, que le rapport public garde pour lui (F1) : elles sont écrites par
`molonari-twin plot temperature|fields`, hors de `public_reports/` et de `results/`, aux niveaux dev et opx.

Ce module ne lit rien : il reçoit des `Answer` obtenues par `LeveledAPI` (`sensor_intervals`, `fields`, `describe`) et
réutilise les primitives de `svgplot.py` et les figures de champs de `figures.py` (F1). Mêmes octets d'une génération à
l'autre ; libellés en anglais, comme les SVG du rapport.
"""

from __future__ import annotations

import numpy as np

from ..questions.answer import Answer
from ..questions.period import within
from .figures import (
    MODEL,
    _padded,
    format_time,
    temperature_field_figure,
    water_flux_field_figure,
)
from .svgplot import BLUE, GREY, ORANGE, VERMILLION, Figure, Plot

ADMIN_MAX_POINTS = 1500
"""Points au plus par courbe et par bande : les mesures de l'admin sont plus fines que celles d'une figure publique
(`svgplot.MAX_POINTS` = 200)."""
PANEL_WIDTH = 760
SENSORS = 3
RESIDUAL_COLORS = (BLUE, ORANGE, VERMILLION)
BAND_OPACITY = (0.3, 0.15)  # bande crédible, bande prédictive
BAND_SWATCH = ("#b2d1e1", "#d9e8f0")  # BLUE à ces opacités sur fond blanc : la légende ne porte pas l'opacité


def role_of(describe: Answer) -> str:
    """`forward` pour un twin direct, `inversion` pour un twin mcmc : le rôle des figures de champs de F1."""
    return "forward" if str(describe.data["mode"].value) == "direct" else "inversion"


def _seconds(times) -> np.ndarray:
    return np.asarray(times).astype("datetime64[s]").astype(np.int64).astype(float)


def _on(seconds: np.ndarray, field_seconds: np.ndarray, series: np.ndarray) -> np.ndarray:
    """La série (3, T′) des dates `field_seconds` interpolée aux dates `seconds` ; hors de son étendue, nan (pas de bande)."""
    if len(field_seconds) == len(seconds) and np.array_equal(field_seconds, seconds):
        return series
    return np.stack([np.interp(seconds, field_seconds, row, left=np.nan, right=np.nan) for row in series])


def _bands(sensors: Answer, seconds: np.ndarray) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """(libellé, bas, haut) des bandes crédible puis prédictive, chacune (3, T) aux dates des mesures ; vide sans bande."""
    if "credible" not in sensors.data or "predictive" not in sensors.data:
        return []
    field_seconds = _seconds(sensors.data["field_time"].value) if "field_time" in sensors.data else seconds
    level = float(sensors.data["level"].value) * 100
    found = []
    for name, label in (("credible", "credible band"), ("predictive", "predictive band")):
        low, high = sensors.data[name].interval
        found.append((f"{level:g} % {label}", _on(seconds, field_seconds, np.asarray(low, float)), _on(seconds, field_seconds, np.asarray(high, float))))
    return found


def _rmse(residual: np.ndarray) -> str:
    finite = residual[np.isfinite(residual)]
    return f"{float(np.sqrt(np.mean(finite**2))):.3f}" if finite.size else "n/a"


def temperature_figure(sensors: Answer, describe: Answer, start=None, end=None) -> bytes:
    """Quatre panneaux : un par thermomètre T1 à T3 (mesure en gris, simulation en bleu et, pour une inversion aux champs
    stockés, bandes crédible et prédictive), puis les résidus (simulé − mesuré) des trois, avec leur RMSE. `start` et `end`
    (datetime, bornes incluses) coupent la période avant le tracé : E_QUESTION_PERIOD_EMPTY si elle est vide. T4, condition
    limite, n'est pas tracé."""
    times = np.asarray(sensors.data["time"].value).astype("datetime64[s]")
    keep = within(times, start, end)
    times, seconds = times[keep], _seconds(times[keep])
    measured = np.asarray(sensors.data["measured"].value, float)[:, keep]
    simulated = np.asarray(sensors.data["simulated"].value, float)[:, keep]
    residual = simulated - measured
    bands = [(label, low[:, keep], high[:, keep]) for label, low, high in _bands(sensors, _seconds(sensors.data["time"].value))]
    inversion = role_of(describe) == "inversion"
    depths = np.asarray(describe.data["sensor_depths"].value, float) + float(describe.data["offset"].value)
    best = "best-fit sample" if inversion else "forward model"

    low, high = (float(seconds[0]), float(seconds[-1])) if len(seconds) else (0.0, 1.0)
    high = high if high > low else low + 1.0
    panels = []
    for i in range(SENSORS):
        plot = Plot(f"T{i + 1}, depth {depths[i]:g} m", "", width=PANEL_WIDTH, left=64, right=20, desc="")
        entries = [(GREY, "measured", "line"), (BLUE, best, "line")]
        entries += [(swatch, label, "box") for (label, _, _), swatch in zip(bands, BAND_SWATCH, strict=False)]
        plot.legend(*entries)
        plot.plot_height(150)
        x = plot.x_scale(low, high)
        values = [measured[i], simulated[i], *[v[i] for _, lo, hi in bands for v in (lo, hi)]]
        finite = np.concatenate([v[np.isfinite(v)] for v in values])
        y = plot.y_scale(*_padded(float(finite.min()), float(finite.max()))) if finite.size else plot.y_scale(0.0, 1.0)
        for (_, lo, hi), opacity in zip(bands, BAND_OPACITY, strict=False):
            plot.band(seconds, lo[i], hi[i], BLUE, opacity, max_points=ADMIN_MAX_POINTS)
        plot.polyline(seconds, measured[i], GREY, 1.0, max_points=ADMIN_MAX_POINTS)
        plot.polyline(seconds, simulated[i], BLUE, 1.5, max_points=ADMIN_MAX_POINTS)
        plot.time_axis(x, times, "date")
        plot.y_axis(y, "temperature (°C)")
        panels.append(plot)

    bottom = Plot("Residuals (simulated − measured)", "", width=PANEL_WIDTH, left=64, right=20)
    bottom.legend(*[(RESIDUAL_COLORS[i], f"T{i + 1}, RMSE {_rmse(residual[i])} °C", "line") for i in range(SENSORS)])
    bottom.plot_height(150)
    x = bottom.x_scale(low, high)
    finite = residual[np.isfinite(residual)]
    extent = float(np.max(np.abs(finite))) * 1.1 if finite.size else 1.0
    y = bottom.y_scale(-(extent or 1.0), extent or 1.0)
    bottom.reference_line(0.0, "", "y", GREY)
    for i in range(SENSORS):
        bottom.polyline(seconds, residual[i], RESIDUAL_COLORS[i], 1.0, max_points=ADMIN_MAX_POINTS)
    bottom.time_axis(x, times, "date")
    bottom.y_axis(y, "residual (°C)")
    panels.append(bottom)

    name = str(describe.data["name"].value)
    if bands:
        bands_text = f"{bands[0][0].split(' ', 1)[0]} % credible and predictive bands"
    else:
        bands_text = "no band: the fields were not stored (store_fields)" if inversion else "no band: a forward model has no interval"
    period = f"{format_time(times[0])} to {format_time(times[-1])}"
    subtitle = f"{best}, T1–T3, {period}; {bands_text}"
    figure = Figure(f"Measured and simulated temperature — {name}",
                    f"Measured and simulated temperature at thermometers T1 to T3 ({MODEL[role_of(describe)]}, {best}) and their residuals, "
                    f"from {period}. Admin figure with measurements: never published.",
                    panels, subtitle=subtitle)
    return figure.to_bytes()


def fields_figures(fields: Answer, describe: Answer, role: str) -> list[tuple[str, bytes]]:
    """`fields-temperature.svg` et `fields-water_flux.svg` : les figures de champs du rapport (F1), tracées d'après les
    champs du meilleur ajustement (réponse `fields`) et la description du twin."""
    published = {f"{role}-fields.npz": fields, f"{role}-describe.json": describe}
    built = []
    for name, make in (("fields-temperature.svg", temperature_field_figure), ("fields-water_flux.svg", water_flux_field_figure)):
        figure = make(published, role)
        if figure is not None:
            built.append((name, figure[1]))
    return built
