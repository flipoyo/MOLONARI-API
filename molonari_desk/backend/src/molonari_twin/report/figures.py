"""Figures du rapport (P2c.3f) : des SVG dérivés des seules entrées publiées, écrits sans bibliothèque graphique.

Les figures sont dessinées à partir de la réponse déjà projetée (voir `svgplot.py` pour les primitives) : elles ne
contiennent donc rien que l'entrée ne contienne pas. Les données de chaque figure sont celles de l'entrée dont elle
dérive, publiée à côté : un lecteur peut refaire une figure de publication sans toucher au rapport.

Libellés en anglais, sans phrase longue : le rapport est lu par la gate, qui traduit autour.
"""

from __future__ import annotations

from xml.sax.saxutils import escape

import numpy as np

from ..questions.answer import Answer
from .svgplot import BLUE, GREY, INK, ORANGE, Plot, num

FORWARD = "forward model, no confidence interval"


def _text(data: dict, name: str):
    return str(data[name].value) if name in data else ""


def flux_figure(answer: Answer) -> bytes:
    """Flux moyen au fond de la rivière (m/s) : point central, intervalle a posteriori s'il existe (modèle
    d'inversion seulement), et étendue de q(t) entre ses percentiles 5 et 95."""
    data = answer.data
    mean = data["mean"]
    marks = [float(mean.value), float(data["p5"].value), float(data["p95"].value)]
    if mean.interval is not None:
        marks += [float(mean.interval[0]), float(mean.interval[1])]
    low, high = min(marks), max(marks)
    if high == low:
        low, high = low - 1e-9, high + 1e-9
    pad = (high - low) * 0.05
    plot = Plot("Water flux through the riverbed", "", height=200)
    x = plot.x_scale(low - pad, high + pad)

    if _text(data, "model") == "forward":
        plot.subtitle = FORWARD
    elif mean.interval is not None:
        plot.subtitle = f"inversion model, {float(data['level'].value) * 100:g} % posterior interval"
    else:
        plot.subtitle = "inversion model"
    legend = [(INK, f"mean {float(mean.value):.3e}"), (GREY, "q(t) 5th to 95th percentile")]
    if mean.interval is not None:
        legend.insert(1, (BLUE, f"{float(data['level'].value) * 100:g} % posterior interval"))
    plot.legend(*legend)

    plot.x_axis(x, "q (m/s)")
    row = num((plot.top + plot.bottom) / 2)
    plot.add(f'<line x1="{num(x(data["p5"].value))}" y1="{row}" x2="{num(x(data["p95"].value))}" y2="{row}" stroke="{GREY}" stroke-width="2"/>')
    if mean.interval is not None:
        plot.add(f'<line x1="{num(x(mean.interval[0]))}" y1="{row}" x2="{num(x(mean.interval[1]))}" y2="{row}" stroke="{BLUE}" stroke-width="6"/>')
    plot.add(f'<circle cx="{num(x(mean.value))}" cy="{row}" r="5" fill="{INK}"/>')
    return plot.to_bytes()


def rmse_figure(answer: Answer) -> bytes:
    """RMSE de l'écart de température aux trois thermomètres (°C), puis RMSE totale."""
    data = answer.data
    names = [str(s) for s in np.asarray(data["sensor"].value)] + ["total"]
    values = [float(v) for v in np.asarray(data["rmse"].value)] + [float(data["rmse_total"].value)]
    top = max(max(values), 1e-12)
    plot = Plot("Temperature misfit (RMSE)", FORWARD if _text(data, "model") == "forward" else "inversion model, best sample")
    y = plot.y_scale(0.0, top * 1.1)
    plot.y_axis(y, "RMSE (°C)")
    slot = (plot.right - plot.left) / len(values)
    for i, (name, value) in enumerate(zip(names, values, strict=True)):
        left = plot.left + i * slot + slot * 0.2
        centre = left + slot * 0.3
        color = ORANGE if name == "total" else BLUE
        plot.add(f'<rect x="{num(left)}" y="{num(y(value))}" width="{num(slot * 0.6)}" height="{num(plot.bottom - y(value))}" fill="{color}"/>')
        plot.add(f'<text x="{num(centre)}" y="{num(y(value) - 4)}" text-anchor="middle" fill="{INK}">{value:.3f}</text>')
        plot.add(f'<text x="{num(centre)}" y="{plot.bottom + 16}" text-anchor="middle" fill="{INK}">{escape(name)}</text>')
    return plot.to_bytes()


FIGURES = {"water_flux": flux_figure, "fit_quality": rmse_figure}
"""Question → fonction qui dessine sa figure. Une entrée dont la question n'est pas ici n'a pas de figure."""
