"""Figures du rapport (P2c.3f) : des SVG dérivés des seules entrées publiées, écrits sans bibliothèque graphique.

Le backend n'a pas le droit d'importer matplotlib (tests/test_architecture.py) : les figures sont du SVG écrit à la
main, à partir de la réponse déjà projetée. Elles ne contiennent donc rien que l'entrée ne contienne pas. Les nombres
sont formatés à largeur fixe : mêmes octets d'une génération à l'autre.

Libellés en anglais, sans phrase longue : le rapport est lu par la gate, qui traduit autour.
"""

from __future__ import annotations

from xml.sax.saxutils import escape

import numpy as np

from ..questions.answer import Answer

WIDTH, HEIGHT = 480, 160
MARGIN = 40
STYLE = "font-family:sans-serif;font-size:11px"
INK, GREY, ACCENT = "#222222", "#999999", "#1f6fb2"

FORWARD = "forward model, no confidence interval"


def _svg(title: str, subtitle: str, body: list[str]) -> bytes:
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}" style="{STYLE}">',
        f'<rect width="{WIDTH}" height="{HEIGHT}" fill="#ffffff"/>',
        f'<text x="{MARGIN}" y="18" font-weight="bold" fill="{INK}">{escape(title)}</text>',
        f'<text x="{MARGIN}" y="32" fill="{GREY}">{escape(subtitle)}</text>',
        *body,
        "</svg>",
    ]
    return ("\n".join(lines) + "\n").encode("utf-8")


def _text(data: dict, name: str):
    return str(data[name].value) if name in data else ""


def flux_figure(answer: Answer) -> bytes:
    """Flux moyen au fond de la rivière (m/s) : point central, intervalle a posteriori à 90 % s'il existe (modèle
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
    low, high = low - pad, high + pad

    def x(value: float) -> str:
        return f"{MARGIN + (float(value) - low) / (high - low) * (WIDTH - 2 * MARGIN):.2f}"

    y = 90
    body = [
        f'<line x1="{MARGIN}" y1="{y + 30}" x2="{WIDTH - MARGIN}" y2="{y + 30}" stroke="{GREY}"/>',
        f'<line x1="{x(data["p5"].value)}" y1="{y}" x2="{x(data["p95"].value)}" y2="{y}" stroke="{GREY}" stroke-width="2"/>',
    ]
    if mean.interval is not None:
        body.append(f'<line x1="{x(mean.interval[0])}" y1="{y}" x2="{x(mean.interval[1])}" y2="{y}" stroke="{ACCENT}" stroke-width="6"/>')
    body += [
        f'<circle cx="{x(mean.value)}" cy="{y}" r="5" fill="{INK}"/>',
        f'<text x="{MARGIN}" y="{y + 46}" fill="{GREY}">{low:.3e}</text>',
        f'<text x="{WIDTH - MARGIN}" y="{y + 46}" text-anchor="end" fill="{GREY}">{high:.3e}</text>',
        f'<text x="{WIDTH // 2}" y="{y + 46}" text-anchor="middle" fill="{GREY}">q (m/s), mean {float(mean.value):.3e}</text>',
    ]
    if _text(data, "model") == "forward":
        subtitle = FORWARD
    elif mean.interval is not None:
        subtitle = f"inversion model, {float(data['level'].value) * 100:g} % posterior interval"
    else:
        subtitle = "inversion model"
    return _svg("Water flux through the riverbed", subtitle, body)


def rmse_figure(answer: Answer) -> bytes:
    """RMSE de l'écart de température aux trois thermomètres (°C), puis RMSE totale."""
    data = answer.data
    names = [str(s) for s in np.asarray(data["sensor"].value)] + ["total"]
    values = [float(v) for v in np.asarray(data["rmse"].value)] + [float(data["rmse_total"].value)]
    top = max(max(values), 1e-12)
    slot = (WIDTH - 2 * MARGIN) / len(values)
    base, height = HEIGHT - 28, HEIGHT - 28 - 50
    body = [f'<line x1="{MARGIN}" y1="{base}" x2="{WIDTH - MARGIN}" y2="{base}" stroke="{GREY}"/>']
    for i, (name, value) in enumerate(zip(names, values, strict=True)):
        bar = value / top * height
        left = MARGIN + i * slot + slot * 0.2
        body += [
            f'<rect x="{left:.2f}" y="{base - bar:.2f}" width="{slot * 0.6:.2f}" height="{bar:.2f}" fill="{ACCENT}"/>',
            f'<text x="{left + slot * 0.3:.2f}" y="{base + 14}" text-anchor="middle" fill="{INK}">{escape(name)}</text>',
            f'<text x="{left + slot * 0.3:.2f}" y="{base - bar - 4:.2f}" text-anchor="middle" fill="{INK}">{value:.3f}</text>',
        ]
    subtitle = FORWARD if _text(data, "model") == "forward" else "inversion model, best sample"
    return _svg("Temperature misfit (RMSE, °C)", subtitle, body)


FIGURES = {"water_flux": flux_figure, "fit_quality": rmse_figure}
"""Question → fonction qui dessine sa figure. Une entrée dont la question n'est pas ici n'a pas de figure."""
