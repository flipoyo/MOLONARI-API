"""Primitives de tracé SVG du rapport (P2c.3f) : échelles, graduations, axes, légende.

Le backend n'a pas le droit d'importer matplotlib (tests/test_architecture.py). Ce module écrit le SVG à la main,
avec des nombres à largeur fixe et sans identifiant ni date : mêmes octets d'une génération à l'autre. La palette
est celle d'Okabe et Ito, lisible par les daltoniens.
"""

from __future__ import annotations

import math
from xml.sax.saxutils import escape

INK, GREY, LIGHT = "#222222", "#666666", "#dddddd"
BLUE, ORANGE = "#0072B2", "#E69F00"
STYLE = "font-family:sans-serif;font-size:11px"
CHAR_WIDTH = 6.0
"""Largeur estimée d'un caractère (px), pour placer les entrées de la légende sans mesurer le texte."""


def num(value: float) -> str:
    return f"{value:.2f}"


def nice_ticks(low: float, high: float, target: int = 5) -> list[float]:
    """Graduations à pas 1, 2 ou 5 × 10^k, comprises dans [low, high]."""
    if not high > low:
        return [low]
    raw = (high - low) / target
    power = 10 ** math.floor(math.log10(raw))
    step = next(m * power for m in (1, 2, 5, 10) if m * power >= raw)
    first = math.ceil(low / step - 1e-9)
    last = math.floor(high / step + 1e-9)
    return [round(k * step, 15) for k in range(first, last + 1)]


def tick_label(value: float, ticks: list[float]) -> str:
    """Étiquette d'une graduation : notation scientifique hors de [1e-3, 1e5[, sinon décimales du pas."""
    if value == 0:
        return "0"
    if not 1e-3 <= abs(value) < 1e5:
        return f"{value:.2e}"
    step = abs(ticks[1] - ticks[0]) if len(ticks) > 1 else abs(value)
    decimals = max(0, -math.floor(math.log10(step) + 1e-9))
    return f"{value:.{decimals}f}"


class Scale:
    """Application affine d'un intervalle de données vers un intervalle de pixels."""

    def __init__(self, low: float, high: float, start: float, end: float):
        self.low, self.high, self.start, self.end = low, high, start, end

    def __call__(self, value: float) -> float:
        return self.start + (float(value) - self.low) / (self.high - self.low) * (self.end - self.start)


class Plot:
    """Un cadre : titre, sous-titre, légende, zone de tracé entre des marges, axes gradués."""

    def __init__(self, title: str, subtitle: str, width: int = 480, height: int = 220,
                 left: int = 56, top: int = 64, right: int = 20, bottom: int = 44):
        self.title, self.subtitle = title, subtitle
        self.width, self.height = width, height
        self.left, self.top = left, top
        self.right, self.bottom = width - right, height - bottom
        self._body: list[str] = []
        self._legend: list[tuple[str, str]] = []

    def x_scale(self, low: float, high: float) -> Scale:
        return Scale(low, high, self.left, self.right)

    def y_scale(self, low: float, high: float) -> Scale:
        return Scale(low, high, self.bottom, self.top)

    def add(self, markup: str) -> None:
        self._body.append(markup)

    def x_axis(self, scale: Scale, label: str) -> None:
        """Axe horizontal en bas de la zone de tracé, avec graduations, traits de repère et unité."""
        ticks = nice_ticks(scale.low, scale.high)
        self.add(f'<line x1="{self.left}" y1="{self.bottom}" x2="{self.right}" y2="{self.bottom}" stroke="{GREY}"/>')
        for tick in ticks:
            px = num(scale(tick))
            self.add(f'<line x1="{px}" y1="{self.top}" x2="{px}" y2="{self.bottom}" stroke="{LIGHT}"/>')
            self.add(f'<line x1="{px}" y1="{self.bottom}" x2="{px}" y2="{self.bottom + 4}" stroke="{GREY}"/>')
            self.add(f'<text x="{px}" y="{self.bottom + 16}" text-anchor="middle" fill="{GREY}">{tick_label(tick, ticks)}</text>')
        self.add(f'<text x="{num((self.left + self.right) / 2)}" y="{self.bottom + 34}" text-anchor="middle" fill="{INK}">{escape(label)}</text>')

    def y_axis(self, scale: Scale, label: str) -> None:
        """Axe vertical à gauche de la zone de tracé, avec graduations, lignes de repère et unité."""
        ticks = nice_ticks(scale.low, scale.high)
        self.add(f'<line x1="{self.left}" y1="{self.top}" x2="{self.left}" y2="{self.bottom}" stroke="{GREY}"/>')
        for tick in ticks:
            py = num(scale(tick))
            self.add(f'<line x1="{self.left}" y1="{py}" x2="{self.right}" y2="{py}" stroke="{LIGHT}"/>')
            self.add(f'<text x="{self.left - 6}" y="{num(scale(tick) + 4)}" text-anchor="end" fill="{GREY}">{tick_label(tick, ticks)}</text>')
        self.add(f'<text x="{self.left}" y="{self.top - 6}" fill="{INK}">{escape(label)}</text>')

    def legend(self, *entries: tuple[str, str]) -> None:
        """Entrées (couleur, libellé) alignées sous le sous-titre."""
        self._legend = list(entries)

    def to_bytes(self) -> bytes:
        head = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.width}" height="{self.height}" viewBox="0 0 {self.width} {self.height}" style="{STYLE}">',
            f'<rect width="{self.width}" height="{self.height}" fill="#ffffff"/>',
            f'<text x="{self.left}" y="18" font-weight="bold" fill="{INK}">{escape(self.title)}</text>',
            f'<text x="{self.left}" y="32" fill="{GREY}">{escape(self.subtitle)}</text>',
        ]
        x = float(self.left)
        for color, label in self._legend:
            head.append(f'<rect x="{num(x)}" y="40" width="10" height="10" fill="{color}"/>')
            head.append(f'<text x="{num(x + 14)}" y="49" fill="{INK}">{escape(label)}</text>')
            x += 14 + len(label) * CHAR_WIDTH + 14
        return ("\n".join([*head, *self._body, "</svg>"]) + "\n").encode("utf-8")
