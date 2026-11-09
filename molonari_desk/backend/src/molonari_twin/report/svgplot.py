"""Primitives de tracé SVG du rapport (P2c.3f, F1) : échelles, axes, marques, cartes de chaleur, panneaux.

Le backend n'a pas le droit d'importer matplotlib (tests/test_architecture.py). Ce module écrit le SVG à la main,
avec des nombres à largeur fixe (`num`), sans identifiant ni date ni locale : mêmes octets d'une génération à l'autre.
Les palettes sont écrites en dur : Okabe et Ito pour les séries, viridis (séquentielle) et une divergente bleu–blanc–
rouge centrée sur 0, lisibles par les daltoniens.

Ordre d'emploi d'un `Plot` : `legend()` d'abord (elle fixe la marge du haut), puis les échelles (`x_scale`, `y_scale`),
les marques, enfin les axes. Un `Figure` empile ou juxtapose plusieurs `Plot` dans un seul SVG.

Les exposants s'écrivent `^{-9}` dans les libellés : ils deviennent un `<tspan baseline-shift="super">` (les caractères
Unicode en exposant manquent dans certaines polices `sans-serif`). Les axes écrivent une fois le facteur commun
`× 10^{n}` quand leurs graduations sortent de [1e-3, 1e5[ (`axis_factor`).
"""

from __future__ import annotations

import math
import re
from xml.sax.saxutils import escape

import numpy as np

INK, GREY, LIGHT = "#222222", "#666666", "#dddddd"
BLUE, ORANGE, VERMILLION, GREEN, SKY, PURPLE, YELLOW = "#0072B2", "#E69F00", "#D55E00", "#009E73", "#56B4E9", "#CC79A7", "#F0E442"
SERIES = (BLUE, ORANGE, GREEN, VERMILLION, SKY, PURPLE, YELLOW)
"""Palette d'Okabe et Ito, dans l'ordre où elle sert aux séries (le jaune, peu contrasté, en dernier)."""

VIRIDIS = (
    "#440154", "#481a6c", "#472f7d", "#414487", "#39568c", "#31688e", "#2a788e", "#23888e",
    "#1f988b", "#22a884", "#35b779", "#54c568", "#7ad151", "#a5db36", "#d2e21b", "#fde725",
)
"""16 niveaux de viridis (séquentielle)."""

DIVERGING = (
    "#2166ac", "#327db8", "#4393c3", "#6bacd1", "#92c5de", "#b2d5e7", "#d1e5f0", "#e8f2f8", "#ffffff",
    "#feede3", "#fddbc7", "#f9c0a5", "#f4a582", "#e58368", "#d6604d", "#c43c3c", "#b2182b",
)
"""17 niveaux, nombre impair : bleu (négatif) – blanc (indice 8, zéro) – rouge (positif). Les neuf couleurs
d'ancrage (indices pairs) sont celles de la palette divergente RdBu de ColorBrewer, centre blanc pur ; les autres sont
les milieux (RGB) de leurs voisines. Lisible par les daltoniens."""
CENTRE = (len(DIVERGING) - 1) // 2

FONT_SIZE, TITLE_SIZE, EXPONENT_SIZE = 12, 13, 9
STYLE = f"font-family:sans-serif;font-size:{FONT_SIZE}px"
CHAR_WIDTH = 6.6
"""Largeur estimée d'un caractère à 12 px (px), pour placer les entrées de la légende sans mesurer le texte."""
LEGEND_ROW = 16  # hauteur d'une ligne de légende (px)
LEGEND_TOP = 42  # ordonnée du haut de la première ligne de légende
TICK_LENGTH = 4  # graduations vers l'extérieur (px)

GRID_COLUMNS, GRID_ROWS = 180, 60
"""Grille d'affichage maximale d'une carte de chaleur : au plus 180 × 60 = 10 800 rectangles."""
MAX_POINTS = 200
"""Points au plus par courbe et par panneau : une série plus longue est décimée."""
DAY = 86400
MAX_TIME_TICKS = 8


def num(value: float) -> str:
    return f"{value:.2f}"


def minus(text: str) -> str:
    """Signe moins typographique à la place du trait d'union (graduations, exposants)."""
    return "−" + text[1:] if text.startswith("-") else text


_EXPONENT = re.compile(r"\^\{([^}]*)\}")


def rich(text: str) -> str:
    """Texte SVG d'un libellé : échappé, avec `^{…}` en exposant (`<tspan baseline-shift="super">`)."""
    out, last = [], 0
    for found in _EXPONENT.finditer(text):
        out.append(escape(text[last : found.start()]))
        out.append(f'<tspan baseline-shift="super" font-size="{EXPONENT_SIZE}">{escape(found.group(1).replace("-", chr(0x2212)))}</tspan>')
        last = found.end()
    out.append(escape(text[last:]))
    return "".join(out)


def rich_number(value: float, significant: int = 3) -> str:
    """Un nombre pour un libellé : décimales fixes dans [1e-3, 1e5[, sinon `m × 10^{e}` (voir `rich`)."""
    if value == 0 or not math.isfinite(value):
        return "0" if value == 0 else str(value)
    exponent = math.floor(math.log10(abs(value)))
    if -3 <= exponent < 5:
        return minus(f"{value:.{max(0, significant - 1 - exponent)}f}")
    mantissa = f"{value / 10**exponent:.{significant - 1}f}"
    if abs(float(mantissa)) >= 10:
        exponent += 1
        mantissa = f"{value / 10**exponent:.{significant - 1}f}"
    return f"{minus(mantissa)} × 10^{{{exponent}}}"


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


def axis_factor(ticks: list[float]) -> tuple[int, list[str]]:
    """(exposant, étiquettes) d'un axe. Quand une graduation non nulle sort de [1e-3, 1e5[, toutes sont divisées par
    10^exposant (celui de la plus grande) et écrites en décimales courtes (`-2`, `0`, `2`) : le facteur `× 10^{exposant}`
    est écrit une seule fois par l'axe. Sinon l'exposant est 0 et les étiquettes sont celles de `tick_label`."""
    nonzero = [abs(t) for t in ticks if t != 0]
    if not nonzero or all(1e-3 <= t < 1e5 for t in nonzero):
        return 0, [tick_label(t, ticks) for t in ticks]
    exponent = math.floor(math.log10(max(nonzero)))
    scaled = [round(t / 10**exponent, 12) for t in ticks]
    step = abs(scaled[1] - scaled[0]) if len(scaled) > 1 else 1.0
    decimals = max(0, -math.floor(math.log10(step) + 1e-9))
    return exponent, ["0" if t == 0 else f"{t:.{decimals}f}" for t in scaled]


def factor_text(exponent: int) -> str:
    return f"× 10^{{{exponent}}}" if exponent else ""


def with_factor(label: str, exponent: int) -> str:
    """Le libellé d'un axe avec son facteur : `{factor}` y est remplacé, ou le facteur est ajouté à la fin."""
    factor = factor_text(exponent)
    if "{factor}" in label:
        return re.sub(r" {2,}", " ", label.replace("{factor}", factor)).replace("( ", "(")
    return f"{label} {factor}".rstrip()


class Scale:
    """Application affine d'un intervalle de données vers un intervalle de pixels (inversée si start > end)."""

    def __init__(self, low: float, high: float, start: float, end: float):
        self.low, self.high, self.start, self.end = low, high, start, end

    def __call__(self, value):
        scaled = self.start + (np.asarray(value, dtype=float) - self.low) / (self.high - self.low) * (self.end - self.start)
        return float(scaled) if scaled.ndim == 0 else scaled


def sample_indices(count: int, wanted: int) -> np.ndarray:
    """Au plus `wanted` indices régulièrement espacés dans [0, count[, premier et dernier compris (déterministe)."""
    if count <= wanted:
        return np.arange(count)
    return np.unique(np.floor(np.linspace(0, count - 1, wanted) + 0.5).astype(int))


def sample_colors(palette: tuple[str, ...], count: int) -> list[str]:
    """`count` couleurs régulièrement choisies dans la palette (de la première à la dernière)."""
    return [palette[i] for i in sample_indices(len(palette), count)] if count <= len(palette) else list(palette)


def decimate(xs, ys) -> tuple[np.ndarray, np.ndarray]:
    """La série (xs, ys) réduite à `MAX_POINTS` points au plus (pas régulier sur l'indice, premier et dernier gardés)."""
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    keep = sample_indices(len(xs), MAX_POINTS)
    return xs[keep], ys[keep]


def sequential_levels(values, low: float, high: float, levels: int) -> np.ndarray:
    """Niveau (0 … levels−1) de chaque valeur entre `low` et `high`, en classes égales ; −1 pour une valeur non finie."""
    values = np.asarray(values, float)
    span = high - low if high > low else 1.0
    index = np.clip(np.floor((np.where(np.isfinite(values), values, low) - low) / span * levels), 0, levels - 1).astype(int)
    return np.where(np.isfinite(values), index, -1)


def diverging_levels(values, high: float | None = None, levels: int = len(DIVERGING)) -> np.ndarray:
    """Niveau (0 … levels−1) de chaque valeur sur la palette divergente : `centre + round(centre · q / high)`, avec
    `high = max |q|` (1 si le champ est nul). q = 0 tombe sur le centre (blanc) ; le signe de q est celui du côté de la
    palette. Une valeur non finie donne −1."""
    values = np.asarray(values, float)
    centre = (levels - 1) // 2
    finite = np.isfinite(values)
    if high is None:
        high = float(np.max(np.abs(values[finite]))) if finite.any() else 0.0
    high = high if high > 0 else 1.0
    index = np.clip(centre + np.rint(centre * np.where(finite, values, 0.0) / high), 0, levels - 1).astype(int)
    return np.where(finite, index, -1)


def block_mean(values, rows: int = GRID_ROWS, columns: int = GRID_COLUMNS) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Moyenne par bloc ramenant (R, C) à (min(R, rows), min(C, columns)) : (valeurs, bornes des lignes, bornes des
    colonnes) en indices du tableau d'origine. Les valeurs non finies sont ignorées ; un bloc qui n'en a que des non
    finies donne nan. Déterministe (indices entiers)."""
    values = np.asarray(values, float)
    height, width = values.shape
    r, c = min(height, rows), min(width, columns)
    row_bounds = np.arange(r + 1) * height // r
    column_bounds = np.arange(c + 1) * width // c
    finite = np.isfinite(values)
    sums = np.add.reduceat(np.add.reduceat(np.where(finite, values, 0.0), row_bounds[:-1], axis=0), column_bounds[:-1], axis=1)
    counts = np.add.reduceat(np.add.reduceat(finite.astype(float), row_bounds[:-1], axis=0), column_bounds[:-1], axis=1)
    mean = np.where(counts > 0, sums / np.maximum(counts, 1.0), np.nan)
    return mean, row_bounds, column_bounds


def edges_from_centers(centers) -> np.ndarray:
    """Bords des cellules dont `centers` sont les centres : milieux entre voisins, prolongés d'une demi-maille aux bouts."""
    centers = np.asarray(centers, float)
    if centers.size == 1:
        return np.array([centers[0] - 0.5, centers[0] + 0.5])
    middles = (centers[:-1] + centers[1:]) / 2
    return np.concatenate([[centers[0] - (middles[0] - centers[0])], middles, [centers[-1] + (centers[-1] - middles[-1])]])


def _segments(xs: np.ndarray, ys: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    """Les tronçons de points consécutifs finis : un point non fini coupe la ligne."""
    finite = np.isfinite(xs) & np.isfinite(ys)
    cuts = np.flatnonzero(np.diff(finite.astype(np.int8))) + 1
    pieces = np.split(np.arange(len(xs)), cuts)
    return [(xs[p], ys[p]) for p in pieces if p.size and finite[p[0]]]


class Plot:
    """Un cadre : titre, sous-titre, légende, zone de tracé entre des marges, axes gradués et marques.

    `legend()` doit être appelée avant toute échelle ou tout axe : elle fixe `top` (la marge du haut grandit d'une
    ligne par ligne de légende), que les axes lisent dès l'appel. `RuntimeError` sinon (erreur de programmation)."""

    def __init__(self, title: str, subtitle: str = "", width: int = 480, height: int = 220,
                 left: int = 56, top: int = 60, right: int = 20, bottom: int = 44, desc: str = ""):
        self.title, self.subtitle, self.desc = title, subtitle, desc
        self.width, self.height = width, height
        self.left, self.top = left, top
        self.right, self.bottom = width - right, height - bottom
        self.margin_bottom = bottom
        self.x: Scale | None = None
        self.y: Scale | None = None
        self.legend_items: list[tuple[float, float, str, str, str]] = []  # (x, haut de la ligne, couleur, libellé, genre)
        self.legend_rows = 0
        self._grid: list[str] = []
        self._body: list[str] = []
        self._axes: list[str] = []
        self._scaled = False

    def plot_height(self, inner: float) -> None:
        """Hauteur de la zone de tracé (px), la légende étant posée : le cadre grandit d'autant. À appeler avant les échelles."""
        if self._scaled:
            raise RuntimeError("plot_height() must be called before any scale or axis")
        self.bottom = self.top + inner
        self.height = self.bottom + self.margin_bottom

    # Échelles -----------------------------------------------------------------------------------
    def x_scale(self, low: float, high: float) -> Scale:
        self._scaled = True
        self.x = Scale(low, high, self.left, self.right)
        return self.x

    def y_scale(self, low: float, high: float, down: bool = False) -> Scale:
        """Échelle verticale : `low` en bas (haut si `down` : profondeur vers le bas)."""
        self._scaled = True
        self.y = Scale(low, high, self.top, self.bottom) if down else Scale(low, high, self.bottom, self.top)
        return self.y

    def add(self, markup: str) -> None:
        self._body.append(markup)

    def text(self, x: float, y: float, content: str, anchor: str = "start", fill: str = GREY, halo: bool = False) -> None:
        """Texte libre en pixels (`^{…}` en exposant). `halo` : un contour blanc le détache de ce qu'il recouvre."""
        outline = ' stroke="#ffffff" stroke-width="3" paint-order="stroke"' if halo else ""
        self._body.append(f'<text x="{num(x)}" y="{num(y)}" text-anchor="{anchor}" fill="{fill}"{outline}>{rich(content)}</text>')

    # Axes ---------------------------------------------------------------------------------------
    def _x_ticks(self, scale: Scale, ticks: list[float], labels: list[str]) -> None:
        self._axes.append(f'<line x1="{self.left}" y1="{self.bottom}" x2="{self.right}" y2="{self.bottom}" stroke="{GREY}"/>')
        for tick, label in zip(ticks, labels, strict=True):
            px = num(scale(tick))
            self._grid.append(f'<line x1="{px}" y1="{self.top}" x2="{px}" y2="{self.bottom}" stroke="{LIGHT}" stroke-width="0.5"/>')
            self._axes.append(f'<line x1="{px}" y1="{self.bottom}" x2="{px}" y2="{self.bottom + TICK_LENGTH}" stroke="{GREY}"/>')
            self._axes.append(f'<text x="{px}" y="{self.bottom + TICK_LENGTH + 13}" text-anchor="middle" fill="{GREY}">{escape(minus(label))}</text>')

    def _x_label(self, label: str) -> None:
        self._axes.append(f'<text x="{num((self.left + self.right) / 2)}" y="{self.bottom + 38}" text-anchor="middle" fill="{INK}">{rich(label)}</text>')

    def x_axis(self, scale: Scale, label: str) -> None:
        """Axe horizontal en bas de la zone de tracé : graduations vers l'extérieur, grille discrète et libellé (`{factor}`
        y place le facteur commun `× 10^{n}`, sinon il est ajouté à la fin)."""
        self._guard()
        ticks = nice_ticks(scale.low, scale.high, max(4, round((self.right - self.left) / 90)))
        exponent, labels = axis_factor(ticks)
        self._x_ticks(scale, ticks, labels)
        self._x_label(with_factor(label, exponent))

    def y_axis(self, scale: Scale, label: str) -> None:
        """Axe vertical à gauche de la zone de tracé (échelle inversée admise : profondeur vers le bas)."""
        self._guard()
        ticks = nice_ticks(scale.low, scale.high, max(3, round((self.bottom - self.top) / 45)))
        exponent, labels = axis_factor(ticks)
        self._axes.append(f'<line x1="{self.left}" y1="{self.top}" x2="{self.left}" y2="{self.bottom}" stroke="{GREY}"/>')
        for tick, tick_text in zip(ticks, labels, strict=True):
            py = num(scale(tick))
            self._grid.append(f'<line x1="{self.left}" y1="{py}" x2="{self.right}" y2="{py}" stroke="{LIGHT}" stroke-width="0.5"/>')
            self._axes.append(f'<line x1="{self.left - TICK_LENGTH}" y1="{py}" x2="{self.left}" y2="{py}" stroke="{GREY}"/>')
            self._axes.append(f'<text x="{self.left - TICK_LENGTH - 3}" y="{num(scale(tick) + 4)}" text-anchor="end" fill="{GREY}">{escape(minus(tick_text))}</text>')
        if label or exponent:
            self._axes.append(f'<text x="{self.left}" y="{self.top - 8}" fill="{INK}">{rich(with_factor(label, exponent))}</text>')

    def time_axis(self, scale: Scale, times, label: str = "date") -> None:
        """Axe du temps en dates : `scale` est en secondes depuis l'époque, `times` (datetime64[s]) dit les années.
        Graduations aux minuits, au pas de 1, 2 ou 7 jours (puis des multiples de 7) : le plus petit qui en donne au
        plus 8. Étiquettes écrites à partir des entiers de la date (jamais de `strftime`, qui dépend de la locale) :
        `MM-JJ` avec l'année dans le libellé quand elle est unique, `AAAA-MM-JJ` sinon."""
        self._guard()
        times = np.asarray(times).astype("datetime64[s]")
        years = {int(str(times.min().astype("datetime64[Y]"))), int(str(times.max().astype("datetime64[Y]")))}
        first, last = math.ceil(scale.low / DAY), math.floor(scale.high / DAY)
        step = next((s for s in (1, 2, 7) if (last - first) // s + 1 <= MAX_TIME_TICKS), None)
        if step is None:
            weeks = math.ceil(((last - first) / (MAX_TIME_TICKS - 1)) / 7)
            step = 7 * weeks
        days = list(range(first, last + 1, step)) or [math.floor(scale.low / DAY)]
        labels = []
        for day in days:
            date = np.datetime64(day, "D").astype(object)
            labels.append(f"{date.month:02d}-{date.day:02d}" if len(years) == 1 else f"{date.year:04d}-{date.month:02d}-{date.day:02d}")
        self._x_ticks(scale, [d * DAY for d in days], labels)
        self._x_label(f"{label} ({years.copy().pop()})" if len(years) == 1 else label)

    def _guard(self) -> None:
        if self.x is None and self.y is None:
            raise RuntimeError("an axis needs a scale: call x_scale() or y_scale() first")

    # Légende ------------------------------------------------------------------------------------
    def legend(self, *entries: tuple[str, ...]) -> None:
        """Entrées (couleur, libellé[, genre]) sous le sous-titre ; genre : `box` (défaut), `line`, `dash` ou `dot`.
        Une entrée qui dépasserait `right` passe à la ligne suivante, et la marge du haut grandit d'autant. La largeur
        d'une entrée est estimée : `len(libellé) × CHAR_WIDTH + 28`. À appeler avant toute échelle."""
        if self._scaled:
            raise RuntimeError("legend() must be called before any scale or axis: it sets the top margin")
        self.legend_items, row, x = [], 0, float(self.left)
        for entry in entries:
            color, label = entry[0], entry[1]
            kind = entry[2] if len(entry) > 2 else "box"
            width = len(label) * CHAR_WIDTH + 28
            if x > self.left and x + width > self.right:
                row, x = row + 1, float(self.left)
            self.legend_items.append((x, LEGEND_TOP + row * LEGEND_ROW, color, label, kind))
            x += width
        self.legend_rows = row + 1 if entries else 0
        if self.legend_rows:
            self.top = max(self.top, LEGEND_TOP + self.legend_rows * LEGEND_ROW + 20)

    # Marques ------------------------------------------------------------------------------------
    def _scales(self) -> tuple[Scale, Scale]:
        if self.x is None or self.y is None:
            raise RuntimeError("a mark needs both scales: call x_scale() and y_scale() first")
        return self.x, self.y

    def polyline(self, xs, ys, color: str, width: float = 1.5, dash: str | None = None) -> None:
        """Courbe en données : décimée à `MAX_POINTS` points ; un point non fini coupe la ligne (un `<polyline>` par
        tronçon, un point isolé devient un petit disque)."""
        sx, sy = self._scales()
        xs, ys = decimate(xs, ys)
        extra = f' stroke-dasharray="{dash}"' if dash else ""
        for seg_x, seg_y in _segments(xs, ys):
            px, py = sx(seg_x), sy(seg_y)
            if len(px) == 1:
                self.add(f'<circle cx="{num(px[0])}" cy="{num(py[0])}" r="1.5" fill="{color}"/>')
                continue
            points = " ".join(f"{num(a)},{num(b)}" for a, b in zip(px, py, strict=True))
            self.add(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="{width:g}" stroke-linejoin="round"{extra}/>')

    def band(self, xs, lows, highs, color: str, opacity: float = 0.25) -> None:
        """Aire entre deux courbes (un `<path>` fermé), décimée comme `polyline` ; les points non finis sont ignorés."""
        sx, sy = self._scales()
        xs, lows, highs = np.asarray(xs, float), np.asarray(lows, float), np.asarray(highs, float)
        keep = sample_indices(len(xs), MAX_POINTS)
        xs, lows, highs = xs[keep], lows[keep], highs[keep]
        finite = np.isfinite(xs) & np.isfinite(lows) & np.isfinite(highs)
        if finite.sum() < 2:
            return
        px, low, high = sx(xs[finite]), sy(lows[finite]), sy(highs[finite])
        forward = " ".join(f"{num(a)} {num(b)}" for a, b in zip(px, low, strict=True))
        backward = " ".join(f"{num(a)} {num(b)}" for a, b in zip(px[::-1], high[::-1], strict=True))
        self.add(f'<path d="M{forward} L{backward} Z" fill="{color}" fill-opacity="{opacity:g}" stroke="none"/>')

    def histogram(self, edges, counts, color: str) -> None:
        """Barres jointives : `edges` (n+1 bords en x), `counts` (n hauteurs)."""
        sx, sy = self._scales()
        base = sy(0.0)
        for left, right, count in zip(edges[:-1], edges[1:], counts, strict=True):
            x0, x1, top = sx(left), sx(right), sy(float(count))
            self.add(f'<rect x="{num(min(x0, x1))}" y="{num(min(top, base))}" width="{num(abs(x1 - x0))}" height="{num(abs(base - top))}" fill="{color}" stroke="#ffffff" stroke-width="0.5"/>')

    def error_bar(self, x: float, y_low: float, y_high: float, color: str = INK, width: float = 1.5) -> None:
        """Barre d'erreur verticale en x, de y_low à y_high, avec des traits de bout de 6 px."""
        sx, sy = self._scales()
        px, lo, hi = sx(x), sy(y_low), sy(y_high)
        self.add(f'<path d="M{num(px)} {num(lo)} V{num(hi)} M{num(px - 3)} {num(lo)} h6 M{num(px - 3)} {num(hi)} h6" fill="none" stroke="{color}" stroke-width="{width:g}"/>')

    def error_bar_h(self, y: float, x_low: float, x_high: float, color: str = INK, width: float = 1.5) -> None:
        """Barre d'erreur horizontale en y, de x_low à x_high, avec des traits de bout de 6 px."""
        sx, sy = self._scales()
        py, lo, hi = sy(y), sx(x_low), sx(x_high)
        self.add(f'<path d="M{num(lo)} {num(py)} H{num(hi)} M{num(lo)} {num(py - 3)} v6 M{num(hi)} {num(py - 3)} v6" fill="none" stroke="{color}" stroke-width="{width:g}"/>')

    def reference_line(self, value: float, label: str = "", axis: str = "x", color: str = INK) -> None:
        """Ligne pointillée annotée : verticale en x = value (`axis="x"`) ou horizontale en y = value (`axis="y"`)."""
        sx, sy = self._scales()
        if axis == "x":
            px = num(sx(value))
            self.add(f'<line x1="{px}" y1="{self.top}" x2="{px}" y2="{self.bottom}" stroke="{color}" stroke-width="1" stroke-dasharray="4 3"/>')
            if label:
                self.text(float(px) + 4, self.top + 12, label, "start", color)
        else:
            py = num(sy(value))
            self.add(f'<line x1="{self.left}" y1="{py}" x2="{self.right}" y2="{py}" stroke="{color}" stroke-width="1" stroke-dasharray="4 3"/>')
            if label:
                self.text(self.right - 4, float(py) - 4, label, "end", color)

    def heatmap(self, x_edges, y_edges, values, palette: tuple[str, ...], low: float, high: float, symmetric: bool = False) -> int:
        """Carte de chaleur en rectangles regroupés par valeur quantifiée. `values` est (lignes en y, colonnes en x),
        entre les bords `y_edges` et `x_edges`. Les valeurs sont ramenées sur au plus `GRID_COLUMNS × GRID_ROWS` cellules
        (moyenne par bloc), quantifiées sur `len(palette)` niveaux (`symmetric` : divergente centrée sur 0, `high`
        étant max |q|), puis les cellules voisines de même niveau d'une ligne sont fusionnées. Un `<path>` par niveau,
        un sous-chemin `M x y h w v h h -w z` par rectangle. Retourne le nombre de rectangles."""
        sx, sy = self._scales()
        mean, row_bounds, column_bounds = block_mean(values)
        x_edges, y_edges = np.asarray(x_edges, float), np.asarray(y_edges, float)
        px = sx(x_edges[column_bounds])
        py = sy(y_edges[row_bounds])
        levels = diverging_levels(mean, high, len(palette)) if symmetric else sequential_levels(mean, low, high, len(palette))
        paths: dict[int, list[str]] = {}
        count = 0
        for i, row in enumerate(levels):
            starts = np.concatenate([[0], np.flatnonzero(np.diff(row)) + 1])
            ends = np.concatenate([starts[1:], [len(row)]])
            top, height = min(py[i], py[i + 1]), abs(py[i + 1] - py[i])
            for start, end in zip(starts, ends, strict=True):
                level = int(row[start])
                if level < 0:
                    continue
                width = px[end] - px[start]
                paths.setdefault(level, []).append(f"M{num(min(px[start], px[end]))} {num(top)}h{num(abs(width))}v{num(height)}h-{num(abs(width))}z")
                count += 1
        for level in sorted(paths):
            color = palette[level]
            self.add(f'<path fill="{color}" stroke="{color}" stroke-width="0.4" d="{"".join(paths[level])}"/>')
        self._axes.append(f'<rect x="{self.left}" y="{self.top}" width="{num(self.right - self.left)}" height="{num(self.bottom - self.top)}" fill="none" stroke="{GREY}" stroke-width="0.5"/>')
        return count

    def colorbar(self, palette: tuple[str, ...], low: float, high: float, label: str, symmetric: bool = False,
                 offset: int = 14, down: bool = False) -> None:
        """Barre de couleur verticale dans la marge de droite, graduée avec le même facteur commun que les axes.
        `symmetric` : palette divergente dont le centre est la classe de 0 (les classes sont centrées sur les multiples
        de high/(n−1)/2, la barre déborde donc d'une demi-classe). `offset` : distance de la barre à la zone de tracé ;
        `down` : les valeurs croissent vers le bas (une profondeur)."""
        n = len(palette)
        x0, width = self.right + offset, 12
        pad = (high - low) / (2 * (n - 1)) if symmetric else 0.0
        bottom_value, top_value = low - pad, high + pad
        scale = Scale(bottom_value, top_value, self.top, self.bottom) if down else Scale(bottom_value, top_value, self.bottom, self.top)
        for i, color in enumerate(palette):
            a, b = scale(bottom_value + i * (top_value - bottom_value) / n), scale(bottom_value + (i + 1) * (top_value - bottom_value) / n)
            self._body.append(f'<rect x="{x0}" y="{num(min(a, b))}" width="{width}" height="{num(abs(b - a) + 0.4)}" fill="{color}"/>')
        self._body.append(f'<rect x="{x0}" y="{self.top}" width="{width}" height="{num(self.bottom - self.top)}" fill="none" stroke="{GREY}" stroke-width="0.5"/>')
        ticks = nice_ticks(low, high)
        exponent, labels = axis_factor(ticks)
        for tick, text in zip(ticks, labels, strict=True):
            py = scale(tick)
            self._body.append(f'<line x1="{x0 + width}" y1="{num(py)}" x2="{x0 + width + TICK_LENGTH}" y2="{num(py)}" stroke="{GREY}"/>')
            self._body.append(f'<text x="{x0 + width + TICK_LENGTH + 3}" y="{num(py + 4)}" fill="{GREY}">{escape(minus(text))}</text>')
        self._body.append(f'<text x="{x0}" y="{self.top - 8}" fill="{INK}">{rich(with_factor(label, exponent))}</text>')

    # Rendu --------------------------------------------------------------------------------------
    def markup(self, heading_size: int = TITLE_SIZE) -> str:
        """Le contenu du cadre (titre, légende, grille, marques, axes), sans l'élément racine."""
        head = [f'<text x="{self.left}" y="18" font-size="{heading_size}" font-weight="bold" fill="{INK}">{rich(self.title)}</text>']
        if self.subtitle:
            head.append(f'<text x="{self.left}" y="34" fill="{GREY}">{rich(self.subtitle)}</text>')
        for x, y, color, label, kind in self.legend_items:
            if kind == "line":
                head.append(f'<line x1="{num(x)}" y1="{num(y + 6)}" x2="{num(x + 10)}" y2="{num(y + 6)}" stroke="{color}" stroke-width="2"/>')
            elif kind == "dash":
                head.append(f'<line x1="{num(x)}" y1="{num(y + 6)}" x2="{num(x + 10)}" y2="{num(y + 6)}" stroke="{color}" stroke-width="2" stroke-dasharray="3 2"/>')
            elif kind == "dot":
                head.append(f'<circle cx="{num(x + 5)}" cy="{num(y + 6)}" r="4" fill="{color}"/>')
            else:
                head.append(f'<rect x="{num(x)}" y="{num(y + 1)}" width="10" height="10" fill="{color}"/>')
            head.append(f'<text x="{num(x + 14)}" y="{num(y + 10)}" fill="{INK}">{rich(label)}</text>')
        return "\n".join([*head, *self._grid, *self._body, *self._axes])

    def to_bytes(self) -> bytes:
        return svg_document(self.width, self.height, self.title, self.desc or self.subtitle or self.title, self.markup())


def svg_document(width: float, height: float, title: str, desc: str, content: str) -> bytes:
    """Le document SVG : racine à `viewBox` (la figure se met à l'échelle) et sans hauteur en dur, `<title>` et `<desc>`
    en premiers enfants, fond blanc, puis `content`."""
    return "\n".join([
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{num(width)}" viewBox="0 0 {num(width)} {num(height)}" style="{STYLE}">',
        f"<title>{escape(title)}</title>",
        f"<desc>{escape(desc)}</desc>",
        f'<rect width="{num(width)}" height="{num(height)}" fill="#ffffff"/>',
        content,
        "</svg>",
        "",
    ]).encode("utf-8")


class Figure:
    """Plusieurs `Plot` dans un seul SVG : `columns` panneaux par ligne, rangés dans l'ordre, chacun dans un
    `<g transform="translate(x y)">` (`None` : une case vide). L'en-tête porte le titre et le sous-titre de la figure."""

    HEADER = 26
    HEADER_WITH_SUBTITLE = 44

    def __init__(self, title: str, desc: str, panels: list[Plot | None], subtitle: str = "", columns: int = 1):
        self.title, self.desc, self.subtitle = title, desc, subtitle
        self.panels, self.columns = panels, columns

    @property
    def header(self) -> int:
        return self.HEADER_WITH_SUBTITLE if self.subtitle else self.HEADER

    def layout(self) -> tuple[float, float, list[tuple[float, float]]]:
        """(largeur, hauteur, position de chaque panneau)."""
        rows = [self.panels[i : i + self.columns] for i in range(0, len(self.panels), self.columns)]
        widths = [max((row[j].width for row in rows if j < len(row) and row[j] is not None), default=0) for j in range(self.columns)]
        heights = [max((panel.height for panel in row if panel is not None), default=0) for row in rows]
        positions = []
        for i, row in enumerate(rows):
            for j, _ in enumerate(row):
                positions.append((float(sum(widths[:j])), float(self.header + sum(heights[:i]))))
        return float(sum(widths)), float(self.header + sum(heights)), positions

    def to_bytes(self) -> bytes:
        width, height, positions = self.layout()
        parts = [f'<text x="12" y="18" font-size="{TITLE_SIZE}" font-weight="bold" fill="{INK}">{rich(self.title)}</text>']
        if self.subtitle:
            parts.append(f'<text x="12" y="34" fill="{GREY}">{rich(self.subtitle)}</text>')
        for panel, (x, y) in zip(self.panels, positions, strict=True):
            if panel is None:  # une case laissée vide
                continue
            parts.append(f'<g transform="translate({num(x)} {num(y)})">')
            parts.append(panel.markup(FONT_SIZE))
            parts.append("</g>")
        return svg_document(width, height, self.title, self.desc, "\n".join(parts))
