"""Primitives de tracé SVG du rapport (P2c.3f, F1) : graduations, étiquettes, marques, cartes de chaleur, panneaux,
palettes, déterminisme, absence de matplotlib."""

from __future__ import annotations

import itertools
import locale
import re
from xml.etree import ElementTree

import numpy as np
import pytest
from molonari_twin.report.svgplot import (
    BLUE,
    CHAR_WIDTH,
    DIVERGING,
    GRID_COLUMNS,
    GRID_ROWS,
    MAX_POINTS,
    ORANGE,
    VERMILLION,
    VIRIDIS,
    Figure,
    Plot,
    Scale,
    axis_factor,
    diverging_levels,
    nice_ticks,
    tick_label,
)

SVG = "{http://www.w3.org/2000/svg}"


@pytest.mark.parametrize(
    ("low", "high", "expected"),
    [(0.0, 10.0, [0.0, 2.0, 4.0, 6.0, 8.0, 10.0]), (0.0, 1.0, [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]), (3.0, 3.0, [3.0])],
)
def test_ticks_use_a_1_2_5_step_inside_the_interval(low, high, expected):
    assert nice_ticks(low, high) == pytest.approx(expected)


def test_ticks_of_a_tiny_interval_stay_inside_it():
    ticks = nice_ticks(-3.2e-7, 4.9e-7)
    assert ticks and all(-3.2e-7 <= t <= 4.9e-7 for t in ticks)


def test_tick_labels_switch_to_scientific_notation_outside_the_readable_range():
    assert tick_label(0.0, [0.0, 1e-7]) == "0"
    assert tick_label(2e-7, [0.0, 1e-7, 2e-7]) == "2.00e-07"
    assert tick_label(0.5, [0.0, 0.5, 1.0]) == "0.5"
    assert tick_label(20.0, [0.0, 10.0, 20.0]) == "20"


def test_scale_maps_the_interval_onto_pixels_and_can_flip():
    assert Scale(0.0, 10.0, 100.0, 300.0)(5.0) == 200.0
    assert Scale(0.0, 10.0, 300.0, 100.0)(10.0) == 100.0


def _plot() -> Plot:
    plot = Plot("Title <&>", "subtitle")
    plot.legend((BLUE, "first"), (ORANGE, "second"))  # la légende fixe la marge du haut : avant les échelles
    x = plot.x_scale(0.0, 1.0)
    plot.x_axis(x, "x (m)")
    plot.y_axis(plot.y_scale(0.0, 2.0), "y (°C)")
    return plot


def test_plot_is_well_formed_svg_with_escaped_text_and_units():
    root = ElementTree.fromstring(_plot().to_bytes())
    assert root.tag.endswith("svg")
    text = " ".join(t.text or "" for t in root.iter("{http://www.w3.org/2000/svg}text"))
    assert "Title <&>" in text and "x (m)" in text and "y (°C)" in text and "first" in text and "second" in text


def test_plot_bytes_are_identical_between_two_generations():
    assert _plot().to_bytes() == _plot().to_bytes()



# F1 : marques, axes, légende, panneaux, palettes ---------------------------------------------------------


def parse(content: bytes) -> ElementTree.Element:
    return ElementTree.fromstring(content)


def test_marks_are_well_formed_svg():
    plot = Plot("marks", "all of them", height=260)
    x, y = plot.x_scale(0.0, 10.0), plot.y_scale(0.0, 10.0)
    xs = np.arange(10.0)
    ys = np.array([1.0, 2.0, np.nan, 4.0, 5.0, 6.0, 7.0, 8.0, 7.0, 6.0])
    plot.polyline(xs, ys, BLUE)
    plot.band(xs, xs, xs + 2, ORANGE)
    plot.histogram(np.arange(0.0, 6.0), [1, 3, 5, 4, 2], VERMILLION)
    plot.error_bar(2.0, 1.0, 3.0)
    plot.error_bar_h(8.0, 1.0, 3.0)
    plot.reference_line(5.0, "mid", "x")
    plot.reference_line(5.0, "level", "y")
    plot.x_axis(x, "x")
    plot.y_axis(y, "y")
    root = parse(plot.to_bytes())
    assert len(list(root.iter(f"{SVG}polyline"))) == 2  # le NaN coupe la ligne en deux
    assert len(list(root.iter(f"{SVG}rect"))) == 1 + 5  # fond + cinq barres
    assert any("mid" in (t.text or "") for t in root.iter(f"{SVG}text"))
    assert all(p.get("d", "M").startswith("M") for p in root.iter(f"{SVG}path"))


def test_an_isolated_point_between_gaps_is_still_drawn():
    plot = Plot("dot")
    plot.x_scale(0.0, 4.0), plot.y_scale(0.0, 4.0)
    plot.polyline([0, 1, 2, 3, 4], [1, np.nan, 2, np.nan, 3], BLUE)
    root = parse(plot.to_bytes())
    assert len(list(root.iter(f"{SVG}circle"))) == 3 and not list(root.iter(f"{SVG}polyline"))


def heat(field, palette=VIRIDIS, symmetric=False, low=0.0, high=1.0):
    plot = Plot("heat", width=640, height=300, right=96)
    plot.x_scale(0, field.shape[1]), plot.y_scale(0, field.shape[0], down=True)
    count = plot.heatmap(np.arange(field.shape[1] + 1), np.arange(field.shape[0] + 1), field, palette, low, high, symmetric)
    return count, plot.to_bytes()


def test_heatmap_groups_cells_by_level_and_bounds_its_size():
    # (a) un champ constant : une seule cellule par ligne de la grille d'affichage
    count, content = heat(np.ones((100, 1430)))
    assert count == GRID_ROWS == 60 and len(list(parse(content).iter(f"{SVG}path"))) == 1
    # (b) pire cas : chaque cellule de la grille d'affichage d'un autre niveau que sa voisine. Mesuré à
    # 30,7 octets par sous-chemin (331 913 octets pour le damier) ; la borne 40 × 10 800 + 20 000 reste au-dessus.
    rng = np.random.default_rng(0)
    checker = (np.indices((GRID_ROWS, GRID_COLUMNS)).sum(axis=0) % 2).astype(float)
    cases = {"random 100 x 1430": rng.random((100, 1430)), "checkerboard 100 x 1430": (np.indices((100, 1430)).sum(axis=0) % 2).astype(float),
             "random display grid": rng.random((GRID_ROWS, GRID_COLUMNS)), "checkerboard display grid": checker}
    limit = 40 * GRID_COLUMNS * GRID_ROWS + 20_000
    for name, field in cases.items():
        count, content = heat(field)
        paths = list(parse(content).iter(f"{SVG}path"))
        subpaths = sum(p.get("d").count("M") for p in paths)
        assert len(paths) <= len(VIRIDIS) and subpaths == count <= GRID_COLUMNS * GRID_ROWS == 10_800, name
        assert len(content) <= limit == 452_000, (name, len(content))
    assert heat(checker)[0] == 10_800  # le damier de la grille d'affichage atteint le maximum


def test_heatmap_leaves_missing_cells_empty_and_is_deterministic():
    field = np.full((4, 6), 0.5)
    field[1, 2] = np.nan
    count, content = heat(field)
    assert count == 4 + 1  # la ligne trouée donne deux rectangles
    assert heat(field)[1] == content


def test_long_series_are_decimated():
    plot = Plot("chains", height=300)
    plot.x_scale(0, 3000), plot.y_scale(0, 16)
    for chain in range(16):
        for panel in range(5):  # chaque panneau a ses propres courbes : ici le même cadre, 80 courbes
            plot.polyline(np.arange(3000.0), np.sin(np.arange(3000.0) / 50 + chain + panel) + 8, BLUE)
    root = parse(plot.to_bytes())
    lines = list(root.iter(f"{SVG}polyline"))
    points = [len(p.get("points").split()) for p in lines]
    assert len(lines) == 80 and sum(points) <= 16 * 5 * MAX_POINTS and max(points) == MAX_POINTS
    first, last = lines[0].get("points").split()[0], lines[0].get("points").split()[-1]
    assert first.startswith(f"{plot.x(0):.2f},") and last.startswith(f"{plot.x(2999):.2f},")  # premier et dernier point
    again = Plot("chains", height=300)
    again.x_scale(0, 3000), again.y_scale(0, 16)
    again.polyline(np.arange(3000.0), np.sin(np.arange(3000.0) / 50 + 8), BLUE)
    twin = Plot("chains", height=300)
    twin.x_scale(0, 3000), twin.y_scale(0, 16)
    twin.polyline(np.arange(3000.0), np.sin(np.arange(3000.0) / 50 + 8), BLUE)
    assert again.to_bytes() == twin.to_bytes()


def test_diverging_palette_is_odd_with_white_at_zero():
    assert len(DIVERGING) == 17 and DIVERGING[8] == "#ffffff" and len(VIRIDIS) == 16
    assert int(diverging_levels([0.0], 3.0)[0]) == 8
    assert int(diverging_levels([3.0], 3.0)[0]) == 16 and int(diverging_levels([-3.0], 3.0)[0]) == 0
    values = np.random.default_rng(1).uniform(-5, 5, 1000)
    assert np.all(diverging_levels(values, 5.0) + diverging_levels(-values, 5.0) == 16)
    assert np.all(diverging_levels(np.zeros(5)) == 8)  # un champ nul ne divise pas par zéro
    assert list(diverging_levels([np.nan, 1.0], 1.0)) == [-1, 16]
    # bleu pour le négatif, rouge pour le positif
    blue_side, red_side = DIVERGING[0], DIVERGING[16]
    assert int(blue_side[5:7], 16) > int(blue_side[1:3], 16) and int(red_side[1:3], 16) > int(red_side[5:7], 16)


def test_figure_stacks_panels():
    first, second = Plot("one", height=200), Plot("two", height=240)
    for plot in (first, second):
        plot.x_scale(0, 1), plot.y_scale(0, 1)
    figure = Figure("both", "two panels", [first, second], subtitle="stacked")
    root = parse(figure.to_bytes())
    groups = [g for g in root.iter(f"{SVG}g") if g.get("transform", "").startswith("translate(")]
    assert len(groups) == 2
    assert groups[0].get("transform") == "translate(0.00 44.00)" and groups[1].get("transform") == "translate(0.00 244.00)"
    assert root.get("viewBox") == f"0 0 480.00 {44 + 200 + 240:.2f}"
    side = Figure("side", "two columns", [first, second], columns=2)
    assert parse(side.to_bytes()).get("viewBox") == f"0 0 960.00 {26 + 240:.2f}"


def test_time_axis_ticks_at_midnight_without_locale():
    times = np.datetime64("2016-06-28T07:30:00") + np.arange(15 * 96) * np.timedelta64(15, "m")
    seconds = times.astype("datetime64[s]").astype(np.int64)

    def labels():
        plot = Plot("time", height=200)
        scale = plot.x_scale(float(seconds[0]), float(seconds[-1]))
        plot.time_axis(scale, times, "date")
        root = parse(plot.to_bytes())
        return [t.text for t in root.iter(f"{SVG}text") if t.text and re.fullmatch(r"\d\d-\d\d", t.text)], plot

    found, plot = labels()
    assert found == ["06-29", "07-01", "07-03", "07-05", "07-07", "07-09", "07-11", "07-13"]  # pas de 2 jours : 15 minuits, au plus 8
    assert "date (2016)" in plot.to_bytes().decode()  # l'année est dite une fois, dans le libellé
    ticks = re.findall(r'<line x1="([\d.]+)" y1="\d+" x2="[\d.]+" y2="\d+" stroke="#dddddd"', plot.to_bytes().decode())
    assert len(ticks) == len(found)
    midnight = Scale(float(seconds[0]), float(seconds[-1]), plot.left, plot.right)(np.datetime64("2016-06-29T00:00:00").astype(np.int64))
    assert float(ticks[0]) == pytest.approx(midnight, abs=0.01)
    try:
        locale.setlocale(locale.LC_TIME, "fr_FR.UTF-8")
    except locale.Error:
        pytest.skip("la locale fr_FR.UTF-8 n'existe pas sur cette machine")
    try:
        assert labels()[0] == found
    finally:
        locale.setlocale(locale.LC_TIME, "")


def test_time_axis_over_two_years_writes_full_dates():
    times = np.array(["2016-12-20T00:00:00", "2017-01-20T00:00:00"], dtype="datetime64[s]")
    plot = Plot("time", height=200)
    plot.time_axis(plot.x_scale(*[float(s) for s in times.astype(np.int64)]), times)
    assert any(re.fullmatch(r"\d{4}-\d\d-\d\d", t.text or "") for t in parse(plot.to_bytes()).iter(f"{SVG}text"))


def test_axis_factor_replaces_scientific_tick_labels():
    ticks = nice_ticks(-2.2e-8, 2.2e-8)
    exponent, labels = axis_factor(ticks)
    assert exponent == -8 and labels == ["-2", "-1", "0", "1", "2"] and not any("e" in label for label in labels)
    assert axis_factor([0.0, 0.5, 1.0]) == (0, ["0", "0.5", "1.0"])
    assert axis_factor([0.0, 1e6, 2e6])[0] == 6
    plot = Plot("factor")
    x = plot.x_scale(-2.2e-8, 2.2e-8)
    plot.x_axis(x, "q ({factor} m·s^{-1})")
    markup = plot.to_bytes().decode()
    assert 'q (× 10<tspan baseline-shift="super" font-size="9">\u22128</tspan> m·s<tspan baseline-shift="super" font-size="9">\u22121</tspan>)' in markup
    assert not any(ch in markup for ch in "\u207b\u00b9\u2078\u2079")  # aucun caractère Unicode en exposant
    assert not re.search(r">-?\d\.\d+e[-+]\d+<", markup)


def test_ticks_point_outwards():
    plot = Plot("ticks", height=240)
    x, y = plot.x_scale(0.0, 10.0), plot.y_scale(0.0, 10.0)
    plot.x_axis(x, "x")
    plot.y_axis(y, "y")
    root = parse(plot.to_bytes())
    marks = [line for line in root.iter(f"{SVG}line") if line.get("stroke") == "#666666"]
    horizontal = [m for m in marks if m.get("x1") == m.get("x2") and float(m.get("y2")) - float(m.get("y1")) < 10]  # graduations de l'axe x
    vertical = [m for m in marks if m.get("y1") == m.get("y2") and float(m.get("x2")) - float(m.get("x1")) < 10]  # graduations de l'axe y
    assert horizontal and vertical
    assert all(float(m.get("y1")) == plot.bottom and float(m.get("y2")) > plot.bottom for m in horizontal)
    assert all(float(m.get("x2")) == plot.left and float(m.get("x1")) < plot.left for m in vertical)
    grid = [line for line in root.iter(f"{SVG}line") if line.get("stroke") == "#dddddd"]
    assert grid and all(line.get("stroke-width") == "0.5" for line in grid)


def test_axes_accept_an_inverted_scale():
    plot = Plot("depth", height=240)
    plot.x_scale(0.0, 1.0)
    depth = plot.y_scale(0.0, 0.4, down=True)
    plot.y_axis(depth, "depth (m)")
    assert depth(0.0) == plot.top and depth(0.4) == plot.bottom
    texts = [t.text for t in parse(plot.to_bytes()).iter(f"{SVG}text")]
    assert "0" in texts and "0.4" in texts


def test_legend_wraps_instead_of_overlapping():
    plot = Plot("legend", width=480, height=300)
    initial_top = plot.top
    entries = [(BLUE, "forward model, no interval"), (ORANGE, "posterior interval at 90 %"), (VERMILLION, "5th to 95th percentile"),
               (BLUE, "best fit sample"), (ORANGE, "thermometer depth"), (VERMILLION, "layer bottom")]
    plot.legend(*entries)
    rows: dict[float, list[tuple[float, float]]] = {}
    for x, y, _, label, _ in plot.legend_items:
        rows.setdefault(y, []).append((x, x + len(label) * CHAR_WIDTH + 28))
    assert len(rows) >= 2 and plot.legend_rows == len(rows)
    for spans in rows.values():
        spans.sort()
        assert all(a_end <= b_start for (_, a_end), (b_start, _) in itertools.pairwise(spans))
        assert spans[-1][1] <= plot.right + 1e-9 or len(spans) == 1
    assert plot.top > initial_top
    last_row = max(rows)
    assert last_row + 12 < plot.top  # la dernière ligne de légende est au-dessus de la zone de tracé
    y = plot.y_scale(0.0, 10.0)
    plot.y_axis(y, "y")
    assert y(10.0) == plot.top  # la plus haute graduation est à `top`
    with pytest.raises(RuntimeError):
        plot.legend((BLUE, "too late"))


def test_a_single_legend_row_keeps_its_entries_on_one_line():
    plot = Plot("short")
    plot.legend((BLUE, "a"), (ORANGE, "b"))
    assert plot.legend_rows == 1 and len({y for _, y, *_ in plot.legend_items}) == 1


def test_svg_root_has_viewbox_title_and_description_and_no_fixed_height():
    plot = Plot("The title", "sub", desc="What it shows")
    plot.x_scale(0, 1), plot.y_scale(0, 1)
    root = parse(plot.to_bytes())
    assert root.get("viewBox") == "0 0 480.00 220.00" and root.get("width") == "480.00" and root.get("height") is None
    assert root[0].tag == f"{SVG}title" and root[0].text == "The title"
    assert root[1].tag == f"{SVG}desc" and root[1].text == "What it shows"
    assert "font-size:12px" in root.get("style") and 'font-size="13"' in plot.to_bytes().decode()


def test_axes_need_their_scales_and_marks_both():
    plot = Plot("x")
    with pytest.raises(RuntimeError):
        plot.x_axis(Scale(0, 1, 0, 1), "x")
    plot.x_scale(0, 1)
    with pytest.raises(RuntimeError):
        plot.polyline([0, 1], [0, 1], BLUE)
