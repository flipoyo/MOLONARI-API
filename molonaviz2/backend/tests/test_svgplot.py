"""Primitives de tracé SVG du rapport (P2c.3f) : graduations, étiquettes, déterminisme, absence de matplotlib."""

from __future__ import annotations

from xml.etree import ElementTree

import pytest
from molonari_twin.report.svgplot import (
    BLUE,
    ORANGE,
    Plot,
    Scale,
    nice_ticks,
    tick_label,
)


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
    x = plot.x_scale(0.0, 1.0)
    plot.x_axis(x, "x (m)")
    plot.y_axis(plot.y_scale(0.0, 2.0), "y (°C)")
    plot.legend((BLUE, "first"), (ORANGE, "second"))
    return plot


def test_plot_is_well_formed_svg_with_escaped_text_and_units():
    root = ElementTree.fromstring(_plot().to_bytes())
    assert root.tag.endswith("svg")
    text = " ".join(t.text or "" for t in root.iter("{http://www.w3.org/2000/svg}text"))
    assert "Title <&>" in text and "x (m)" in text and "y (°C)" in text and "first" in text and "second" in text


def test_plot_bytes_are_identical_between_two_generations():
    assert _plot().to_bytes() == _plot().to_bytes()

