"""Figures du rapport (P2c.3f, F1) : des SVG dérivés des seules entrées publiées, écrits sans bibliothèque graphique.

`build_figures(published, chains)` reçoit les **réponses publiées** (`{nom de fichier: Answer}`, relues des octets que le
rapport écrit) et le dictionnaire de `chains.npz` : rien d'autre, ni `TwinAPI`, ni configuration, ni résultat brut. Une
figure ne contient donc rien que le rapport ne publie pas (bornes a priori, σ, mesures), et une figure dont une source
manque n'est pas produite. Les primitives de tracé sont dans `svgplot.py`.

Aucune figure ne lit un seuil ni un verdict (R̂, ESS, identifiabilité) : elles écrivent des valeurs.

Libellés en anglais, sans phrase longue : le rapport est lu par la gate, qui traduit autour (la page `README.md` est en
français, voir `readme.py`).
"""

from __future__ import annotations

import numpy as np

from ..questions.answer import Answer
from ..questions.soil import hydraulic_conductivity
from .svgplot import (
    BLUE,
    DIVERGING,
    GREY,
    INK,
    LIGHT,
    ORANGE,
    SERIES,
    VERMILLION,
    VIRIDIS,
    Figure,
    Plot,
    edges_from_centers,
    factor_text,
    minus,
    num,
    rich_number,
    sample_indices,
)

ROLES = ("forward", "inversion")
MODEL = {"forward": "forward model", "inversion": "inversion model"}
FIELD_WIDTH, COLORBAR_MARGIN = 760, 116
DEPTH_LINES, PROFILE_DATES = 10, 5
HISTOGRAM_BINS = 30
PARAMETERS = {  # paramètre inversé → (libellé court, libellé d'axe)
    "neg_log10_k": ("−log10 k", "−log10 k (k in m^{2})"),
    "porosity": ("porosity n", "porosity n (1)"),
    "lambda_s": ("λs", "λs ({factor} W·m^{-1}·K^{-1})"),
    "rhos_cs": ("ρs·cs", "ρs·cs ({factor} J·m^{-3}·K^{-1})"),
}

Built = tuple[str, bytes, list[str]]


def _text(answer: Answer, name: str) -> str:
    return str(answer.data[name].value) if name in answer.data else ""


def _number(answer: Answer, name: str) -> float:
    return float(answer.data[name].value)


def _array(answer: Answer, name: str) -> np.ndarray:
    return np.asarray(answer.data[name].value)


def format_time(moment) -> str:
    """`2016-06-27 13:00`, écrit par numpy (jamais `strftime`, que la locale change)."""
    return str(np.datetime64(moment, "m")).replace("T", " ")


def step_text(seconds: np.ndarray) -> str:
    """Pas médian d'une série de dates en secondes : `1 h` ou `15 min`."""
    step = float(np.median(np.diff(seconds))) if len(seconds) > 1 else 0.0
    return f"{step / 3600:g} h" if step % 3600 == 0 else f"{step / 60:g} min"


def common_exponent(values) -> tuple[int, list[float]]:
    """(exposant, valeurs divisées) : un facteur commun quand une valeur non nulle sort de [1e-3, 1e5[."""
    nonzero = [abs(v) for v in values if v != 0]
    if not nonzero or all(1e-3 <= v < 1e5 for v in nonzero):
        return 0, list(values)
    exponent = int(np.floor(np.log10(max(nonzero))))
    return exponent, [v / 10**exponent for v in values]


def shared(values, significant: int = 3) -> tuple[list[str], int]:
    """(textes, exposant) de quelques valeurs écrites avec un même facteur `× 10^{exposant}` (voir `svgplot.rich`)."""
    exponent, scaled = common_exponent(values)
    return [rich_number(v, significant) for v in scaled], exponent


def _published(published: dict[str, Answer], *names: str) -> list[Answer] | None:
    return [published[name] for name in names] if all(name in published for name in names) else None


# 1. Flux d'eau ------------------------------------------------------------------------------------------------------


def water_flux_figure(published: dict[str, Answer]) -> Built | None:
    """Une ligne par modèle sur un axe q commun : moyenne (point), P5–P95 de q(t) (trait gris), intervalle a posteriori à
    90 % (barre bleue, inversion seulement). q ≥ 0 : infiltration (la rivière alimente la nappe), q < 0 : exfiltration."""
    names = [f"{role}-water_flux.json" for role in ROLES if f"{role}-water_flux.json" in published]
    if not names:
        return None
    rows = []
    for name in names:
        answer = published[name]
        mean = answer.data["mean"]
        rows.append({
            "role": name.split("-")[0], "mean": float(mean.value), "interval": None if mean.interval is None else (float(mean.interval[0]), float(mean.interval[1])),
            "p5": _number(answer, "p5"), "p95": _number(answer, "p95"), "level": _number(answer, "level"),
        })
    marks = [0.0] + [v for row in rows for v in (row["mean"], row["p5"], row["p95"], *(row["interval"] or ()))]
    low, high = _room_around_zero(min(marks), max(marks), FIELD_WIDTH - 56 - 20 - 100)
    plot = Plot("Water flux through the riverbed", "positive q: infiltration, the river feeds the aquifer; negative q: exfiltration",
                width=FIELD_WIDTH - 100, left=56, right=20,
                desc="Mean water flux through the riverbed, its 5th to 95th percentile over time and, for the inversion, "
                     f"its posterior interval, from {', '.join(names)}.")
    plot.legend((INK, "mean", "dot"), (GREY, "q(t) 5th to 95th percentile", "line"),
                *(((BLUE, f"{rows[-1]['level'] * 100:g} % posterior interval of the mean", "line"),) if any(r["interval"] for r in rows) else ()))
    band = 24
    row_height = 70
    plot.plot_height(band + row_height * len(rows))
    x = plot.x_scale(low, high)
    plot.y_scale(0.0, 1.0)
    for i, row in enumerate(rows):
        top = plot.top + band + i * row_height
        middle = top + 48
        if i:
            plot.add(f'<line x1="{plot.left}" y1="{num(top)}" x2="{plot.right}" y2="{num(top)}" stroke="{LIGHT}" stroke-width="0.5"/>')
        label = "forward model — no interval" if row["role"] == "forward" else f"inversion model — {row['level'] * 100:g} % posterior interval"
        plot.text(plot.left + 6, top + 16, label, "start", INK, halo=True)
        plot.text(plot.left + 6, top + 32, _flux_numbers(row), "start", GREY, halo=True)  # les valeurs, quand une barre est trop courte pour se lire
        plot.add(f'<line x1="{num(x(row["p5"]))}" y1="{num(middle)}" x2="{num(x(row["p95"]))}" y2="{num(middle)}" stroke="{GREY}" stroke-width="3"/>')
        if row["interval"] is not None:
            plot.add(f'<line x1="{num(x(row["interval"][0]))}" y1="{num(middle)}" x2="{num(x(row["interval"][1]))}" y2="{num(middle)}" stroke="{BLUE}" stroke-width="8"/>')
        plot.add(f'<circle cx="{num(x(row["mean"]))}" cy="{num(middle)}" r="5" fill="{INK}" stroke="#ffffff" stroke-width="1.5"/>')
    plot.reference_line(0.0, "", "x")
    zero = x(0.0)
    plot.text(zero + 6, plot.top + 14, "infiltration (q ≥ 0) →", "start", INK)
    plot.text(zero - 6, plot.top + 14, "← exfiltration (q < 0)", "end", INK)
    plot.x_axis(x, "water flux q ({factor} m·s^{-1})")
    return "figures/water_flux.svg", plot.to_bytes(), names


def _flux_numbers(row: dict) -> str:
    """La moyenne (et l'intervalle a posteriori) écrits avec un même facteur commun : `mean −4.5, 90 % [−7.3, −1.9] × 10^{-9} m·s^{-1}`."""
    values = [row["mean"], *(row["interval"] or ())]
    parts, exponent = shared(values)
    text = f"mean {parts[0]}" + (f", {row['level'] * 100:g} % interval [{parts[1]}, {parts[2]}]" if row["interval"] else "")
    return f"{text} {factor_text(exponent)} m\u00b7s^{{-1}}".replace("  ", " ")


def _room_around_zero(low: float, high: float, width: float, left_room: float = 150, right_room: float = 175) -> tuple[float, float]:
    """Étendue [A, B] de l'axe q, avec de la place de part et d'autre de 0 pour les deux annotations (en px sur `width`)."""
    low, high = min(low, 0.0), max(high, 0.0)
    if high == low:
        high = low + 1e-9
    pad = (high - low) * 0.05
    lo, hi = low - pad, high + pad
    total = hi - lo
    for _ in range(60):
        total = max(-lo, left_room * total / width) + max(hi, right_room * total / width)
    return -max(-lo, left_room * total / width), max(hi, right_room * total / width)


# 2 à 4. Champs ------------------------------------------------------------------------------------------------------


def _fields(published: dict[str, Answer], role: str) -> dict | None:
    fields, describe = _published(published, f"{role}-fields.npz", f"{role}-describe.json") or (None, None)
    if fields is None:
        return None
    seconds = _array(fields, "time").astype("datetime64[s]").astype(np.int64).astype(float)
    return {
        "depth": _array(fields, "depth").astype(float), "time": _array(fields, "time").astype("datetime64[s]"), "seconds": seconds,
        "temperature": _array(fields, "temperature").astype(float), "water_flux": _array(fields, "water_flux").astype(float),
        "sensors": _array(describe, "sensor_depths").astype(float), "offset": _number(describe, "offset") if "offset" in describe.data else 0.0,
    }


def _field_subtitle(role: str, what: str, seconds: np.ndarray) -> str:
    best = "best-fit sample" if role == "inversion" else "forward model, no interval"
    return f"{what}, {best}, one value every {step_text(seconds)}"


def _field_plot(title: str, subtitle: str, desc: str, height: int = 340) -> Plot:
    return Plot(title, subtitle, width=FIELD_WIDTH, height=height, left=64, right=COLORBAR_MARGIN, desc=desc)


def _time_edges(data: dict) -> np.ndarray:
    return edges_from_centers(data["seconds"])


def temperature_field_figure(published: dict[str, Answer], role: str) -> Built | None:
    """Carte de chaleur T(z, t) du meilleur ajustement, profondeur vers le bas, axe du temps en dates, profondeurs des
    thermomètres en tirets à droite, barre de couleur en °C."""
    data = _fields(published, role)
    if data is None:
        return None
    sources = [f"{role}-fields.npz", f"{role}-describe.json"]
    temperature = data["temperature"]
    low, high = float(np.nanmin(temperature)), float(np.nanmax(temperature))
    plot = _field_plot(f"Temperature field — {MODEL[role]}", _field_subtitle(role, "T(z, t)", data["seconds"]),
                       f"Simulated temperature as a function of depth and date for the {MODEL[role]}, from {' and '.join(sources)}.")
    plot.legend((INK, "thermometers T1–T3", "line"), (INK, "aquifer thermometer T4 (boundary condition)", "dash"))
    plot.plot_height(250)
    edges = _time_edges(data)
    x = plot.x_scale(edges[0], edges[-1])
    depth_edges = edges_from_centers(data["depth"])
    y = plot.y_scale(max(depth_edges[0], 0.0), depth_edges[-1], down=True)
    plot.heatmap(edges, depth_edges, temperature, VIRIDIS, low, high)
    for i, depth in enumerate(data["sensors"] + data["offset"]):
        if not y.low <= depth <= y.high:
            continue
        dash = ' stroke-dasharray="3 2"' if i == len(data["sensors"]) - 1 else ""
        plot.add(f'<line x1="{plot.right}" y1="{num(y(depth))}" x2="{plot.right + 10}" y2="{num(y(depth))}" stroke="{INK}" stroke-width="2"{dash}/>')
        plot.text(plot.right + 13, y(depth) + 4, f"{depth:g} m", "start", INK)
    plot.colorbar(VIRIDIS, low, high, "T (°C)", offset=64)
    plot.time_axis(x, data["time"], "date")
    plot.y_axis(y, "depth (m)")
    return f"figures/temperature_field-{role}.svg", plot.to_bytes(), sources


def temperature_depths_figure(published: dict[str, Answer], role: str) -> Built | None:
    """Deux panneaux : T(t) à au plus 10 profondeurs (couleur selon la profondeur), puis le profil T(z) à 5 dates."""
    fields = published.get(f"{role}-fields.npz")
    if fields is None:
        return None
    depth, temperature = _array(fields, "depth").astype(float), _array(fields, "temperature").astype(float)
    time = _array(fields, "time").astype("datetime64[s]")
    seconds = time.astype(np.int64).astype(float)
    sources = [f"{role}-fields.npz"]

    chosen = sample_indices(len(depth), DEPTH_LINES)
    top = Plot(f"Temperature over time, {len(chosen)} depths", "", width=FIELD_WIDTH, left=64, right=COLORBAR_MARGIN,
               desc="")
    top.plot_height(190)
    x = top.x_scale(seconds[0], seconds[-1])
    y = top.y_scale(*_padded(float(np.nanmin(temperature[chosen])), float(np.nanmax(temperature[chosen]))))
    span = float(depth.max() - depth.min()) or 1.0
    for index in chosen:
        level = round((depth[index] - depth.min()) / span * (len(VIRIDIS) - 1))
        top.polyline(seconds, temperature[index], VIRIDIS[level], 1.5)
    top.colorbar(VIRIDIS, float(depth.min()), float(depth.max()), "depth (m)", down=True)
    top.time_axis(x, time, "date")
    top.y_axis(y, "temperature (°C)")

    dates = sample_indices(len(seconds), PROFILE_DATES)
    bottom = Plot(f"Temperature profile T(z), {len(dates)} dates", "", width=FIELD_WIDTH, left=64, right=COLORBAR_MARGIN)
    bottom.legend(*[(SERIES[i % len(SERIES)], format_time(time[d]), "line") for i, d in enumerate(dates)])
    bottom.plot_height(200)
    edges = (float(np.nanmin(temperature[:, dates])), float(np.nanmax(temperature[:, dates])))
    px = bottom.x_scale(*_padded(*edges))
    py = bottom.y_scale(0.0, float(edges_from_centers(depth)[-1]), down=True)
    for i, d in enumerate(dates):
        bottom.polyline(temperature[:, d], depth, SERIES[i % len(SERIES)], 1.5)
    bottom.x_axis(px, "temperature (°C)")
    bottom.y_axis(py, "depth (m)")

    title = f"Temperature at depth and date — {MODEL[role]}"
    figure = Figure(title, f"Temperature over time at {len(chosen)} depths and temperature profiles at {len(dates)} dates for the {MODEL[role]}, from {sources[0]}.",
                    [top, bottom], subtitle=_field_subtitle(role, "T", seconds))
    return f"figures/temperature_depths-{role}.svg", figure.to_bytes(), sources


def _padded(low: float, high: float, fraction: float = 0.05) -> tuple[float, float]:
    span = (high - low) or 1.0
    return low - span * fraction, high + span * fraction


def water_flux_field_figure(published: dict[str, Answer], role: str) -> Built | None:
    """Deux panneaux : frise q(z, t) en palette divergente centrée sur 0 (rouge : infiltration), puis q(t) dans la maille du
    fond avec la ligne q = 0."""
    fields = published.get(f"{role}-fields.npz")
    if fields is None:
        return None
    depth, flux = _array(fields, "depth").astype(float), _array(fields, "water_flux").astype(float)
    time = _array(fields, "time").astype("datetime64[s]")
    seconds = time.astype(np.int64).astype(float)
    sources = [f"{role}-fields.npz"]
    high = float(np.nanmax(np.abs(flux))) or 1.0

    top = Plot("Water flux field q(z, t)", "", width=FIELD_WIDTH, left=64, right=COLORBAR_MARGIN)
    top.legend((DIVERGING[16], "red: infiltration, q > 0 (river to aquifer)", "box"), (DIVERGING[0], "blue: exfiltration, q < 0", "box"))
    top.plot_height(220)
    edges = edges_from_centers(seconds)
    x = top.x_scale(edges[0], edges[-1])
    depth_edges = edges_from_centers(depth)
    y = top.y_scale(max(depth_edges[0], 0.0), depth_edges[-1], down=True)
    top.heatmap(edges, depth_edges, flux, DIVERGING, -high, high, symmetric=True)
    top.colorbar(DIVERGING, -high, high, "q ({factor} m·s^{-1})", symmetric=True)
    top.time_axis(x, time, "date")
    top.y_axis(y, "depth (m)")

    first = f"{depth[0]:g}"
    bottom = Plot("Water flux in the first cell", "", width=FIELD_WIDTH, left=64, right=COLORBAR_MARGIN)
    bottom.legend((INK, f"best-fit field, first cell (z = {first} m), one value every {step_text(seconds)}", "line"))
    bottom.plot_height(170)
    series = flux[0]
    px = bottom.x_scale(edges[0], edges[-1])
    py = bottom.y_scale(*_padded(min(float(np.nanmin(series)), 0.0), max(float(np.nanmax(series)), 0.0)))
    bottom.polyline(seconds, series, INK, 1.5)
    bottom.reference_line(0.0, "q = 0", "y", GREY)
    bottom.time_axis(px, time, "date")
    bottom.y_axis(py, "q ({factor} m·s^{-1})")

    figure = Figure(f"Water flux field — {MODEL[role]}",
                    f"Water flux as a function of depth and date, and at the first cell, for the {MODEL[role]}, from {sources[0]}.",
                    [top, bottom], subtitle=_field_subtitle(role, "q(z, t)", seconds))
    return f"figures/water_flux_field-{role}.svg", figure.to_bytes(), sources


# 5. Distribution a posteriori ---------------------------------------------------------------------------------------


def _summary(values: list[float]) -> str:
    parts, exponent = shared(values)
    median, low, high = parts
    text = f"median {median}, 90 % [{low}, {high}]"
    return text + (f" × 10^{{{exponent}}}" if exponent else "")


def _histogram_panel(title: str, label: str, samples: np.ndarray, median: float, low: float, high: float, width: int = 280) -> Plot:
    counts, edges = np.histogram(samples, bins=HISTOGRAM_BINS)
    plot = Plot(title, _summary([median, low, high]), width=width, height=222, left=44, right=12, top=62, bottom=50)
    top = float(counts.max()) * 1.25 or 1.0
    x = plot.x_scale(float(samples.min()), float(samples.max()) if samples.max() > samples.min() else float(samples.min()) + 1.0)
    y = plot.y_scale(0.0, top)
    plot.histogram(edges, counts, BLUE)
    plot.reference_line(median, "", "x", INK)
    plot.error_bar_h(float(counts.max()) * 1.12, low, high, VERMILLION)
    plot.x_axis(x, label)
    plot.y_axis(y, "samples")
    return plot


def posterior_figure(published: dict[str, Answer]) -> Built | None:
    """Une ligne de panneaux par couche : l'histogramme des échantillons de chaque paramètre inversé (médiane et intervalle à
    90 % calculés sur ces échantillons), puis celui de log10 K (médiane et intervalle lus dans `soil_properties`). L'axe va
    du minimum au maximum des échantillons : ni bornes a priori, ni identifiabilité."""
    pair = _published(published, "inversion-posterior_samples.npz", "inversion-soil_properties.json")
    if pair is None:
        return None
    posterior, soil = pair
    sources = ["inversion-posterior_samples.npz", "inversion-soil_properties.json"]
    samples = _array(posterior, "samples").astype(float)  # (chaînes, itérations, couches, paramètres)
    names = [str(p) for p in _array(posterior, "parameter")]
    layers = [str(layer) for layer in _array(posterior, "layer")]
    panels: list[Plot | None] = []
    for layer_index, layer in enumerate(layers):
        prefix = f"{layer} — " if len(layers) > 1 else ""
        for k, name in enumerate(names):
            values = samples[:, :, layer_index, k].ravel()
            low, median, high = (float(v) for v in np.quantile(values, [0.05, 0.5, 0.95]))
            short, axis_label = PARAMETERS.get(name, (name, name))
            panels.append(_histogram_panel(prefix + short, axis_label, values, median, low, high))
        if "neg_log10_k" in names:
            k = names.index("neg_log10_k")
            conductivity = np.log10(hydraulic_conductivity(10.0 ** (-samples[:, :, layer_index, k].ravel())))
            interval = soil.data["K"].interval
            bounds = (float(np.log10(np.asarray(interval[0])[layer_index])), float(np.log10(np.asarray(interval[1])[layer_index])))
            median = float(np.log10(_array(soil, "K_median")[layer_index]))
            panels.append(_histogram_panel(prefix + "log10 K", "log10 K (K in m·s^{-1})", conductivity, median, *bounds))
        while len(panels) % 3:
            panels.append(None)
    chains, iterations = samples.shape[0], samples.shape[1]
    figure = Figure("Posterior distribution of the soil parameters",
                    "Histograms of the posterior samples of each inverted parameter and of log10 K, per layer, with median and 90 % interval, "
                    f"from {' and '.join(sources)}.",
                    panels, subtitle=f"{chains} chains × {iterations} samples; median and 90 % interval (5th to 95th percentile); axes span the sampled range",
                    columns=3)
    return "figures/posterior.svg", figure.to_bytes(), sources


# 6. Chaînes ---------------------------------------------------------------------------------------------------------


def chains_figure(published: dict[str, Answer], chains: dict | None) -> Built | None:
    """Un panneau pour l'énergie, puis un par paramètre et par couche, en fonction de l'itération, une couleur par chaîne
    (Okabe et Ito, répétées en pointillé au-delà de 7)."""
    convergence = published.get("inversion-public_convergence.json")
    if chains is None or convergence is None:
        return None
    sources = ["chains.npz", "inversion-public_convergence.json"]
    iteration = np.asarray(chains["iteration"], float)
    panels = [("energy", "", np.asarray(chains["energy"], float))]
    for name in PARAMETERS:
        key = f"samples.{name}"
        if key not in chains:
            continue
        series = np.asarray(chains[key], float)  # (itérations, chaînes, couches)
        for layer in range(series.shape[2]):
            title = PARAMETERS[name][1].replace("{factor} ", "") + (f", layer {layer + 1}" if series.shape[2] > 1 else "")
            panels.append((title, "", series[:, :, layer]))
    plots = []
    for title, label, series in panels:
        plot = Plot(title, "", width=420, height=198, left=56, right=14, top=44, bottom=50)
        x = plot.x_scale(float(iteration.min()), float(iteration.max()) if iteration.max() > iteration.min() else float(iteration.min()) + 1)
        y = plot.y_scale(*_padded(float(np.nanmin(series)), float(np.nanmax(series))))
        for chain in range(series.shape[1]):
            plot.polyline(iteration, series[:, chain], SERIES[chain % len(SERIES)], 1.0, "4 3" if chain >= len(SERIES) else None)
        plot.x_axis(x, "iteration")
        plot.y_axis(y, label)
        plots.append(plot)
    if len(plots) % 2:
        plots.append(None)
    count = int(_number(convergence, "chains"))
    subtitle = (f"R̂ max = {rich_number(_number(convergence, 'rhat_max'))}, ESS min = {rich_number(_number(convergence, 'ess_min'))}; "
                f"{count} chains, burn-in {int(_number(convergence, 'burn_in_iterations'))} iterations")
    figure = Figure("Chains of the inversion", f"Energy and parameter values along the iterations, one colour per chain, from {' and '.join(sources)}.",
                    plots, subtitle=subtitle + "; one colour per chain (dashed beyond 7)", columns=2)
    return "figures/chains.svg", figure.to_bytes(), sources


# 7 et 8. Ajustement et comparaison ---------------------------------------------------------------------------------


def fit_quality_figure(published: dict[str, Answer]) -> Built | None:
    """RMSE par capteur et totale, barres groupées forward / inversion, valeur écrite au-dessus (°C, trois décimales)."""
    names = [f"{role}-fit_quality.json" for role in ROLES if f"{role}-fit_quality.json" in published]
    if not names:
        return None
    groups = None
    series = []
    for name in names:
        answer = published[name]
        sensors = [str(s).upper() for s in _array(answer, "sensor")] + ["Total"]
        groups = groups or sensors
        series.append((name.split("-")[0], [float(v) for v in _array(answer, "rmse")] + [_number(answer, "rmse_total")]))
    plot = Plot("Temperature misfit (RMSE)", "RMSE at the thermometers; inversion: best sample",
                width=560, height=270, left=64, right=20, bottom=30,
                desc=f"Root mean square error of the simulated temperature at each thermometer and in total, from {', '.join(names)}.")
    colors = {"forward": BLUE, "inversion": ORANGE}
    plot.legend(*[(colors[role], MODEL[role], "box") for role, _ in series])
    plot.plot_height(170)
    top = max(max(values) for _, values in series) * 1.18 or 1e-12
    y = plot.y_scale(0.0, top)
    slot = (plot.right - plot.left) / len(groups)
    bar = slot * 0.8 / len(series)
    for g, group in enumerate(groups):
        for s, (role, values) in enumerate(series):
            left = plot.left + g * slot + slot * 0.1 + s * bar
            plot.add(f'<rect x="{num(left)}" y="{num(y(values[g]))}" width="{num(bar)}" height="{num(plot.bottom - y(values[g]))}" fill="{colors[role]}"/>')
            plot.add(f'<text x="{num(left + bar / 2)}" y="{num(y(values[g]) - 4)}" text-anchor="middle" fill="{INK}">{values[g]:.3f}</text>')
        plot.add(f'<text x="{num(plot.left + g * slot + slot / 2)}" y="{plot.bottom + 17}" text-anchor="middle" fill="{GREY}">{group}</text>')
    plot.y_axis(y, "RMSE (°C)")
    return "figures/fit_quality.svg", plot.to_bytes(), names


def compare_figure(published: dict[str, Answer]) -> Built | None:
    """Différence moyenne forward − inversion par capteur (barres de part et d'autre de zéro) et sa RMSE en barre d'erreur."""
    answer = published.get("compare.json")
    if answer is None:
        return None
    sensors = [str(s).upper() for s in _array(answer, "sensor")]
    mean, rmse, total = _array(answer, "mean_difference").astype(float), _array(answer, "rmse").astype(float), _number(answer, "rmse_total")
    plot = Plot("Forward minus inversion", f"total RMSE of the difference {total:.3f} °C",
                width=560, height=300, left=64, right=20,
                desc="Mean difference between the forward and the inversion models at each thermometer, with its RMSE, from compare.json.")
    plot.legend((BLUE, "mean difference (forward − inversion)", "box"), (INK, "± RMSE of the difference", "line"))
    plot.plot_height(170)
    extent = float(np.max(np.abs(mean)) + np.max(rmse)) * 1.2 or 1e-12
    plot.x_scale(0.0, 1.0)
    y = plot.y_scale(-extent, extent)
    slot = (plot.right - plot.left) / len(sensors)
    for i, name in enumerate(sensors):
        left = plot.left + i * slot + slot * 0.25
        centre = left + slot * 0.25
        value = float(mean[i])
        plot.add(f'<rect x="{num(left)}" y="{num(min(y(value), y(0.0)))}" width="{num(slot * 0.5)}" height="{num(abs(y(value) - y(0.0)))}" fill="{BLUE}"/>')
        plot.add(f'<path d="M{num(centre)} {num(y(value - rmse[i]))} V{num(y(value + rmse[i]))} M{num(centre - 4)} {num(y(value - rmse[i]))} h8 M{num(centre - 4)} {num(y(value + rmse[i]))} h8" fill="none" stroke="{INK}" stroke-width="1.5"/>')
        label_y = y(max(value + rmse[i], 0.0)) - 5
        plot.add(f'<text x="{num(centre + 8)}" y="{num(label_y)}" fill="{INK}">{minus(f"{value:.3f}")}</text>')
        plot.add(f'<text x="{num(plot.left + i * slot + slot / 2)}" y="{plot.bottom + 17}" text-anchor="middle" fill="{GREY}">{name}</text>')
    plot.reference_line(0.0, "", "y", GREY)
    plot.y_axis(y, "forward − inversion (°C)")
    return "figures/compare.svg", plot.to_bytes(), ["compare.json"]


# Ensemble -----------------------------------------------------------------------------------------------------------


def build_figures(published: dict[str, Answer], chains: dict | None = None) -> list[Built]:
    """Les figures du rapport, dans l'ordre fixe : (chemin, octets SVG, entrées sources). `published` : `{nom de fichier:
    réponse}` des entrées publiées ; `chains` : le dictionnaire de `chains.npz`, ou None. Une figure dont une source manque
    n'est pas produite."""
    built: list[Built | None] = [water_flux_figure(published)]
    for make in (temperature_field_figure, temperature_depths_figure, water_flux_field_figure):
        built += [make(published, role) for role in ROLES]
    built += [posterior_figure(published), chains_figure(published, chains), fit_quality_figure(published), compare_figure(published)]
    return [figure for figure in built if figure is not None]
