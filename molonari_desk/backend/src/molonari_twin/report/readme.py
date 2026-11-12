"""Page lisible du rapport public (F1) : `README.md`, en français, avec les figures et leurs constats.

`readme(index, findings, published)` écrit, pour chaque figure produite et dans l'ordre de l'index, un titre, l'image en
lien relatif, une légende et les constats qui portent sur cette figure (modèles `I_…` de `errors.MESSAGES`, traduits),
puis la section « Limites ». La langue est fixe (`"fr"`) : la page est un fichier du rapport, pas une sortie de la
ligne de commande, donc ses octets ne dépendent ni de `MOLONARI_LANG` ni de la locale. Les nombres sont mis en forme à la
française (virgule décimale, `× 10⁻⁹` en exposants Unicode, pourcentages entiers).

Seuls ce module et `interpretation.py` choisissent les mots : aucun seuil, aucun verdict. Aucune date autre que
`index["generated_at"]` (la date du calcul), aucun chemin hors des liens relatifs vers les figures.

Tous les textes sont des chaînes sources anglaises marquées `N_()` (ou des modèles de `MESSAGES`), traduites par `translate`.
"""

from __future__ import annotations

import math
import re

import numpy as np

from ..errors import MESSAGES
from ..i18n import N_, translate
from ..questions.answer import Answer
from .figures import format_time

LANGUAGE = "fr"
SUPERSCRIPT = str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻")
MINUS = "−"

MODELS = {"forward": N_("forward model"), "inversion": N_("inversion model")}

# type de figure → (titre, légende). `{model}` : le modèle (figures par modèle).
FIGURES = {
    "water_flux": (
        N_("Water flux through the riverbed"),
        N_("One line per model on a common axis. The dot is the mean flux, the grey line spans the 5th to 95th percentile of "
           "q over time and, for the inversion, the blue bar is the posterior interval of the mean. A positive q is an "
           "infiltration (the river feeds the aquifer), a negative q an exfiltration."),
    ),
    "temperature_field": (
        N_("Temperature field — {model}"),
        N_("Simulated temperature as a function of depth (downwards) and date, for the best fit. The dashes on the right "
           "mark the depths of the thermometers T1 to T3 and of the aquifer thermometer T4."),
    ),
    "temperature_depths": (
        N_("Temperature at depth and date — {model}"),
        N_("Top: temperature over time at up to ten depths, coloured by depth. Bottom: temperature profile against depth "
           "at five dates."),
    ),
    "water_flux_field": (
        N_("Water flux field — {model}"),
        N_("Top: water flux as a function of depth and date, red for an infiltration (q > 0) and blue for an exfiltration "
           "(q < 0). Bottom: the flux in the cell at the bottom of the river, with the q = 0 line."),
    ),
    "posterior": (
        N_("Posterior distribution of the soil parameters"),
        N_("Histogram of the posterior samples of each inverted parameter and of log10 K, per layer. The dashed line is the "
           "median and the red bar the 90 % interval; the horizontal axis spans the sampled range."),
    ),
    "chains": (
        N_("Chains of the inversion"),
        N_("Energy and parameter values along the iterations, one colour per chain (dashed beyond seven chains), with the "
           "largest R̂ and the smallest effective sample size."),
    ),
    "fit_quality": (
        N_("Temperature misfit (RMSE)"),
        N_("Root mean square error of the simulated temperature at each thermometer and in total, for the forward model and "
           "for the best sample of the inversion."),
    ),
    "compare": (
        N_("Forward minus inversion"),
        N_("Mean difference of the simulated temperatures of the two models at each thermometer; the error bar is the RMSE "
           "of the difference."),
    ),
}

FIGURE_NAME = re.compile(r"^figures/(?P<kind>[a-z_]+?)(?:-(?P<role>forward|inversion))?\.svg$")


def fr(text: str) -> str:
    return translate(text, LANGUAGE)


def french(value: float, significant: int = 3) -> str:
    """Un nombre à la française : virgule décimale, signe moins typographique, `× 10⁻⁹` hors de [1e-3, 1e5[."""
    if value == 0:
        return "0"
    exponent = math.floor(math.log10(abs(value)))
    if -3 <= exponent < 5:
        text = f"{value:.{significant}g}"
        if "e" in text:
            text = f"{value:.0f}"
        return text.replace("-", MINUS).replace(".", ",")
    mantissa = f"{value / 10**exponent:.{significant - 1}g}"
    if abs(float(mantissa)) >= 10:
        exponent += 1
        mantissa = f"{value / 10**exponent:.{significant - 1}g}"
    return f"{mantissa.replace('-', MINUS).replace('.', ',')} × 10{str(exponent).translate(SUPERSCRIPT)}"


def format_value(key: str, value) -> str:
    """Un paramètre d'un constat, mis en forme pour le modèle de message."""
    if key == "model":
        return fr(MODELS[value])
    if key in ("share",):
        return str(round(value))
    if key in ("level", "chains", "burn_in"):
        return str(int(value))
    if key == "exponent":
        return str(int(value)).replace("-", MINUS)
    return french(value)


def finding_text(finding: dict) -> str:
    params = {key: format_value(key, value) for key, value in finding["params"].items()}
    return fr(MESSAGES[finding["code"]]).format(**params)


def _period(published: dict[str, Answer]) -> tuple[str, str] | None:
    for name in ("forward-describe.json", "inversion-describe.json", "forward-water_flux.json", "inversion-water_flux.json"):
        answer = published.get(name)
        if answer is not None and "start" in answer.data and "end" in answer.data:
            start, end = answer.data["start"].value, answer.data["end"].value
            if isinstance(start, np.datetime64) and isinstance(end, np.datetime64) and not (np.isnat(start) or np.isnat(end)):
                return format_time(start), format_time(end)  # un twin sans période déclarée n'a pas de date dans `describe`
    return None


def _fields_step(published: dict[str, Answer]) -> list[float]:
    """Pas de temps (s) des champs publiés, un par modèle."""
    steps = []
    for role in MODELS:
        fields = published.get(f"{role}-fields.npz")
        if fields is not None:
            seconds = np.asarray(fields.data["time"].value).astype("datetime64[s]").astype(np.int64)
            steps.append(float(np.median(np.diff(seconds))) if len(seconds) > 1 else math.inf)
    return steps


def readme(index: dict, findings: list[dict], published: dict[str, Answer]) -> bytes:
    """La page `README.md` du rapport (en français), à partir de l'index (sans `findings` ni `readme`), des constats et des
    entrées publiées."""
    lines = [f"# {fr(N_('Public report — {point}')).format(point=index['point'])}", ""]
    period = _period(published)
    if period is not None:
        lines += [fr(N_("Period: {start} to {end}.")).format(start=period[0], end=period[1]), ""]
    space = index["space"]
    models = [f"{fr(MODELS[role])} ({space[role]['twin']})" for role in MODELS if space.get(role)]
    lines += [fr(N_("Models: {models}.")).format(models=", ".join(models)), ""]
    lines += [fr(N_("Computed on {date}.")).format(date=index["generated_at"]), ""]
    lines += [fr(N_("Each figure is drawn from the entries published next to it (see report.json), so it can be redrawn.")), ""]

    by_figure: dict[str, list[dict]] = {}
    for finding in findings:
        by_figure.setdefault(finding["figure"], []).append(finding)
    for figure in index["figures"]:
        found = FIGURE_NAME.match(figure["file"])
        title, caption = FIGURES[found["kind"]]
        model = fr(MODELS[found["role"]]) if found["role"] else ""
        heading = fr(title).format(model=model) if "{model}" in title else fr(title)
        lines += [f"## {heading}", "", f"![{heading}]({figure['file']})", "", fr(caption), ""]
        lines += [f"- {finding_text(finding)}" for finding in by_figure.get(figure["file"], [])]
        if by_figure.get(figure["file"]):
            lines.append("")

    lines += [f"## {fr(N_('Limits'))}", ""]
    lines.append(f"- {fr(N_('The forward model has no uncertainty: its parameters are fixed, so it has no interval.'))}")
    steps = _fields_step(published)
    if steps:
        lines.append(f"- {fr(N_('The fields are those of the best fit (a single sample for the inversion), not an average over the posterior.'))}")
        if any(step > 3 * 3600 for step in steps):
            lines.append(f"- {fr(N_('The daily damping of the temperature is not computed: the dates of the fields are more than 3 h apart.'))}")
    else:
        lines.append(f"- {fr(N_('The temperature and water flux fields are not published: the developer keeps the states private.'))}")
    lines.append(f"- {fr(N_('The measurements are not published: this report never contains them.'))}")
    lines.append("")
    return "\n".join(lines).encode("utf-8")


__all__ = ["FIGURES", "readme"]
