"""Onglet « Inversion » : lois a priori, réglages DREAM, suivi en direct, annulation."""

from __future__ import annotations

import html
import random
import re

from PySide6 import QtCore, QtWidgets

from molonaviz2 import facade as api
from molonaviz2.facade import (
    InversionConfig,
    InversionResult,
    LayerSpec,
    UserError,
    default_priors,
    setting_limits,
)

from ...i18n import N_, t
from ...workers.tasks import Task, TaskManager
from ..dialogs.notice import attach
from ..errors import error_text
from ..plots.live import LiveMonitor
from ..widgets import NumberField, format_number, heading, hint
from .context import PointContext
from .layers import LayersTable

# Choix de « Champs stockés » : libellé et valeur de InversionConfig.store_fields.
STORE_FIELDS_CHOICES = ((N_("No fields"), "none"), (N_("At the sensors"), "sensors"), (N_("Full fields"), "full"))

PHASE_LABELS = {
    "compile": N_("Compiling the model…"),
    "burn_in": N_("Burn-in"),
    "sampling": N_("Sampling"),
    "finalize": N_("Computing quantiles and best fit…"),
}


def _duration(seconds: float) -> str:
    seconds = round(seconds)
    if seconds < 60:
        return f"{seconds} s"
    minutes, seconds = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes} min {seconds:02d} s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours} h {minutes:02d} min"


class InversionTab(QtWidgets.QWidget):
    show_results = QtCore.Signal(int)

    def __init__(self, context: PointContext, tasks: TaskManager, parent=None):
        super().__init__(parent)
        self.context = context
        self.tasks = tasks
        self._task: Task | None = None
        self._sampling_started: float | None = None
        defaults = InversionConfig(layers=[LayerSpec("", context.column_depth)])

        self.layers = LayersTable("priors", context.column_depth)
        add = QtWidgets.QPushButton(t("Add a layer"))
        add.clicked.connect(self.layers.add_layer)
        remove = QtWidgets.QPushButton(t("Remove selected layer"))
        remove.clicked.connect(self.layers.remove_selected)
        layer_buttons = QtWidgets.QHBoxLayout()
        layer_buttons.addWidget(add)
        layer_buttons.addWidget(remove)
        layer_buttons.addStretch()

        # Réglages essentiels
        self.sigma_t = NumberField(context.default_sigma_t(), "°C", **setting_limits("sigma_t"))
        self.nb_chains = NumberField(defaults.nb_chains, t("chains"), integer=True, **setting_limits("nb_chains"))
        self.nb_iter = NumberField(defaults.nb_iter, t("iterations"), integer=True, **setting_limits("nb_iter"))
        self.burn_in_max = NumberField(defaults.burn_in_max, t("iterations"), integer=True, **setting_limits("burn_in_max"))
        self.quantiles = QtWidgets.QLineEdit(", ".join(f"{q:g}" for q in defaults.quantiles))
        self.seed = NumberField(defaults.seed, "", integer=True, **setting_limits("seed"))
        new_seed = QtWidgets.QToolButton()
        new_seed.setText(t("New"))
        new_seed.setToolTip(t("Draw a new random seed"))
        new_seed.clicked.connect(lambda: self.seed.set_value(random.randint(0, 2**31 - 1)))
        seed_row = QtWidgets.QHBoxLayout()
        seed_row.addWidget(self.seed, 1)
        seed_row.addWidget(new_seed)
        basic = QtWidgets.QFormLayout()
        basic.addRow(t("Measurement uncertainty σT"), self.sigma_t)
        basic.addRow(t("Number of chains"), self.nb_chains)
        basic.addRow(t("Iterations kept"), self.nb_iter)
        basic.addRow(t("Maximum burn-in"), self.burn_in_max)
        basic.addRow(t("Quantiles"), self.quantiles)
        basic.addRow(t("Random seed"), seed_row)

        # Réglages avancés (repliés)
        self.advanced = QtWidgets.QGroupBox(t("Advanced settings"))
        self.advanced.setCheckable(True)
        self.advanced.setChecked(False)
        self.nb_cells = NumberField(defaults.nb_cells, t("cells"), integer=True, **setting_limits("nb_cells"))
        self.delta = NumberField(defaults.delta, t("pairs"), integer=True, **setting_limits("delta"))
        self.n_cr = NumberField(defaults.n_cr, "", integer=True, **setting_limits("n_cr"))
        self.c = NumberField(defaults.c, "", **setting_limits("c"))
        self.c_star = NumberField(defaults.c_star, "", **setting_limits("c_star"))
        self.gr_threshold = NumberField(defaults.gr_threshold, "", **setting_limits("gr_threshold"))
        self.n_field_samples = NumberField(defaults.n_field_samples, t("fields"), integer=True, **setting_limits("n_field_samples"))
        self.space_step = NumberField(defaults.space_step, "", integer=True, **setting_limits("space_step"))
        self.time_step = NumberField(defaults.time_step, "", integer=True, **setting_limits("time_step"))
        self.thin = NumberField(defaults.thin, "", integer=True, **setting_limits("thin"))
        self.store_fields = QtWidgets.QComboBox()
        for label, value in STORE_FIELDS_CHOICES:
            self.store_fields.addItem(t(label), value)
        self.store_fields.setCurrentIndex(self.store_fields.findData(defaults.store_fields))
        advanced_form = QtWidgets.QFormLayout()
        advanced_form.addRow(t("Discretisation"), self.nb_cells)
        advanced_form.addRow(t("Pairs of chains per jump (δ)"), self.delta)
        advanced_form.addRow(t("Crossover values (nCR)"), self.n_cr)
        advanced_form.addRow("c", self.c)
        advanced_form.addRow("c*", self.c_star)
        advanced_form.addRow(t("Gelman-Rubin threshold"), self.gr_threshold)
        advanced_form.addRow(t("Fields kept for quantiles"), self.n_field_samples)
        advanced_form.addRow(t("Spatial sub-sampling step"), self.space_step)
        advanced_form.addRow(t("Temporal sub-sampling step"), self.time_step)
        advanced_form.addRow(t("Thinning step (thin)"), self.thin)
        advanced_form.addRow(t("Fields stored"), self.store_fields)
        advanced_body = QtWidgets.QWidget()
        advanced_body.setLayout(advanced_form)
        advanced_layout = QtWidgets.QVBoxLayout(self.advanced)
        advanced_layout.addWidget(advanced_body)
        self.advanced.toggled.connect(advanced_body.setVisible)
        advanced_body.setVisible(False)

        restore = QtWidgets.QPushButton(t("Restore defaults"))
        restore.clicked.connect(self.restore_defaults)
        self.issues = QtWidgets.QLabel()
        self.issues.setProperty("role", "error")
        self.issues.setWordWrap(True)
        self.run_button = QtWidgets.QPushButton(t("Run inversion  (F5)"))
        self.run_button.setDefault(True)
        self.run_button.clicked.connect(self.run)
        self.cancel_button = QtWidgets.QPushButton(t("Cancel  (Esc)"))
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self.cancel)
        run_row = QtWidgets.QHBoxLayout()
        run_row.addWidget(restore)
        run_row.addStretch()
        run_row.addWidget(self.cancel_button)
        run_row.addWidget(self.run_button)

        settings = QtWidgets.QWidget()
        side = QtWidgets.QVBoxLayout(settings)
        side.addWidget(heading(t("Layers and prior ranges")))
        side.addWidget(hint(t("Uniform priors between min and max. Permeability as −log₁₀ k (k in m²).")))
        side.addWidget(self.layers)
        side.addLayout(layer_buttons)
        side.addWidget(heading(t("Sampling")))
        side.addLayout(basic)
        side.addWidget(self.advanced)
        self.per_chain = QtWidgets.QLabel()
        self.per_chain.setWordWrap(True)
        self.per_chain.setTextFormat(QtCore.Qt.TextFormat.RichText)
        per_chain_box = QtWidgets.QGroupBox(t("Per chain"))
        per_chain_layout = QtWidgets.QVBoxLayout(per_chain_box)
        per_chain_layout.addWidget(self.per_chain)
        side.addWidget(per_chain_box)
        for widget, key in ((self.sigma_t, "inversion.sigma_t"), (self.nb_chains, "inversion.nb_chains"), (self.nb_iter, "inversion.nb_iter"),
                            (self.burn_in_max, "inversion.burn_in_max"), (self.quantiles, "inversion.quantiles"), (self.seed, "inversion.seed"),
                            (self.nb_cells, "numerics.nb_cells"), (self.delta, "inversion.delta"), (self.n_cr, "inversion.n_cr"),
                            (self.c, "inversion.c"), (self.c_star, "inversion.c_star"), (self.gr_threshold, "inversion.gr_threshold"),
                            (self.n_field_samples, "inversion.n_field_samples"), (self.space_step, "inversion.space_step"),
                            (self.time_step, "inversion.time_step"), (self.thin, "inversion.thin"),
                            (self.store_fields, "inversion.store_fields"), (per_chain_box, "inversion.per_chain")):
            attach(widget, key)
        side.addWidget(self.issues)
        side.addLayout(run_row)
        side.addStretch()
        scroll = QtWidgets.QScrollArea()
        scroll.setWidget(settings)
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(560)

        self.status = QtWidgets.QLabel(t("No inversion running."))
        self.status.setWordWrap(True)
        self.progress = QtWidgets.QProgressBar()
        self.progress.setTextVisible(False)
        self.progress.hide()
        self.monitor = LiveMonitor()
        self.result_button = QtWidgets.QPushButton(t("Open results"))
        self.result_button.setEnabled(False)
        self.result_button.clicked.connect(lambda: self.show_results.emit(self._last_id))
        right = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right)
        right_layout.addWidget(heading(t("Convergence")))
        status_row = QtWidgets.QHBoxLayout()
        status_row.addWidget(self.status, 1)
        status_row.addWidget(self.result_button)
        right_layout.addLayout(status_row)
        right_layout.addWidget(self.progress)
        right_layout.addWidget(self.monitor, 1)

        splitter = QtWidgets.QSplitter()
        splitter.addWidget(scroll)
        splitter.addWidget(right)
        splitter.setStretchFactor(1, 1)
        layout = QtWidgets.QHBoxLayout(self)
        layout.addWidget(splitter)

        self._last_id = -1
        for field in self._fields():
            field.valueChanged.connect(self._validate)
        self.quantiles.textChanged.connect(self._validate)
        self.store_fields.currentIndexChanged.connect(self._validate)
        self.layers.edited.connect(self._validate)
        tasks.task_started.connect(lambda _: self._validate())
        tasks.task_finished.connect(lambda _: self._validate())
        self._nb_times: int | None = None
        context.changed.connect(self._count_dates)
        self._count_dates()
        self._load_defaults()

    def _fields(self) -> list[NumberField]:
        return [self.sigma_t, self.nb_chains, self.nb_iter, self.burn_in_max, self.seed, self.nb_cells, self.delta, self.n_cr,
                self.c, self.c_star, self.gr_threshold, self.n_field_samples, self.space_step, self.time_step, self.thin]

    # Configuration --------------------------------------------------------------------
    def _load_defaults(self) -> None:
        project = self.context.project
        previous = project.computations(self.context.study, self.context.name, "inversion")
        if previous:
            self.load_config(InversionConfig.from_dict(previous[0].config))
            self._last_id = previous[0].id
            self.result_button.setEnabled(True)
            return
        direct = project.computations(self.context.study, self.context.name, "direct")
        if direct:
            layers = [LayerSpec(layer["name"], layer["z_low"]) for layer in direct[0].config["layers"]]
        else:
            layers = [LayerSpec(t("Layer {n}").format(n=1), self.context.column_depth)]
        self.layers.set_layers(layers, [default_priors() for _ in layers])
        self._validate()

    def restore_defaults(self) -> None:
        defaults = InversionConfig(layers=[LayerSpec("", self.context.column_depth)])
        layers = self.layers.layers(strict=False)
        self.layers.set_layers(layers, [default_priors() for _ in layers])
        for name in ("nb_chains", "nb_iter", "burn_in_max", "nb_cells", "delta", "n_cr", "c", "c_star", "gr_threshold",
                     "n_field_samples", "space_step", "time_step", "thin", "seed"):
            getattr(self, name).set_value(getattr(defaults, name))
        self.store_fields.setCurrentIndex(self.store_fields.findData(defaults.store_fields))
        self.sigma_t.set_value(self.context.default_sigma_t())
        self.quantiles.setText(", ".join(f"{q:g}" for q in defaults.quantiles))

    def load_config(self, config: InversionConfig) -> None:
        self.layers.set_layers(config.layers, config.priors)
        for name in ("sigma_t", "nb_chains", "nb_iter", "burn_in_max", "nb_cells", "delta", "n_cr", "c", "c_star",
                     "gr_threshold", "n_field_samples", "space_step", "time_step", "thin", "seed"):
            getattr(self, name).set_value(getattr(config, name))
        self.store_fields.setCurrentIndex(self.store_fields.findData(config.store_fields))
        self.quantiles.setText(", ".join(f"{q:g}" for q in config.quantiles))
        self._validate()

    def _parse_quantiles(self) -> tuple[float, ...] | None:
        """Quantiles séparés par des virgules, points-virgules ou espaces (point décimal)."""
        parts = [p for p in re.split(r"[;,\s]+", self.quantiles.text().strip()) if p]
        try:
            values = tuple(float(p) for p in parts)
        except ValueError:
            return None
        return values or None

    def config(self) -> InversionConfig:
        return InversionConfig(
            layers=self.layers.layers(),
            priors=self.layers.priors(),
            nb_cells=int(self.nb_cells.value()),
            sigma_t=float(self.sigma_t.value()),
            nb_chains=int(self.nb_chains.value()),
            nb_iter=int(self.nb_iter.value()),
            burn_in_max=int(self.burn_in_max.value()),
            delta=int(self.delta.value()),
            n_cr=int(self.n_cr.value()),
            c=float(self.c.value()),
            c_star=float(self.c_star.value()),
            gr_threshold=float(self.gr_threshold.value()),
            quantiles=self._parse_quantiles(),
            n_field_samples=int(self.n_field_samples.value()),
            space_step=int(self.space_step.value()),
            time_step=int(self.time_step.value()),
            thin=int(self.thin.value()),
            store_fields=self.store_fields.currentData(),
            seed=int(self.seed.value()),
        )

    def _count_dates(self) -> None:
        """Nombre de dates des mesures nettoyées (pour la grille des champs), None avant le nettoyage."""
        try:
            self._nb_times = len(self.context.column_inputs().times)
        except UserError:
            self._nb_times = None
        self._show_layout()

    def _show_layout(self) -> None:
        """Encadré « Par chaîne » : ce que donnent les réglages, calculé comme le moteur."""
        try:
            if any(not f.is_valid() for f in self._fields()):
                raise ValueError("invalid settings")
            config = self.config()
        except ValueError:
            self.per_chain.setText(t("Fix the highlighted settings to see what they give per chain."))
            return
        layout = api.sampling_layout(config, self._nb_times)
        lines = [
            t("Burn-in: up to {n} iterations per chain (stops early once R-hat < {threshold}).").format(
                n=layout.burn_in_max, threshold=format_number(self.gr_threshold.value())),
        ]
        if layout.thin == 1:
            lines += [
                t("Kept: {n} iterations per chain, {total} parameter sets in all.").format(n=layout.kept, total=layout.parameter_sets),
                t("Fields (T, q): {k} per chain, one every {m} iterations, {total} in all.").format(
                    k=layout.fields_per_chain, m=layout.interval, total=layout.fields_total),
            ]
        else:
            lines += [
                t("Kept: {n} iterations per chain out of {computed} computed (one in {thin}), {total} parameter sets in all.").format(
                    n=layout.kept, computed=layout.nb_iter, thin=layout.thin, total=layout.parameter_sets),
                t("Fields (T, q): {k} per chain, one every {m} kept iterations, {total} in all.").format(
                    k=layout.fields_per_chain, m=layout.interval, total=layout.fields_total),
            ]
        if layout.fields_total != layout.fields_requested:
            lines[-1] += " " + t("({requested} requested, rounded to a whole number per chain.)").format(requested=layout.fields_requested)
        if layout.field_times is not None:
            lines.append(t("Field grid: {cells} cells × {times} dates, {kept_cells} × {kept_times} after sub-sampling, about {size} MB.").format(
                cells=layout.nb_cells, times=layout.nb_times, kept_cells=layout.field_cells, kept_times=layout.field_times,
                size=format_number(round(layout.field_bytes / 1e6, 1))))
            stored = self.store_fields.currentData()
            if stored == "none":
                lines.append(t("Fields stored in the result: none."))
            else:
                lines.append(t("Fields stored in the result: {kind}, about {size} MB.").format(
                    kind=self.store_fields.currentText(), size=format_number(round(layout.stored_bytes(stored) / 1e6, 1))))
        lines.append(t("Direct-model evaluations: at most {n}.").format(n=layout.max_evaluations))
        self.per_chain.setText("<br>".join(html.escape(line) for line in lines))  # « R-hat < seuil » n'est pas une balise

    def _validate(self) -> None:
        self._show_layout()
        problems = self.layers.validate()
        if any(not f.is_valid() for f in self._fields()):
            problems.append(t("Some settings are invalid (see the highlighted fields)."))
        quantiles = self._parse_quantiles()
        if quantiles is None or any(not 0 < q < 1 for q in quantiles):
            problems.append(t("Quantiles: numbers strictly between 0 and 1, separated by commas."))
        if not problems:
            problems += [error_text(issue) for issue in self.config().validate(self.context.column_depth)]
        if self.tasks.busy:
            problems.append(t("A computation is already running."))
        self.issues.setText("\n".join(dict.fromkeys(problems)))
        self.run_button.setEnabled(not problems)

    # Calcul ---------------------------------------------------------------------------
    def run(self) -> None:
        if not self.run_button.isEnabled():
            return
        config = self.config()
        inputs = self.context.column_inputs()
        task = Task(
            t("Inversion — {point}").format(point=self.context.name),
            lambda ctx: api.run_inversion(inputs, config, on_progress=ctx.report, should_cancel=ctx.cancelled),
        )
        task.signals.progress.connect(self._progress)
        task.signals.succeeded.connect(lambda result: self._finished(task, config, result))
        task.signals.cancelled.connect(self._cancelled)
        task.signals.failed.connect(lambda _: self._stopped(t("The inversion stopped because of an error.")))
        self._task = task
        self._sampling_started = None
        self.monitor.reset(config.gr_threshold, config.nb_chains)
        self.progress.setRange(0, 0)
        self.progress.show()
        self.cancel_button.setEnabled(True)
        self.result_button.setEnabled(False)
        self.status.setText(t(PHASE_LABELS["compile"]))
        self.tasks.start(task)

    def cancel(self) -> None:
        if self._task is not None:
            self._task.cancel()
            self.cancel_button.setEnabled(False)
            self.status.setText(t("Cancelling at the end of the current block…"))

    def _progress(self, progress) -> None:
        self.monitor.update(progress)
        label = t(PHASE_LABELS.get(progress.phase, progress.phase))
        if progress.phase in ("compile", "finalize"):
            self.progress.setRange(0, 0)
            self.status.setText(label)
            return
        self.progress.setRange(0, progress.total)
        self.progress.setValue(progress.iteration)
        text = f"{label} · " + t("iteration {i} / {n} (per chain)").format(i=progress.iteration, n=progress.total) + " · " \
            + t("elapsed {time}").format(time=_duration(progress.elapsed))
        if progress.phase == "burn_in":
            text += " · " + t("stops early once R-hat < threshold")
            if progress.rhat_max == progress.rhat_max:  # pas nan
                text += f" · R-hat {progress.rhat_max:.2f}"
        else:
            if self._sampling_started is None:
                self._sampling_started = progress.elapsed
                self._sampling_first = progress.iteration
            done = progress.iteration - self._sampling_first
            if done > 0:
                rate = (progress.elapsed - self._sampling_started) / done
                text += " · " + t("about {time} left").format(time=_duration(rate * (progress.total - progress.iteration)))
        text += " · " + t("acceptance {rate:.0%}").format(rate=progress.acceptance)
        self.status.setText(text)

    def _finished(self, task: Task, config: InversionConfig, result: InversionResult) -> None:
        self._last_id = self.context.project.add_computation(
            self.context.study, self.context.name, "inversion", config.to_dict(), result.to_arrays(), task.duration, api.engine_version(), api.device()
        )
        converged = t("converged") if result.burn_in_converged else t("not converged (maximum reached)")
        self._stopped(
            t("Done in {time}. Burn-in: {n} iterations, {state}.").format(time=_duration(task.duration), n=result.burn_in_iterations, state=converged)
        )
        self.result_button.setEnabled(True)
        self.context.notify()

    def _cancelled(self) -> None:
        self._stopped(t("Inversion cancelled. Nothing was saved."))

    def _stopped(self, message: str) -> None:
        self._task = None
        self.progress.hide()
        self.cancel_button.setEnabled(False)
        self.status.setText(message)
