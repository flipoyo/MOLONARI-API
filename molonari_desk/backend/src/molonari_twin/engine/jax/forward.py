"""Modèle direct (écoulement + chaleur 1D) en JAX.

Portage fidèle de `pyheatmy` (core.Column.compute_solve_transi et
linear_system.H_stratified / T_stratified) :
- charge hydraulique H, puis température T, schéma semi-implicite (alpha = 0,6) ;
- conditions aux limites : rivière en haut (charge et température mesurées),
  aquifère en bas (charge nulle, température du thermomètre le plus profond) ;
- traitement des interfaces entre couches identique à pyheatmy.

La référence est pyheatmy 0.9.7. Comme lui, le cas monocouche corrige une
« interface » fictive à la cellule nb_cells // 2 ; les conductivités y étant égales
des deux côtés, cette correction ne change rien.

Tout ce qui ne dépend pas des paramètres physiques est précalculé dans
`Geometry` ; `make_forward` renvoie une fonction pure des paramètres par couche,
compilable (`jit`) et vectorisable sur les chaînes MCMC (`vmap`).
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from ..column import ZERO_CELSIUS, ColumnInputs
from ..results import ForwardResult
from . import device  # noqa: F401  (active le float64)

RHO_W = 1000.0
C_W = 4185.0
G = 9.81
MU = 1e-3
LAMBDA_W = 0.6071
HEAT_REFERENCE_C = 0.0  # température de référence du flux advectif ρw·cw·q·(T − T_ref), en °C
ALPHA = 0.6
EPSILON = 1e-10


@dataclass(frozen=True, eq=False)
class Geometry:
    """Données statiques d'une colonne discrétisée (indépendantes des paramètres)."""

    nb_cells: int
    dz: float
    z_solve: np.ndarray  # (C,)
    height: float  # profondeur de la colonne (m)
    sensor_index: np.ndarray  # (3,)
    dt: np.ndarray  # (T−1,)
    head_river: np.ndarray  # (T,)
    t_river: np.ndarray  # K, (T,)
    t_aquifer: np.ndarray  # K, (T,)
    t_init: np.ndarray  # K, (C,)
    observed: np.ndarray  # K, (3, T)
    nb_layers: int
    layer_z_low: np.ndarray  # (L,)
    cell_layer: np.ndarray  # couche de chaque cellule, −1 si non couverte, (C,)
    interfaces: np.ndarray  # (I, 2) indices de cellules (convention pyheatmy « inter_cara »)
    interface_z: np.ndarray  # (I,)
    segment: np.ndarray  # couche utilisée pour la charge initiale de chaque cellule, (C,)


def build_geometry(inputs: ColumnInputs, z_lows, nb_cells: int) -> Geometry:
    z_lows = np.asarray(z_lows, np.float64)
    real_z = inputs.real_depths
    z_max = real_z[-1]
    dz = z_max / nb_cells
    z_solve = dz / 2 + np.arange(nb_cells) * dz
    sensor_index = np.array([np.argmin(np.abs(z - z_solve)) for z in real_z[1:-1]])

    t_river = inputs.t_river + ZERO_CELSIUS
    t_aquifer = inputs.t_aquifer + ZERO_CELSIUS
    t_sensors = inputs.t_sensors + ZERO_CELSIUS
    profile0 = np.array([t_river[0], *t_sensors[0], t_aquifer[0]])
    t_init = np.interp(z_solve, real_z, profile0)

    # Couche de chaque cellule, comme pyheatmy.layers.getListParameters.
    dz_layers = z_lows[-1] / nb_cells
    centers = np.linspace(dz_layers / 2, z_lows[-1] - dz_layers / 2, nb_cells)
    cell_layer = np.full(nb_cells, -1, np.int64)
    previous = 0.0
    for index, z_low in enumerate(z_lows):
        cell_layer[(centers > previous) & (centers <= z_low)] = index
        previous = z_low

    if len(z_lows) == 1:
        interfaces = np.array([[nb_cells // 2, 0]], np.int64)
        interface_z = np.array([0.2])
        segment = np.zeros(nb_cells, np.int64)
    else:
        interface_z = z_lows[:-1]
        interfaces = np.zeros((len(interface_z), 2), np.int64)
        for k, z_low in enumerate(interface_z):
            for i in range(nb_cells - 1):
                if z_solve[i] <= z_low < z_solve[i + 1]:
                    if abs(z_solve[i] - z_low) < EPSILON:
                        interfaces[k] = (i, 0)
                    elif abs(z_solve[i + 1] - z_low) < EPSILON:
                        interfaces[k] = (i + 1, 0)
                    else:
                        interfaces[k] = (i, i + 1)
        # Découpage des cellules par couche pour la charge initiale (boucle de pyheatmy).
        segment = np.full(nb_cells, len(z_lows) - 1, np.int64)
        count = 0
        for index in range(len(z_lows) - 1):
            start = count
            while count < nb_cells:
                if count + 1 < nb_cells and z_solve[count] <= z_lows[index] < z_solve[count + 1]:
                    segment[start : count + 1] = index
                    count += 1
                    break
                count += 1
        segment[count:] = len(z_lows) - 1

    return Geometry(
        nb_cells=nb_cells,
        dz=float(dz),
        z_solve=z_solve,
        height=float(abs(z_max - real_z[0])),
        sensor_index=sensor_index,
        dt=inputs.time_steps(),
        head_river=inputs.head.astype(np.float64),
        t_river=t_river,
        t_aquifer=t_aquifer,
        t_init=t_init,
        observed=t_sensors.T.copy(),
        nb_layers=len(z_lows),
        layer_z_low=z_lows,
        cell_layer=cell_layer,
        interfaces=interfaces,
        interface_z=interface_z,
        segment=segment,
    )


def _tri_product(lower, diag, upper, x):
    """Produit d'une matrice tridiagonale (diagonales inférieure, principale, supérieure) par x."""
    out = diag * x
    out = out.at[1:].add(lower * x[:-1])
    return out.at[:-1].add(upper * x[1:])


def _tri_solve(lower, diag, upper, rhs):
    dl = jnp.concatenate([jnp.zeros(1), lower])
    du = jnp.concatenate([upper, jnp.zeros(1)])
    return jax.lax.linalg.tridiagonal_solve(dl, diag, du, rhs[:, None])[:, 0]


def make_forward(geom: Geometry):
    """Construit la fonction `params (L, 4) -> champs` pour une géométrie donnée.

    params[l] = (k [m²], n, λs [W/m/K], ρs·cs [J/m³/K]) pour la couche l.
    Retourne un dictionnaire : temperature (K), water_flux, head (C, T) et lambda_m.
    """
    n_cell = geom.nb_cells
    dz = geom.dz
    dz2 = dz**2
    alpha = ALPHA
    z = jnp.asarray(geom.z_solve)
    dt_all = jnp.asarray(geom.dt)
    h_riv = jnp.asarray(geom.head_river)
    t_riv = jnp.asarray(geom.t_river)
    t_aq = jnp.asarray(geom.t_aquifer)
    t_init = jnp.asarray(geom.t_init)
    cell_layer = np.asarray(geom.cell_layer)
    covered = jnp.asarray(cell_layer >= 0)
    cell_layer_idx = jnp.asarray(np.maximum(cell_layer, 0))
    single = geom.nb_layers == 1
    interfaces = [tuple(int(v) for v in row) for row in geom.interfaces]
    interface_z = [float(v) for v in geom.interface_z]
    z_np = geom.z_solve

    def forward(params):
        params = jnp.asarray(params, jnp.float64)
        per_cell = jnp.where(covered[:, None], params[cell_layer_idx], 0.0)
        k, n, lambda_s, rhos_cs = (per_cell[:, i] for i in range(4))
        k_layer = params[:, 0]

        k_list = RHO_W * G * k / MU
        ss_list = n / geom.height
        # Conductivités par couche utilisées aux interfaces.
        array_k = RHO_W * G * k_layer / MU
        if single:
            array_k = jnp.concatenate([array_k, array_k])

        # Charge initiale.
        if single:
            h_init = h_riv[0] - h_riv[0] * z / geom.height
        else:
            n_if = geom.nb_layers
            z_lows = geom.layer_z_low
            eps = np.diff(np.concatenate([[0.0], z_lows]))
            size = n_if - 1
            left = jnp.zeros((size, size))
            right = jnp.zeros((size, size))
            left = left.at[0, 0].set(-array_k[0] / eps[0])
            for d in range(1, size):
                left = left.at[d, d - 1].set(array_k[d] / eps[d])
                left = left.at[d, d].set(-array_k[d] / eps[d])
            right = right.at[size - 1, size - 1].set(array_k[-1] / eps[-1])
            for d in range(size - 1):
                right = right.at[d, d + 1].set(-array_k[d + 1] / eps[d + 1])
                right = right.at[d, d].set(array_k[d + 1] / eps[d + 1])
            scalar_left = jnp.zeros(size).at[0].set(array_k[0] * h_riv[0] / eps[0])
            h_sol = jnp.linalg.solve(right - left, scalar_left)
            h_inter = jnp.concatenate([h_riv[:1], h_sol, jnp.zeros(1)])
            seg = np.asarray(geom.segment)
            z_prev = np.concatenate([[0.0], z_lows[:-1]])[seg]
            h_init = h_inter[seg] - (h_inter[seg] - h_inter[seg + 1]) / eps[seg] * (z - z_prev)

        def h_diagonals(dt):
            lower_b = (k_list[1:] * (1 - alpha) / dz2).at[-1].set(4 * k_list[-1] * (1 - alpha) / (3 * dz2))
            diag_b = ss_list / dt - 2 * k_list * (1 - alpha) / dz2
            diag_b = diag_b.at[0].set(ss_list[0] / dt - 4 * k_list[0] * (1 - alpha) / dz2)
            diag_b = diag_b.at[-1].set(ss_list[-1] / dt - 4 * k_list[-1] * (1 - alpha) / dz2)
            upper_b = (k_list[:-1] * (1 - alpha) / dz2).at[0].set(4 * k_list[0] * (1 - alpha) / (3 * dz2))
            lower_a = (-k_list[1:] * alpha / dz2).at[-1].set(-4 * k_list[-1] * alpha / (3 * dz2))
            diag_a = ss_list / dt + 2 * k_list * alpha / dz2
            diag_a = diag_a.at[0].set(ss_list[0] / dt + 4 * k_list[0] * alpha / dz2)
            diag_a = diag_a.at[-1].set(ss_list[-1] / dt + 4 * k_list[-1] * alpha / dz2)
            upper_a = (-k_list[:-1] * alpha / dz2).at[0].set(-4 * k_list[0] * alpha / (3 * dz2))
            # Correction aux interfaces (pyheatmy : correct_numerical_schema).
            for idx, (i0, i1) in enumerate(interfaces):
                k1, k2 = array_k[idx], array_k[idx + 1]
                ss0 = ss_list[i0]
                if i1 == 0:
                    diag_b = diag_b.at[i0].set(ss0 / dt - (k1 + k2) * (1 - alpha) / dz2)
                    lower_b = lower_b.at[i0 - 1].set(k1 * (1 - alpha) / dz2)
                    upper_b = upper_b.at[i0].set(k2 * (1 - alpha) / dz2)
                    diag_a = diag_a.at[i0].set(ss0 / dt + (k1 + k2) * alpha / dz2)
                    lower_a = lower_a.at[i0 - 1].set(-k1 * alpha / dz2)
                    upper_a = upper_a.at[i0].set(-k2 * alpha / dz2)
                else:
                    x = (interface_z[idx] - z_np[i0]) / (z_np[i0 + 1] - z_np[i0])
                    keq = 1 / (x / k1 + (1 - x) / k2)
                    diag_b = diag_b.at[i0].set(ss0 / dt - (k1 + keq) * (1 - alpha) / dz2)
                    lower_b = lower_b.at[i0 - 1].set(k1 * (1 - alpha) / dz2)
                    upper_b = upper_b.at[i0].set(keq * (1 - alpha) / dz2)
                    diag_a = diag_a.at[i0].set(ss0 / dt + (k1 + keq) * alpha / dz2)
                    lower_a = lower_a.at[i0 - 1].set(-k1 * alpha / dz2)
                    upper_a = upper_a.at[i0].set(-keq * alpha / dz2)
                    diag_b = diag_b.at[i0 + 1].set(ss_list[i0 + 1] / dt - (k2 + keq) * (1 - alpha) / dz2)
                    lower_b = lower_b.at[i0].set(keq * (1 - alpha) / dz2)
                    upper_b = upper_b.at[i0 + 1].set(k2 * (1 - alpha) / dz2)
                    diag_a = diag_a.at[i0 + 1].set(ss_list[i0 + 1] / dt + (k2 + keq) * alpha / dz2)
                    lower_a = lower_a.at[i0].set(-keq * alpha / dz2)
                    upper_a = upper_a.at[i0 + 1].set(-k2 * alpha / dz2)
            return lower_b, diag_b, upper_b, lower_a, diag_a, upper_a

        def h_step(h, xs):
            dt, hr0, hr1 = xs
            lower_b, diag_b, upper_b, lower_a, diag_a, upper_a = h_diagonals(dt)
            c = jnp.zeros(n_cell).at[0].set(8 * k_list[0] / (3 * dz2) * (alpha * hr1 + (1 - alpha) * hr0))
            h_next = _tri_solve(lower_a, diag_a, upper_a, _tri_product(lower_b, diag_b, upper_b, h) + c)
            return h_next, h_next

        _, h_steps = jax.lax.scan(h_step, h_init, (dt_all, h_riv[:-1], h_riv[1:]))
        head = jnp.concatenate([h_init[:, None], h_steps.T], axis=1)

        nabla = jnp.zeros_like(head)
        nabla = nabla.at[0].set(2 * (head[1] - h_riv) / (3 * dz))
        nabla = nabla.at[1:-1].set((head[2:] - head[:-2]) / (2 * dz))
        nabla = nabla.at[-1].set(2 * (0.0 - head[-2]) / (3 * dz))

        if single:
            k_scalar = k_layer[0] * RHO_W * G / MU
            flows = -k_scalar * nabla
        else:
            k_flow = k * RHO_W * G / MU
            flows = -k_flow[:, None] * nabla
            for idx, (i0, i1) in enumerate(interfaces):
                if i1 == 0:
                    nabla = nabla.at[i0].set(jnp.where(k_flow[i0] == k_flow[i0 + 1], nabla[i0 + 1], nabla[i0]))
                    nabla = nabla.at[i0].set(jnp.where(k_flow[i0] == k_flow[i0 - 1], nabla[i0 - 1], nabla[i0]))
                    flows = flows.at[i0].set(-k_flow[i0] * nabla[i0])
                else:
                    nabla = nabla.at[i0].set((head[i0 + 1] - head[i0]) / dz)
                    nabla = nabla.at[i1].set((head[i1] - head[i1 - 1]) / dz)
                    x = (interface_z[idx] - z_np[i0]) / (z_np[i1] - z_np[i0])
                    k_flow = k_flow.at[i0].set(1 / (x / k_flow[i0] + (1 - x) / k_flow[i0 + 1]))
                    flows = flows.at[i0].set(-k_flow[i0] * nabla[i0])
                    flows = flows.at[i1].set(-k_flow[i0] * nabla[i1])

        rho_mc = n * RHO_W * C_W + (1 - n) * rhos_cs
        lambda_m = (n * LAMBDA_W**0.5 + (1.0 - n) * lambda_s**0.5) ** 2
        ke = lambda_m / rho_mc
        ae = RHO_W * C_W * k_list / rho_mc

        def t_step(temp, xs):
            dt, grad, tr0, tr1, ta0, ta1 = xs
            lower_b = ke[1:] * (1 - alpha) / dz2 - ((1 - alpha) * ae[1:] / (2 * dz)) * grad[1:]
            lower_b = lower_b.at[-1].set(4 * ke[-1] * (1 - alpha) / (3 * dz2) - (2 * (1 - alpha) * ae[-1] / (3 * dz)) * grad[-1])
            diag_b = 1 / dt - 2 * ke * (1 - alpha) / dz2
            diag_b = diag_b.at[0].set(1 / dt - 4 * ke[0] * (1 - alpha) / dz2)
            diag_b = diag_b.at[-1].set(1 / dt - 4 * ke[-1] * (1 - alpha) / dz2)
            upper_b = ke[:-1] * (1 - alpha) / dz2 + ((1 - alpha) * ae[:-1] / (2 * dz)) * grad[:-1]
            upper_b = upper_b.at[0].set(4 * ke[0] * (1 - alpha) / (3 * dz2) + (2 * (1 - alpha) * ae[0] / (3 * dz)) * grad[0])
            c = jnp.zeros(n_cell)
            c = c.at[0].set(
                (8 * ke[0] * alpha / (3 * dz2) - 2 * alpha * ae[0] * grad[0] / (3 * dz)) * tr1
                + (8 * ke[0] * (1 - alpha) / (3 * dz2) - 2 * (1 - alpha) * ae[0] * grad[0] / (3 * dz)) * tr0
            )
            c = c.at[-1].set(
                (8 * ke[-1] * alpha / (3 * dz2) + 2 * alpha * ae[-1] * grad[-1] / (3 * dz)) * ta1
                + (8 * ke[-1] * (1 - alpha) / (3 * dz2) + 2 * (1 - alpha) * ae[-1] * grad[-1] / (3 * dz)) * ta0
            )
            lower_a = -(ke[1:] * alpha / dz2) + (alpha * ae[1:] / (2 * dz)) * grad[1:]
            lower_a = lower_a.at[-1].set(-4 * ke[-1] * alpha / (3 * dz2) + (2 * alpha * ae[-1] / (3 * dz)) * grad[-1])
            diag_a = 1 / dt + 2 * ke * alpha / dz2
            diag_a = diag_a.at[0].set(1 / dt + 4 * ke[0] * alpha / dz2)
            diag_a = diag_a.at[-1].set(1 / dt + 4 * ke[-1] * alpha / dz2)
            upper_a = -(ke[:-1] * alpha / dz2) - (alpha * ae[:-1] / (2 * dz)) * grad[:-1]
            upper_a = upper_a.at[0].set(-4 * ke[0] * alpha / (3 * dz2) - (2 * alpha * ae[0] / (3 * dz)) * grad[0])
            t_next = _tri_solve(lower_a, diag_a, upper_a, _tri_product(lower_b, diag_b, upper_b, temp) + c)
            return t_next, t_next

        xs = (dt_all, nabla[:, :-1].T, t_riv[:-1], t_riv[1:], t_aq[:-1], t_aq[1:])
        _, t_steps = jax.lax.scan(t_step, t_init, xs)
        temperature = jnp.concatenate([t_init[:, None], t_steps.T], axis=1)

        return {
            "temperature": temperature,
            "water_flux": flows,
            "head": head,
            "lambda_m": lambda_m,
        }

    return forward


def conductive_flux(temperature, lambda_m, dz):
    """Flux conductif λm·∂T/∂z (bords recopiés), comme Column.get_conduc_flows_solve."""
    grad = jnp.zeros_like(temperature)
    grad = grad.at[1:-1].set((temperature[2:] - temperature[:-2]) / (2 * dz))
    grad = grad.at[0].set(grad[1])
    grad = grad.at[-1].set(grad[-2])
    return lambda_m[:, None] * grad


def rmse(temperature, geom: Geometry):
    """RMSE aux 3 capteurs puis total, comme Column.get_RMSE (même unité que les entrées)."""
    simulated = temperature[geom.sensor_index]
    per_sensor = jnp.sqrt(jnp.nansum((simulated - geom.observed) ** 2, axis=1) / geom.observed.shape[1])
    total = jnp.sqrt(jnp.nansum(per_sensor**2) / geom.observed.shape[0])
    return jnp.append(per_sensor, total)


def energy(temperature, geom: Geometry, sigma2: float):
    """Énergie (−log vraisemblance gaussienne), identique à pyheatmy.core.compute_energy
    avec une loi a priori constante sur σ²."""
    residual = temperature[geom.sensor_index] - geom.observed
    return geom.observed.size / 2 * jnp.log(sigma2) + jnp.sum(residual**2) / (2 * sigma2)


def run_forward(inputs: ColumnInputs, layers_physical, z_lows, nb_cells: int) -> ForwardResult:
    """Modèle direct complet pour une configuration, résultats en unités d'affichage (°C)."""
    geom = build_geometry(inputs, z_lows, nb_cells)
    fields = jax.jit(make_forward(geom))(jnp.asarray(layers_physical, jnp.float64))
    return to_forward_result(fields, geom, inputs)


def to_forward_result(fields, geom: Geometry, inputs: ColumnInputs) -> ForwardResult:
    temperature = fields["temperature"]
    water_flux = fields["water_flux"]
    advective = RHO_W * C_W * water_flux * (temperature - (ZERO_CELSIUS + HEAT_REFERENCE_C))
    conductive = conductive_flux(temperature, fields["lambda_m"], geom.dz)
    temperature_np = np.asarray(temperature)
    if not np.all(np.isfinite(temperature_np)) or not np.all(np.isfinite(np.asarray(water_flux))):
        from ...errors import UserError

        raise UserError("E_MODEL_NON_FINITE")
    return ForwardResult(
        depths=geom.z_solve.copy(),
        times=inputs.times,
        temperature=temperature_np - ZERO_CELSIUS,
        water_flux=np.asarray(water_flux),
        advective_flux=np.asarray(advective),
        conductive_flux=np.asarray(conductive),
        sensor_index=geom.sensor_index.copy(),
        observed=inputs.t_sensors,
        rmse=np.asarray(rmse(temperature, geom)),
    )
