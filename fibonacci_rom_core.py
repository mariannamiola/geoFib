"""Core implementation of an interface-aware reservoir Fibonacci ROM.

This module implements a compact, self-contained version of the method used in
our paper. It intentionally contains no HYDRUS or external-solver comparison:
it only constructs a Fibonacci graph, assigns synthetic geotechnical properties,
runs the reduced-order recurrence, and exposes level-aggregated diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import math
import numpy as np


@dataclass(frozen=True)
class Node:
    """Node of the Fibonacci propagation graph."""

    index: int
    level: int
    local_index: int


@dataclass
class Graph:
    """Fibonacci graph with primary and secondary parent maps."""

    levels: List[List[int]]
    nodes: List[Node]
    primary_parent: Dict[int, Optional[int]]
    secondary_parent: Dict[int, Optional[int]]


@dataclass
class SoilProperties:
    """Synthetic node-scale geotechnical properties."""

    porosity: np.ndarray
    kappa: np.ndarray
    tau_c: np.ndarray


@dataclass
class ROMParameters:
    """Numerical parameters of the interface-aware reservoir ROM."""

    c0: float = 0.35
    c_kappa: float = 0.30
    c_phi: float = 0.40
    b_min: float = 0.05
    b_max: float = 0.65
    eta_alpha: Tuple[float, float, float] = (0.95, 1.00, 0.80)
    eta_beta: Tuple[float, float, float] = (0.40, 0.35, 0.30)
    rho_e: float = 0.5
    q_theta: float = 0.03
    dt_h: float = 1.0
    kappa50: float = 3.0e-6
    gamma: float = 1.2
    interface_strength: float = 1.25
    interface_contrast_threshold_log10: float = 0.25
    fill_min_h: float = 4.0
    fill_scale_h: float = 16.0
    drain_h: float = 96.0


def fibonacci_numbers(n: int) -> List[int]:
    """Return Fibonacci numbers F_1, ..., F_n with F_1=F_2=1."""
    if n <= 0:
        return []
    fib = [1]
    if n == 1:
        return fib
    fib.append(1)
    for _ in range(2, n):
        fib.append(fib[-1] + fib[-2])
    return fib


def build_fibonacci_graph(n_levels: int = 7) -> Graph:
    """Construct a Fibonacci graph with levels |V_k| = F_{k+1}.

    Parents are assigned with the proportional ceiling map used in the paper:
    p1(v_{k,j}) = v_{k-1, ceil(j F_k / F_{k+1})},
    p2(v_{k,j}) = v_{k-2, ceil(j F_{k-1} / F_{k+1})}.
    """
    if n_levels < 1:
        raise ValueError("n_levels must be positive")

    fib = fibonacci_numbers(n_levels + 1)
    levels: List[List[int]] = []
    nodes: List[Node] = []
    primary_parent: Dict[int, Optional[int]] = {}
    secondary_parent: Dict[int, Optional[int]] = {}

    index = 0
    for k in range(n_levels):
        count = fib[k]
        level_nodes = []
        for j in range(1, count + 1):
            nodes.append(Node(index=index, level=k, local_index=j))
            level_nodes.append(index)
            index += 1
        levels.append(level_nodes)

    for k, level_nodes in enumerate(levels):
        for node_index in level_nodes:
            j = nodes[node_index].local_index
            if k == 0:
                primary_parent[node_index] = None
                secondary_parent[node_index] = None
                continue

            # fib list is zero-based: fib[k] = F_{k+1}.
            parent1_local = int(math.ceil(j * fib[k - 1] / fib[k]))
            primary_parent[node_index] = levels[k - 1][parent1_local - 1]

            if k >= 2:
                parent2_local = int(math.ceil(j * fib[k - 2] / fib[k]))
                secondary_parent[node_index] = levels[k - 2][parent2_local - 1]
            else:
                secondary_parent[node_index] = None

    return Graph(levels=levels, nodes=nodes, primary_parent=primary_parent, secondary_parent=secondary_parent)


def make_synthetic_layered_properties(graph: Graph, seed: int = 7) -> SoilProperties:
    """Assign a reproducible synthetic sand-to-clay layered profile.

    The profile is deliberately simple and self-contained: conductivity decreases
    with depth and small within-level variability is added to create lateral
    heterogeneity across graph nodes.
    """
    rng = np.random.default_rng(seed)
    n = len(graph.nodes)
    phi = np.zeros(n)
    kappa = np.zeros(n)

    # Seven representative levels, from sandy top layers to clay-rich deeper layers.
    phi_by_level = np.array([0.43, 0.41, 0.43, 0.41, 0.43, 0.36, 0.38])
    kappa_by_level = np.array([8.25e-5, 1.23e-5, 2.89e-6, 7.22e-7, 1.94e-7, 5.56e-8, 5.56e-8])

    for node in graph.nodes:
        k = min(node.level, len(phi_by_level) - 1)
        phi[node.index] = np.clip(phi_by_level[k] + rng.normal(0.0, 0.015), 0.25, 0.55)
        kappa[node.index] = kappa_by_level[k] * 10 ** rng.normal(0.0, 0.08)

    # Simple pressure threshold: less conductive nodes have higher threshold.
    logk = np.log10(kappa)
    k_norm = (logk - logk.min()) / (logk.max() - logk.min() + 1e-12)
    tau_c = 0.25 + 0.50 * (1.0 - k_norm)
    return SoilProperties(porosity=phi, kappa=kappa, tau_c=tau_c)


def gaussian_forcing(n_steps: int = 49, peak: float = 1.0, center_h: float = 16.0, sigma_h: float = 4.0, dt_h: float = 1.0) -> np.ndarray:
    """Gaussian storm forcing sampled at uniform time steps."""
    t = np.arange(n_steps, dtype=float) * dt_h
    return peak * np.exp(-0.5 * ((t - center_h) / sigma_h) ** 2)


def harmonic_mean(a: float, b: float) -> float:
    """Harmonic mean used as the unnormalised hydraulic edge weight."""
    return 2.0 * a * b / (a + b + 1e-30)


def projection_unit_interval(x: np.ndarray | float) -> np.ndarray | float:
    """Projection Pi_[0,1](x)=min(1,max(0,x))."""
    return np.minimum(1.0, np.maximum(0.0, x))


def coefficient_vectors(properties: SoilProperties, params: ROMParameters) -> Tuple[np.ndarray, np.ndarray]:
    """Compute alpha and beta coefficient vectors from porosity and conductivity."""
    logk = np.log10(properties.kappa)
    k_norm = (logk - logk.min()) / (logk.max() - logk.min() + 1e-12)
    base = params.c0 + params.c_kappa * k_norm * (1.0 - params.c_phi * properties.porosity)
    base = np.clip(base, params.b_min, params.b_max)
    alpha = base[:, None] * np.asarray(params.eta_alpha, dtype=float)[None, :]
    beta = base[:, None] * np.asarray(params.eta_beta, dtype=float)[None, :]

    # Conservative bounded-gain rescaling for the feed-forward pressure/proxy update.
    total = alpha + beta
    scale = np.minimum(1.0, 0.98 / np.maximum(total, 1e-12))
    alpha *= scale
    beta *= scale
    return alpha, beta


def normalised_parent_weights(graph: Graph, properties: SoilProperties, node_index: int) -> Tuple[float, float]:
    """Return normalised weights for the primary and secondary parents."""
    p1 = graph.primary_parent[node_index]
    p2 = graph.secondary_parent[node_index]
    if p1 is None:
        return 0.0, 0.0
    if p2 is None:
        return 1.0, 0.0
    w1 = harmonic_mean(properties.kappa[p1], properties.kappa[node_index])
    w2 = harmonic_mean(properties.kappa[p2], properties.kappa[node_index])
    return w1 / (w1 + w2 + 1e-30), w2 / (w1 + w2 + 1e-30)


def interface_transmission(kappa_parent: float, kappa_child: float, params: ROMParameters) -> Tuple[float, float]:
    """Absolute transmission g and interface factor chi for one parent-child edge."""
    w = harmonic_mean(kappa_parent, kappa_child)
    g = (w / (w + params.kappa50)) ** params.gamma
    contrast = abs(math.log10(kappa_child / kappa_parent))
    chi = 1.0 + params.interface_strength * max(0.0, contrast - params.interface_contrast_threshold_log10)
    return g, chi


def run_rom(graph: Graph, properties: SoilProperties, forcing: np.ndarray, params: ROMParameters) -> Dict[str, np.ndarray]:
    """Run the interface-aware reservoir Fibonacci ROM."""
    n_nodes = len(graph.nodes)
    n_steps = len(forcing)
    theta = np.zeros((n_steps, n_nodes))
    pressure = np.zeros((n_steps, n_nodes))
    erosion_proxy = np.zeros((n_steps, n_nodes))
    alpha, beta = coefficient_vectors(properties, params)

    root = graph.levels[0][0]
    for t in range(n_steps - 1):
        # Surface boundary. The reservoir state is bounded; pressure/proxy remain signals.
        theta[t, root] = projection_unit_interval(forcing[t])
        pressure[t, root] = forcing[t]
        erosion_proxy[t, root] = params.rho_e * forcing[t]
        theta[t + 1, root] = projection_unit_interval(forcing[t + 1])
        pressure[t + 1, root] = forcing[t + 1]
        erosion_proxy[t + 1, root] = params.rho_e * forcing[t + 1]

        for level in graph.levels[1:]:
            for v in level:
                p1 = graph.primary_parent[v]
                p2 = graph.secondary_parent[v]
                w1, w2 = normalised_parent_weights(graph, properties, v)
                assert p1 is not None

                # Transient pressure and erosion-propensity propagation.
                pressure_in = alpha[v, 1] * w1 * pressure[t, p1]
                erosion_in = alpha[v, 2] * w1 * erosion_proxy[t, p1]
                if p2 is not None:
                    pressure_in += beta[v, 1] * w2 * pressure[t, p2]
                    erosion_in += beta[v, 2] * w2 * erosion_proxy[t, p2]
                pressure[t + 1, v] = pressure_in
                erosion_proxy[t + 1, v] = erosion_in

                # Interface-aware local reservoir wetting update.
                g1, chi1 = interface_transmission(properties.kappa[p1], properties.kappa[v], params)
                wetting_input = w1 * g1 * chi1 * theta[t, p1]
                g_bar = w1 * g1
                if p2 is not None:
                    g2, chi2 = interface_transmission(properties.kappa[p2], properties.kappa[v], params)
                    wetting_input += w2 * g2 * chi2 * theta[t, p2]
                    g_bar += w2 * g2
                fill_time = params.fill_min_h + params.fill_scale_h * (1.0 - g_bar)
                drainage_fraction = params.dt_h / params.drain_h
                theta[t + 1, v] = projection_unit_interval((1.0 - drainage_fraction) * theta[t, v] + params.dt_h / fill_time * wetting_input)

    # Ensure final root values match forcing.
    theta[-1, root] = projection_unit_interval(forcing[-1])
    pressure[-1, root] = forcing[-1]
    erosion_proxy[-1, root] = params.rho_e * forcing[-1]
    return {"theta": theta, "pressure": pressure, "erosion_proxy": erosion_proxy}


def level_mean(graph: Graph, values: np.ndarray) -> np.ndarray:
    """Aggregate node values by depth level using the level mean."""
    return np.asarray([values[:, level].mean(axis=1) for level in graph.levels]).T


def wetting_front_depth(graph: Graph, theta: np.ndarray, q_theta: float = 0.03, cumulative: bool = True) -> np.ndarray:
    """Compute the deepest wet level as a time series."""
    threshold = q_theta * max(float(theta.max()), 1e-12)
    front = []
    for t in range(theta.shape[0]):
        deepest = 0
        for k, level in enumerate(graph.levels):
            if np.any(theta[t, level] > threshold):
                deepest = k
        front.append(deepest)
    front_arr = np.asarray(front, dtype=float)
    return np.maximum.accumulate(front_arr) if cumulative else front_arr


def activated_nodes(pressure: np.ndarray, tau_c: np.ndarray) -> np.ndarray:
    """Number of nodes whose pressure exceeds the local threshold at each time."""
    return (pressure > tau_c[None, :]).sum(axis=1)


def exceedance_ratio(pressure: np.ndarray, tau_c: np.ndarray) -> np.ndarray:
    """Pressure-to-threshold exceedance ratio C(v)=max_t p_v(t)/tau_c,v."""
    return pressure.max(axis=0) / tau_c


def attenuation_profile(graph: Graph, pressure: np.ndarray) -> np.ndarray:
    """Depth-level attenuation from level-wise mean peak pressure, normalised to the root."""
    level_peak = []
    for level in graph.levels:
        level_peak.append(float(pressure[:, level].mean(axis=1).max()))
    level_peak = np.asarray(level_peak)
    return level_peak / max(level_peak[0], 1e-12)
