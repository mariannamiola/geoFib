"""Run a self-contained synthetic Fibonacci ROM demonstration.

Usage:
    python run_synthetic_demo.py --out outputs_demo

The script generates CSV diagnostics and a small set of figures for a Gaussian
storm applied to a seven-level Fibonacci graph. No external solver is used.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt

from fibonacci_rom_core import (
    ROMParameters,
    activated_nodes,
    attenuation_profile,
    build_fibonacci_graph,
    exceedance_ratio,
    gaussian_forcing,
    level_mean,
    make_synthetic_layered_properties,
    run_rom,
    wetting_front_depth,
)


def save_csv(path: Path, header: str, data: np.ndarray) -> None:
    np.savetxt(path, data, delimiter=",", header=header, comments="")


def main() -> None:
    parser = argparse.ArgumentParser(description="Synthetic interface-aware reservoir Fibonacci ROM demo")
    parser.add_argument("--out", default="outputs_demo", help="Output directory")
    parser.add_argument("--levels", type=int, default=7, help="Number of Fibonacci graph levels")
    parser.add_argument("--steps", type=int, default=49, help="Number of hourly output steps")
    parser.add_argument("--seed", type=int, default=7, help="Random seed for synthetic properties")
    args = parser.parse_args()

    out = Path(args.out)
    fig_dir = out / "figures"
    out.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    params = ROMParameters()
    graph = build_fibonacci_graph(args.levels)
    props = make_synthetic_layered_properties(graph, seed=args.seed)
    forcing = gaussian_forcing(n_steps=args.steps, peak=1.0, center_h=16.0, sigma_h=4.0, dt_h=params.dt_h)
    result = run_rom(graph, props, forcing, params)

    theta = result["theta"]
    pressure = result["pressure"]
    time_h = np.arange(args.steps) * params.dt_h
    front = wetting_front_depth(graph, theta, q_theta=params.q_theta, cumulative=True)
    active = activated_nodes(pressure, props.tau_c)
    C = exceedance_ratio(pressure, props.tau_c)
    atten = attenuation_profile(graph, pressure)
    theta_level = level_mean(graph, theta)
    pressure_level = level_mean(graph, pressure)

    # CSV outputs.
    save_csv(out / "forcing.csv", "time_h,forcing", np.column_stack([time_h, forcing]))
    save_csv(out / "wetting_front.csv", "time_h,wetting_front_level", np.column_stack([time_h, front]))
    save_csv(out / "activated_nodes.csv", "time_h,activated_nodes", np.column_stack([time_h, active]))
    save_csv(out / "level_mean_theta.csv", "time_h," + ",".join([f"level_{k}" for k in range(args.levels)]), np.column_stack([time_h, theta_level]))
    save_csv(out / "level_mean_pressure.csv", "time_h," + ",".join([f"level_{k}" for k in range(args.levels)]), np.column_stack([time_h, pressure_level]))
    save_csv(out / "attenuation_profile.csv", "level,normalised_peak_pressure", np.column_stack([np.arange(args.levels), atten]))
    save_csv(out / "exceedance_ratio_by_node.csv", "node,level,C", np.column_stack([[n.index for n in graph.nodes], [n.level for n in graph.nodes], C]))

    metrics = {
        "n_levels": args.levels,
        "n_nodes": len(graph.nodes),
        "peak_wetting_front_level": float(front.max()),
        "peak_activated_nodes": int(active.max()),
        "time_of_peak_activation_h": float(time_h[int(np.argmax(active))]),
        "max_exceedance_ratio": float(C.max()),
        "mean_exceedance_ratio": float(C.mean()),
        "level_6_attenuation": float(atten[-1]),
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    # Figures.
    plt.figure(figsize=(6, 3.5))
    plt.plot(time_h, forcing)
    plt.xlabel("Time [h]")
    plt.ylabel("Root forcing [-]")
    plt.tight_layout()
    plt.savefig(fig_dir / "forcing.png", dpi=300)
    plt.close()

    plt.figure(figsize=(6, 3.5))
    plt.step(time_h, front, where="post")
    plt.gca().invert_yaxis()
    plt.xlabel("Time [h]")
    plt.ylabel("Wetting-front level")
    plt.tight_layout()
    plt.savefig(fig_dir / "wetting_front.png", dpi=300)
    plt.close()

    plt.figure(figsize=(6, 3.5))
    plt.plot(time_h, active)
    plt.xlabel("Time [h]")
    plt.ylabel("Activated nodes")
    plt.tight_layout()
    plt.savefig(fig_dir / "activated_nodes.png", dpi=300)
    plt.close()

    plt.figure(figsize=(6, 3.5))
    plt.plot(np.arange(args.levels), atten, marker="o")
    plt.xlabel("Depth level")
    plt.ylabel("Normalised peak pressure")
    plt.tight_layout()
    plt.savefig(fig_dir / "attenuation_profile.png", dpi=300)
    plt.close()

    plt.figure(figsize=(7, 4))
    scatter_x = [node.local_index for node in graph.nodes]
    scatter_y = [node.level for node in graph.nodes]
    plt.scatter(scatter_x, scatter_y, c=C, s=70)
    plt.gca().invert_yaxis()
    plt.xlabel("Local node index within level")
    plt.ylabel("Depth level")
    cbar = plt.colorbar()
    cbar.set_label("Exceedance ratio C(v)")
    plt.tight_layout()
    plt.savefig(fig_dir / "exceedance_map.png", dpi=300)
    plt.close()

    print(f"Demo complete. Outputs written to: {out}")
    for key, value in metrics.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
