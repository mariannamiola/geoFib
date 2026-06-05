# Fibonacci ROM synthetic demo

This repository contains a Python implementation of the
interface-aware reservoir Fibonacci reduced-order model (ROM) described in the
manuscript:

**A Fibonacci-Like Propagation Graph for Infiltration and Internal Erosion in Heterogeneous Layered Soils**

The code demonstrates the proposed ROM on a single synthetic Gaussian storm scenario.

## Contents

- `fibonacci_rom_core.py`  
  Core implementation: Fibonacci graph construction, node properties,
  interface-aware reservoir wetting update, pressure propagation, exceedance
  ratio, wetting-front depth, and attenuation diagnostics.

- `run_synthetic_demo.py`  
  End-to-end demo script. It runs a seven-level graph under Gaussian forcing and
  writes CSV files, summary metrics, and figures.

## Requirements

Python 3.10 or newer is recommended. The only third-party dependencies are:

```bash
pip install numpy matplotlib
```

## Quick start

```bash
python run_synthetic_demo.py --out outputs_demo
```

The script creates:

```text
outputs_demo/
  metrics.json
  forcing.csv
  wetting_front.csv
  activated_nodes.csv
  level_mean_theta.csv
  level_mean_pressure.csv
  attenuation_profile.csv
  exceedance_ratio_by_node.csv
  figures/
    forcing.png
    wetting_front.png
    activated_nodes.png
    attenuation_profile.png
    exceedance_map.png
```

## What the demo does

1. Builds a directed acyclic Fibonacci graph with seven levels and 33 nodes.
2. Assigns a synthetic layered sand-to-clay hydraulic profile.
3. Applies a Gaussian surface forcing to the root node.
4. Propagates a transient pressure-like signal and an erosion-propensity proxy.
5. Updates the wetting component through a local reservoir law with an
   interface-aware hydraulic-contrast correction.
6. Reports wetting-front depth, activated-node count, exceedance ratio, and
   depth-level attenuation.

## Notes

The parameters in this repository are provided for an illustrative synthetic
case. They are not calibrated to a specific field site. The purpose of the code
is to expose the structure of the proposed ROM in a transparent and compact
form, suitable for inspection, reproduction, and extension.

## Suggested citation

If you use this code, please cite the accompanying manuscript once available.
