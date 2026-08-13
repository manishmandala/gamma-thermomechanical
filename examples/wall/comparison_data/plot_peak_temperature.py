# Plots peak temperature at the wall's central node across the pure-TI64,
# pure-IN718, and graded runs.

import glob
import os

import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt

# Central wall node: geometric center of the deposited wall (x=0, mid-length;
# y=-0.1, mid-thickness; z=2, mid-build-height). All three runs share the same
# mesh/toolpath/timing, so node-birth ordering (and hence this point index) is
# identical across runs - confirmed by checking coords at this index match
# (0, -0.1, 2) in every run's final frame.
CENTRAL_IDX = 14302

RUNS = [
    ("TI64", "../results/pure_TI64", "#2a78d6"),
    ("IN718", "../results/pure_IN718", "#eb6834"),
    ("Graded", "../results/graded", "#1baf7a"),
]

peak_node_temp = {}
central_node_temp = {}

for label, folder, _ in RUNS:
    frames = sorted(glob.glob(os.path.join(folder, "wall_*.vtu")))
    global_peak = -np.inf
    central_peak = -np.inf
    for f in frames:
        mesh = pv.read(f)
        temp = mesh.point_data["temp"]
        global_peak = max(global_peak, float(temp.max()))
        if temp.shape[0] > CENTRAL_IDX:
            central_peak = max(central_peak, float(temp[CENTRAL_IDX]))
    peak_node_temp[label] = global_peak
    central_node_temp[label] = central_peak
    print(f"{label}: peak-node={global_peak:.1f} K, central-node={central_peak:.1f} K")

labels = [r[0] for r in RUNS]
colors = [r[2] for r in RUNS]

fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))

for ax, data, title in (
    (axes[0], peak_node_temp, "Peak Temperature @ Peak Node"),
    (axes[1], central_node_temp, "Peak Temperature @ Central Node"),
):
    values = [data[l] for l in labels]
    bars = ax.bar(labels, values, color=colors, width=0.55)
    for bar, v in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            v,
            f"{v:,.0f}",
            ha="center",
            va="bottom",
            fontsize=10,
            color="#0b0b0b",
        )
    ax.set_title(title)
    ax.set_xlabel("Material")
    ax.set_ylabel("Temperature (K)")
    ax.set_ylim(0, max(values) * 1.15)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#dddddd", linewidth=1, zorder=0)
    ax.set_axisbelow(True)

fig.tight_layout()
out_path = "peak_temperature_comparison.png"
fig.savefig(out_path, dpi=200)
print(f"Saved {out_path}")
