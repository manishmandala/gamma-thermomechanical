# Plots peak temperature across the pure-TI64, pure-Cu, and graded runs -
# the high-thermal-contrast counterpart to plot_peak_temperature.py's
# TI64/IN718 comparison.

import glob
import os

import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt

# TI64-vs-Cu-vs-graded comparison: Cu is a much more drastic thermal contrast
# than IN718 (~60x higher conductivity, ~550K lower melting point), chosen to
# make the composition gradient's effect on temperature obvious - the original
# TI64-vs-IN718 pair (both poor conductors, similar melting points) barely
# differed. Cu's much smaller stable timestep made the original 0.5 build-depth
# too slow to run in reasonable time, so this comparison uses a shallower
# STOP_FRACTION=0.15 preview instead (still deep enough to show the gradient's
# effect). The existing results/pure_TI64/ run (from the TI64-vs-IN718 comparison)
# went to STOP_FRACTION=0.5, so it's filtered here to only the frames within
# the same sim-time window as the new Cu/graded runs, for a fair comparison.

MAX_SIM_TIME = 5.87  # matches results/pure_Cu / results/graded_cu's STOP_FRACTION=0.15 depth

# Central wall node at this shallower depth: geometric center of the deposited
# region reached so far (x=0, mid-length; y=-0.1, mid-thickness; z=0.4,
# mid-height of the ~0.8mm built at this depth - NOT z=2, which isn't
# deposited yet at STOP_FRACTION=0.15). All three runs share the same
# mesh/toolpath, so node-birth ordering (and hence this point index) is
# identical across runs - confirmed by checking coords at this index match
# (0, -0.1, 0.4) in every run's frame at matching sim time.
CENTRAL_IDX = 7068

RUNS = [
    ("TI64", "../results/pure_TI64", "#2a78d6"),
    ("Cu", "../results/pure_Cu", "#eb6834"),
    ("Graded", "../results/graded_cu", "#1baf7a"),
]

peak_node_temp = {}
central_node_temp = {}

for label, folder, _ in RUNS:
    frames = sorted(glob.glob(os.path.join(folder, "wall_*.vtu")))
    global_peak = -np.inf
    central_peak = -np.inf
    for f in frames:
        mesh = pv.read(f)
        sim_time = float(mesh.field_data["TIME"][0])
        if sim_time > MAX_SIM_TIME:
            continue
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
out_path = "peak_temperature_comparison_cu.png"
fig.savefig(out_path, dpi=200)
print(f"Saved {out_path}")
