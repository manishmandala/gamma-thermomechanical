import glob

import pyvista as pv
import matplotlib.pyplot as plt

# Shows the finding from the early-frame render comparison: the TI64-vs-Cu-vs-
# graded temperature gap, at each material's own true peak (whole-mesh max,
# wherever/whenever it actually occurs - not the same time across materials),
# largely washes out by the run's final frame (t~5.87s) - copper's high
# conductivity re-equalizes the field within a few seconds of the peak.
FINAL_TIME = 5.87

RUNS = [
    ("TI64", "../results_TI64", "#2a78d6"),
    ("Cu", "../results_Cu", "#eb6834"),
    ("Graded", "../results_graded_cu", "#1baf7a"),
]

peak_temp = {}
final_temp = {}

for label, folder, _ in RUNS:
    frames = sorted(glob.glob(folder + "/wall_*.vtu"))
    peak_t, peak_v = None, -1
    final = None
    for f in frames:
        mesh = pv.read(f)
        t = float(mesh.field_data["TIME"][0])
        if t > FINAL_TIME + 0.01:
            continue
        maxt = float(mesh.point_data["temp"].max())
        if maxt > peak_v:
            peak_v = maxt
            peak_t = t
        final = maxt
    peak_temp[label] = peak_v
    final_temp[label] = final
    print(f"{label}: true peak={peak_v:.1f}K at t={peak_t:.3f}s, final(t~{FINAL_TIME}s)={final:.1f}K")

labels = [r[0] for r in RUNS]
colors = [r[2] for r in RUNS]

fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))

for ax, data, title in (
    (axes[0], peak_temp, "True Peak Temperature (whole sim, any time)"),
    (axes[1], final_temp, f"Peak Temperature @ t={FINAL_TIME}s (final frame)"),
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
out_path = "transient_vs_final_temperature.png"
fig.savefig(out_path, dpi=200)
print(f"Saved {out_path}")
