# Plots the true 4-material external-mesh comparison (all 4 real composition
# columns) alongside the original 2-material collapsed version and the pure
# IN718/1018 controls.

import glob

import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt

# Extends plot_external_mesh_comparison.py with the true 4-material graded
# run (results/external_mesh_graded_4mat, using all 4 of the externally-
# imported mesh's real composition columns via external_mesh_composition_4mat.py)
# alongside the original 2-material collapsed graded run
# (results/external_mesh_graded) and the two majority-material
# pure controls (IN718, 1018). HA25/MCrAlY (proxied by CPTi/Al) are always
# minor admixtures in this mesh (IN625+SS316 >= 65% everywhere), so IN718/1018
# still bound the result even though their pure controls aren't included here.
#
# Same setup as the original comparison: no laser (toolpath_off.crs), pure
# relaxation of a fixed initial T field, sort_birth=False so node indices
# mean the same physical location across every run.

PROBES = {
    'Hottest (node 6, T0=800K)': 6,
    'Coldest (node 0, T0=200K)': 0,
    'SS316-proxy region (node 1606)': 1606,
    'IN625-proxy region (node 99)': 99,
}

RUNS = [
    ('IN718 (IN625 proxy)', '../results/external_mesh_IN718', '#2a78d6'),
    ('1018 (SS316 proxy)', '../results/external_mesh_1018', '#eb6834'),
    ('Graded, 2-material (collapsed)', '../results/external_mesh_graded', '#1baf7a'),
    # was '../results_rowan_graded_4mat' - a stale path that never existed
    # (the actual run only ever had the post-dtfix name); fixed here as part
    # of the results/ consolidation rather than left dangling.
    ('Graded, 4-material (true composition)', '../results/external_mesh_graded_4mat', '#a83fd1'),
]

data = {label: {probe: [] for probe in PROBES} for label, _, _ in RUNS}
relax = {label: [] for label, _, _ in RUNS}

for label, folder, _ in RUNS:
    frames = sorted(glob.glob(folder + '/wall_*.vtu'))
    for f in frames:
        mesh = pv.read(f)
        t = float(mesh.field_data['TIME'][0])
        temp = mesh.point_data['temp']
        for probe, idx in PROBES.items():
            data[label][probe].append((t, float(temp[idx])))
        relax[label].append((t, float(temp.mean()), float(temp.std())))

# --- probe time-series ---
fig, axes = plt.subplots(2, 2, figsize=(11, 8))
for ax, probe in zip(axes.flat, PROBES):
    for label, _, color in RUNS:
        pts = data[label][probe]
        times, temps = zip(*pts)
        ax.plot(times, temps, label=label, color=color, linewidth=2)
    ax.set_title(probe)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Temperature (K)')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(color='#dddddd', linewidth=1, zorder=0)
    ax.set_axisbelow(True)
axes.flat[0].legend(fontsize=8, loc='upper right')
fig.suptitle("Probe-node temperature over time: IN718 vs 1018 vs 2-material vs 4-material graded", fontsize=12)
fig.tight_layout()
fig.savefig('external_mesh_probe_nodes_4mat.png', dpi=180)
print('Saved external_mesh_probe_nodes_4mat.png')

# --- domain-wide relaxation ---
fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
for label, _, color in RUNS:
    pts = relax[label]
    times, means, stds = zip(*pts)
    ax1.plot(times, means, label=label, color=color, linewidth=2)
    ax2.plot(times, stds, label=label, color=color, linewidth=2)
ax1.set_title('Domain mean temperature vs time')
ax2.set_title('Domain temperature std-dev vs time')
for ax in (ax1, ax2):
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Temperature (K)')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(color='#dddddd', linewidth=1, zorder=0)
    ax.set_axisbelow(True)
ax1.legend(fontsize=8)
fig2.tight_layout()
fig2.savefig('external_mesh_relaxation_4mat.png', dpi=180)
print('Saved external_mesh_relaxation_4mat.png')

# --- printed summary: does each graded variant sit between the two majority endpoints? ---
print()
for probe in PROBES:
    vals_at_end = {label: data[label][probe][-1][1] for label, _, _ in RUNS}
    lo = min(vals_at_end['IN718 (IN625 proxy)'], vals_at_end['1018 (SS316 proxy)'])
    hi = max(vals_at_end['IN718 (IN625 proxy)'], vals_at_end['1018 (SS316 proxy)'])
    g2 = vals_at_end['Graded, 2-material (collapsed)']
    g4 = vals_at_end['Graded, 4-material (true composition)']
    print(f'{probe}: final T - IN718={vals_at_end["IN718 (IN625 proxy)"]:.1f}K, '
          f'1018={vals_at_end["1018 (SS316 proxy)"]:.1f}K, '
          f'2-mat graded={g2:.1f}K (between: {lo <= g2 <= hi}), '
          f'4-mat graded={g4:.1f}K (between: {lo <= g4 <= hi}), '
          f'delta 4mat-2mat={g4-g2:+.2f}K')
