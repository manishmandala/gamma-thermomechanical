import glob

import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt

# Validates that importing Rowan's mesh (sol_100.vtu) + her own per-element
# composition actually drives real, per-location thermal behavior - not a
# silent default to one material. There's no laser here (toolpath_off.crs),
# so this is pure relaxation of a fixed initial T field toward ambient/
# boundary conditions - peak temperature is fixed by the identical initial
# condition across all 3 runs and isn't a useful metric (it can only
# decrease). Instead: track temperature over time at 4 fixed probe nodes
# (valid across all 3 runs since sort_birth=False preserves sol_100.vtu's
# own node order), and a domain-wide relaxation curve.
#
# Probe nodes (computed once from sol_100.vtu itself - see the values below):
#   node 6    - hottest initial point (T=800K)
#   node 0    - coldest initial point (T=200K)
#   node 1606 - inside the element with frac_B nearest 1 (SS316-proxy-dominant, frac_B=0.993)
#   node 99   - inside the element with frac_B nearest 0 (IN625-proxy-dominant, frac_B=0.259)

PROBES = {
    'Hottest (node 6, T0=800K)': 6,
    'Coldest (node 0, T0=200K)': 0,
    'SS316-proxy region (node 1606)': 1606,
    'IN625-proxy region (node 99)': 99,
}

RUNS = [
    ('IN718 (IN625 proxy)', '../results_rowan_IN718', '#2a78d6'),
    ('1018 (SS316 proxy)', '../results_rowan_1018', '#eb6834'),
    ('Graded (Rowan\'s composition)', '../results_rowan_graded', '#1baf7a'),
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
fig.suptitle("Probe-node temperature over time: IN718 vs 1018 vs Rowan's graded mesh", fontsize=12)
fig.tight_layout()
fig.savefig('rowan_mesh_probe_nodes.png', dpi=180)
print('Saved rowan_mesh_probe_nodes.png')

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
fig2.savefig('rowan_mesh_relaxation.png', dpi=180)
print('Saved rowan_mesh_relaxation.png')

# --- printed summary: does graded sit between the two pure endpoints at every probe? ---
print()
for probe in PROBES:
    vals_at_end = {label: data[label][probe][-1][1] for label, _, _ in RUNS}
    lo = min(vals_at_end['IN718 (IN625 proxy)'], vals_at_end['1018 (SS316 proxy)'])
    hi = max(vals_at_end['IN718 (IN625 proxy)'], vals_at_end['1018 (SS316 proxy)'])
    graded = vals_at_end["Graded (Rowan's composition)"]
    between = lo <= graded <= hi
    print(f'{probe}: final T - IN718={vals_at_end["IN718 (IN625 proxy)"]:.1f}K, '
          f'1018={vals_at_end["1018 (SS316 proxy)"]:.1f}K, graded={graded:.1f}K, '
          f'graded between endpoints: {between}')
