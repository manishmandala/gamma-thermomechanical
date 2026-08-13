# Toolpath/birth-consistency validation for a synthetic-toolpath dataset,
# run BEFORE any thermal simulation. Complements diagnose_composition_mapping.py
# (which covers composition-field-specific checks: s-range, projection
# distance, seam diagnostics) - this script covers the toolpath/mesh/birth
# layer underneath the composition field: bounds, speed, path length/ENDTIM,
# coverage, never-reached count, birth-time monotonicity, spatial alignment.

import argparse

import numpy as np
from gamma.simulator.gamma import domain_mgr, load_toolpath
from gamma.simulator.preprocessor import load_mesh_file

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--dataset', required=True)
parser.add_argument('--kfile', required=True)
parser.add_argument('--toolpath', required=True)
parser.add_argument('--base-k', required=True, help='original unmodified mesh source .k, for a mesh-identity cross-check')
parser.add_argument('--expected-speed', type=float, required=True)
parser.add_argument('--out', required=True)
args = parser.parse_args()

lines = []


def report(line):
    print(line)
    lines.append(line)


import os
# sort_birth=False: keep domain_mgr's internal element order matching the
# file's own order, since this script cross-references d.element_birth
# against centroids computed independently via load_mesh_file's file-order
# arrays - domain_mgr's default sort_birth=True would silently reorder
# elements by birth time, making any index-based comparison meaningless.
d = domain_mgr(filename=os.path.join(args.dataset, args.kfile),
                toolpathdir=os.path.join(args.dataset, args.toolpath),
                input_data_dir=args.dataset, verbose=False, sort_birth=False)
toolpath = load_toolpath(os.path.join(args.dataset, args.toolpath))
nodes, elements = load_mesh_file(os.path.join(args.dataset, args.kfile))
base_nodes, base_elements = load_mesh_file(args.base_k)

report('=== Synthetic-toolpath validation: {} ==='.format(args.kfile))
report('')

# ---- mesh identity vs. the original, unmodified base .k ----
mesh_identical = np.array_equal(nodes, base_nodes) and np.array_equal(elements, base_elements)
report('1. Mesh identity vs. base .k (must be UNCHANGED): {}'.format('PASS' if mesh_identical else 'FAIL'))

# ---- toolpath bounds vs mesh bounds ----
tp_xyz = toolpath[:, 1:4]
mesh_bounds = (nodes.min(axis=0), nodes.max(axis=0))
tp_bounds = (tp_xyz.min(axis=0), tp_xyz.max(axis=0))
report('2. Toolpath bounds: x[{:.3f},{:.3f}] y[{:.3f},{:.3f}] z[{:.3f},{:.3f}]'.format(
    tp_bounds[0][0], tp_bounds[1][0], tp_bounds[0][1], tp_bounds[1][1], tp_bounds[0][2], tp_bounds[1][2]))
report('   Mesh bounds:     x[{:.3f},{:.3f}] y[{:.3f},{:.3f}] z[{:.3f},{:.3f}]'.format(
    mesh_bounds[0][0], mesh_bounds[1][0], mesh_bounds[0][1], mesh_bounds[1][1], mesh_bounds[0][2], mesh_bounds[1][2]))
bounds_ok = (tp_bounds[1][2] <= mesh_bounds[1][2] + 1e-6)
report('   toolpath z-extent within mesh z-extent: {}'.format('PASS' if bounds_ok else 'FAIL'))

# ---- measured scan speed ----
dt = np.diff(toolpath[:, 0])
dd = np.linalg.norm(np.diff(tp_xyz, axis=0), axis=1)
valid = dt > 1e-9
speeds = dd[valid] / dt[valid]
report('')
report('3. Measured scan speed: mean={:.6f} median={:.6f} min={:.6f} max={:.6f} (expected {:.4f})'.format(
    speeds.mean(), np.median(speeds), speeds.min(), speeds.max(), args.expected_speed))
speed_ok = np.allclose(speeds, args.expected_speed, atol=1e-4)
report('   uniform speed across ALL segments (on and off): {}'.format('PASS' if speed_ok else 'FAIL'))

# ---- total path length / ENDTIM ----
total_length = dd.sum()
report('')
report('4. Total path length: {:.4f}   ENDTIM: {:.4f}   toolpath duration: {:.4f}'.format(
    total_length, d.end_sim_time if hasattr(d, 'end_sim_time') else float('nan'), toolpath[-1, 0]))

# ---- build-element coverage / never-reached ----
mat_ids = [m[0] for m in d.mat_thermal]
substrate_pid = [m for m in mat_ids if bool((d.element_birth[d.element_mat == m] == 0).all())][0]
build_pid = [m for m in mat_ids if m != substrate_pid][0]
build_mask = d.element_mat == build_pid
n_build = int(build_mask.sum())
never_reached = int(((d.element_birth >= 1e5) & build_mask).sum())
report('')
report('5. Build elements: {}   never-reached (birth>=1e5 sentinel): {} ({:.3f}%)'.format(
    n_build, never_reached, 100 * never_reached / n_build))

# ---- birth-time range and monotonic active-element growth ----
real_birth = d.element_birth[build_mask & (d.element_birth < 1e5)]
report('')
report('6. Birth-time range (real, non-sentinel build elements): [{:.4f}, {:.4f}]'.format(
    real_birth.min(), real_birth.max()))
sample_times = np.linspace(0, toolpath[-1, 0], 50)
active_counts = [int((d.element_birth[build_mask] <= t).sum()) for t in sample_times]
monotonic = all(b >= a for a, b in zip(active_counts, active_counts[1:]))
report('   active-element count at 50 sampled times is monotonically non-decreasing: {}'.format(
    'PASS' if monotonic else 'FAIL'))
report('   active count: t=0 -> {}, t=ENDTIM -> {}'.format(active_counts[0], active_counts[-1]))

# ---- spatial alignment: for each build element, is its centroid within
# beam-radius-ish distance of SOME toolpath point at/after its own birth time? ----
ele_nodes = nodes[elements]
centroids = ele_nodes.mean(axis=1)
build_idx = np.where(build_mask)[0]
sample = np.random.RandomState(0).choice(build_idx, size=min(200, len(build_idx)), replace=False)
misaligned = 0
for idx in sample:
    bt = d.element_birth[idx]
    if bt >= 1e5:
        continue
    # interpolate the true laser position AT the birth time (not the nearest
    # raw waypoint) - deposition tracks here are long single segments, so
    # snapping to a waypoint can be tens of mm from the true interpolated
    # position if birth falls mid-track
    laser_x = np.interp(bt, toolpath[:, 0], toolpath[:, 1])
    laser_y = np.interp(bt, toolpath[:, 0], toolpath[:, 2])
    dist = np.linalg.norm([laser_x - centroids[idx, 0], laser_y - centroids[idx, 1]])
    if dist > 2.0:  # generous multiple of beam radius (1.12)
        misaligned += 1
report('')
report('7. Spatial alignment sample ({} elements): {} exceed 2.0-unit distance from toolpath at their birth time '
       '({:.1f}%)'.format(len(sample), misaligned, 100 * misaligned / len(sample)))

report_path = args.out
with open(report_path, 'w') as f:
    f.write('\n'.join(lines) + '\n')
print('\nwrote {}'.format(report_path))
