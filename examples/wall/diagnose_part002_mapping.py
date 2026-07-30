# Diagnostic + validation report for the toolpath_arc_length composition
# mapping on part002 - the recommended curved-path test dataset (see
# project memory: genuinely curving toolpath, has a reference VTK, only a
# minor known toolpath-coverage gap unrelated to this mapping).
#
# Does NOT run the GAMMA thermal solver - this only exercises the
# preprocessing pipeline (compute_centroid -> coordinate_function ->
# composition_function) and exports what it computed, for numeric and
# visual (ParaView) inspection before spending any GPU time on an actual
# simulation.
#
# Outputs, written into the dataset's toolpath_arc_length_composition/ folder:
#   part002_composition_diagnostics.csv - eid, cx, cy, cz, s, composition, projection_distance
#   part002_composition_diagnostics.vtu - same fields as cell_data, viewable in ParaView
# Plus a printed statistics report (also saved as a .txt).

import os

import numpy as np
import pyvista as pv
import vtk
import cupy as cp
cp.cuda.Device(0).use()
from gamma.simulator.gamma import domain_mgr, load_toolpath
from composition_lib import (compute_centroid, compute_bounds, coordinate_function, composition_function,
                              build_toolpath_arc_length_table, project_to_toolpath)

DATASET = '../incoming_dataset/part002_LP800_SSp10_H2.24_SSt3_LH0.9'
KFILE = '2.k'
TOOLPATH = 'toolpath.crs'
OUTDIR = os.path.join(DATASET, 'toolpath_arc_length_composition')
os.makedirs(OUTDIR, exist_ok=True)

# ---- load source geometry + auto-detect build region (same pattern as prepare_composition.py) ----
d = domain_mgr(filename=os.path.join(DATASET, KFILE), toolpathdir=os.path.join(DATASET, TOOLPATH),
                input_data_dir=DATASET, verbose=False)
mat_ids = [m[0] for m in d.mat_thermal]
substrate_pid = [m for m in mat_ids if bool((d.element_birth[d.element_mat == m] == 0).all())][0]
build_pid = [m for m in mat_ids if m != substrate_pid][0]

with open(os.path.join(DATASET, KFILE)) as f:
    lines = f.readlines()
node_start = next(i for i, l in enumerate(lines) if l.startswith('*NODE'))
node_end = next(i for i in range(node_start + 1, len(lines)) if lines[i].startswith('*'))
nodes = {}
for l in lines[node_start + 1:node_end]:
    if l.startswith('$'):
        continue
    text = l.split()
    nodes[int(text[0])] = (float(text[1]), float(text[2]), float(text[3]))

elem_start = next(i for i, l in enumerate(lines) if l.startswith('*ELEMENT_SOLID'))
elem_end = next(i for i in range(elem_start + 1, len(lines)) if lines[i].startswith('*'))
build_eids, build_node_ids = [], {}
for l in lines[elem_start + 1:elem_end]:
    if l.startswith('$'):
        continue
    text = l.split()
    eid, pid = int(text[0]), int(text[1])
    if pid == build_pid:
        build_eids.append(eid)
        build_node_ids[eid] = [int(t) for t in text[2:10]]

print('build region: matID {} ({} elements)'.format(build_pid, len(build_eids)))

centroids = np.array([compute_centroid(nodes, build_node_ids[eid]) for eid in build_eids])

# ---- toolpath-arc-length mapping ----
toolpath_xyz = load_toolpath(os.path.join(DATASET, TOOLPATH))[:, 1:4]
arc_table = build_toolpath_arc_length_table(toolpath_xyz)
total_length = arc_table[-1]
arc_at, proj_dist, best_seg = project_to_toolpath(centroids, toolpath_xyz, arc_table, return_segment=True)
s = coordinate_function(centroids, 'toolpath_arc_length', toolpath=toolpath_xyz)
assert np.allclose(s, np.clip(arc_at / total_length, 0, 1)), 's does not match arc_at/total_length - inconsistent'

composition = np.array([composition_function(float(si), 0.0, 'sinusoidal') for si in s])

# ---- also compute global_x for the comparison stage ----
bounds = compute_bounds(centroids)
s_global_x = coordinate_function(centroids, 'global_x', bounds=bounds)
composition_global_x = np.array([composition_function(float(si), 0.0, 'sinusoidal') for si in s_global_x])

# ---- CSV export ----
csv_path = os.path.join(OUTDIR, 'part002_composition_diagnostics.csv')
with open(csv_path, 'w') as f:
    f.write('element_id,centroid_x,centroid_y,centroid_z,s_arc_length,composition_arc_length,'
            'projection_distance,best_segment,s_global_x,composition_global_x\n')
    for i, eid in enumerate(build_eids):
        f.write('{},{:.6f},{:.6f},{:.6f},{:.8f},{:.8f},{:.6f},{},{:.8f},{:.8f}\n'.format(
            eid, centroids[i, 0], centroids[i, 1], centroids[i, 2],
            s[i], composition[i], proj_dist[i], best_seg[i], s_global_x[i], composition_global_x[i]))
print('wrote {}'.format(csv_path))

# ---- VTU export (real HEX8 geometry, cell_data - viewable in ParaView) ----
active_cells = []
for eid in build_eids:
    active_cells += [8] + build_node_ids[eid]
active_cells = np.array(active_cells)
cell_types = np.array([vtk.VTK_HEXAHEDRON] * len(build_eids))

all_node_ids = sorted(nodes)
node_id_to_local = {nid: i for i, nid in enumerate(all_node_ids)}
points = np.array([nodes[nid] for nid in all_node_ids])
local_cells = []
for eid in build_eids:
    local_cells += [8] + [node_id_to_local[n] for n in build_node_ids[eid]]
local_cells = np.array(local_cells)

grid = pv.UnstructuredGrid(local_cells, cell_types, points)
grid.cell_data['s_arc_length'] = s
grid.cell_data['composition_arc_length'] = composition
grid.cell_data['projection_distance'] = proj_dist
grid.cell_data['best_segment'] = best_seg.astype(float)
grid.cell_data['s_global_x'] = s_global_x
grid.cell_data['composition_global_x'] = composition_global_x
grid.cell_data['composition_difference'] = np.abs(composition - composition_global_x)
vtu_path = os.path.join(OUTDIR, 'part002_composition_diagnostics.vtu')
grid.save(vtu_path)
print('wrote {}'.format(vtu_path))

# ---- statistics ----
report_lines = []


def report(line):
    print(line)
    report_lines.append(line)


report('\npart002 toolpath_arc_length mapping - validation statistics')
report('=' * 60)
report('build elements: {}'.format(len(build_eids)))
report('toolpath: {} waypoints, {} segments, total length {:.4f}'.format(
    len(toolpath_xyz), len(toolpath_xyz) - 1, total_length))
report('')
report('s (normalized path coordinate):')
report('  min={:.6f}  max={:.6f}'.format(s.min(), s.max()))
near0 = (s < 0.05).sum()
near1 = (s > 0.95).sum()
report('  elements with s < 0.05 (near start): {} ({:.2f}%)'.format(near0, 100 * near0 / len(s)))
report('  elements with s > 0.95 (near end):   {} ({:.2f}%)'.format(near1, 100 * near1 / len(s)))
report('')
report('nearest-projection distance (element centroid to toolpath):')
report('  mean={:.4f}  median={:.4f}  max={:.4f}'.format(
    proj_dist.mean(), float(np.median(proj_dist)), proj_dist.max()))
report('')
n_segments_used = len(np.unique(best_seg))
report('segment assignment: {} of {} segments used by at least one element'.format(
    n_segments_used, len(toolpath_xyz) - 1))
seg_counts = np.bincount(best_seg, minlength=len(toolpath_xyz) - 1)
report('  elements per used segment: min={} median={:.1f} max={}'.format(
    seg_counts[seg_counts > 0].min(), float(np.median(seg_counts[seg_counts > 0])), seg_counts.max()))
report('')
report('composition field validity:')
report('  any NaN? {}'.format(bool(np.isnan(composition).any())))
report('  any inf? {}'.format(bool(np.isinf(composition).any())))
report('  range: [{:.6f}, {:.6f}] (must be within [0,1])'.format(composition.min(), composition.max()))
report('')
report('comparison vs. global_x composition field:')
diff = np.abs(composition - composition_global_x)
report('  mean abs difference: {:.6f}'.format(diff.mean()))
report('  max abs difference:  {:.6f}'.format(diff.max()))
corr = float(np.corrcoef(composition, composition_global_x)[0, 1])
report('  correlation: {:.4f}'.format(corr))

report_path = os.path.join(OUTDIR, 'part002_mapping_validation_report.txt')
with open(report_path, 'w') as f:
    f.write('\n'.join(report_lines) + '\n')
print('\nsaved report to {}'.format(report_path))
