# General diagnostic + validation report for the toolpath_arc_length
# composition mapping on any dataset (parameterized version of
# diagnose_part002_mapping.py, which stays as-is for reproducibility of the
# part002 results already reported). Does NOT run the GAMMA thermal solver -
# this only exercises the preprocessing pipeline (compute_centroid ->
# coordinate_function -> composition_function) and exports what it computed.
# Does not modify composition_lib.py's algorithm or normalization in any way.

import argparse
import os
import sys

import numpy as np
import pyvista as pv
import vtk
import cupy as cp
cp.cuda.Device(0).use()
from gamma.simulator.gamma import domain_mgr, load_toolpath

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from composition_lib import (compute_centroid, compute_bounds, coordinate_function, composition_function,
                              build_toolpath_arc_length_table, project_to_toolpath, filter_deposit_segments)

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--dataset', required=True)
parser.add_argument('--kfile', required=True)
parser.add_argument('--toolpath', default='toolpath.crs')
parser.add_argument('--out-dir', required=True)
parser.add_argument('--seam-jump-threshold', type=float, default=0.15,
                     help='s-jump-to-neighbor threshold for flagging seam/repeated-pass elements '
                          '(same value used for part002)')
args = parser.parse_args()
os.makedirs(args.out_dir, exist_ok=True)

# ---- load source geometry + auto-detect build region (same pattern as prepare_composition.py) ----
d = domain_mgr(filename=os.path.join(args.dataset, args.kfile), toolpathdir=os.path.join(args.dataset, args.toolpath),
                input_data_dir=args.dataset, verbose=False)
mat_ids = [m[0] for m in d.mat_thermal]
substrate_pid = [m for m in mat_ids if bool((d.element_birth[d.element_mat == m] == 0).all())][0]
build_pid = [m for m in mat_ids if m != substrate_pid][0]

with open(os.path.join(args.dataset, args.kfile)) as f:
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

print('build region: matID {} ({} elements), substrate matID {}'.format(build_pid, len(build_eids), substrate_pid))
centroids = np.array([compute_centroid(nodes, build_node_ids[eid]) for eid in build_eids])

# ---- toolpath-arc-length mapping ----
toolpath_raw = load_toolpath(os.path.join(args.dataset, args.toolpath))
toolpath_xyz = filter_deposit_segments(toolpath_raw[:, 1:4], toolpath_raw[:, 4])
arc_table = build_toolpath_arc_length_table(toolpath_xyz)
total_length = arc_table[-1]
arc_at, proj_dist, best_seg = project_to_toolpath(centroids, toolpath_xyz, arc_table, return_segment=True)
s = coordinate_function(centroids, 'toolpath_arc_length', toolpath=toolpath_xyz)
assert np.allclose(s, np.clip(arc_at / total_length, 0, 1)), 's does not match arc_at/total_length'
composition = np.array([composition_function(float(si), 0.0, 'sinusoidal') for si in s])

# ---- global_x, for the comparison figure/stats ----
bounds = compute_bounds(centroids)
s_global_x = coordinate_function(centroids, 'global_x', bounds=bounds)
composition_global_x = np.array([composition_function(float(si), 0.0, 'sinusoidal') for si in s_global_x])

# ---- CSV export ----
csv_path = os.path.join(args.out_dir, 'composition_diagnostics.csv')
with open(csv_path, 'w') as f:
    f.write('element_id,centroid_x,centroid_y,centroid_z,s_arc_length,composition_arc_length,'
            'projection_distance,best_segment,s_global_x,composition_global_x\n')
    for i, eid in enumerate(build_eids):
        f.write('{},{:.6f},{:.6f},{:.6f},{:.8f},{:.8f},{:.6f},{},{:.8f},{:.8f}\n'.format(
            eid, centroids[i, 0], centroids[i, 1], centroids[i, 2],
            s[i], composition[i], proj_dist[i], best_seg[i], s_global_x[i], composition_global_x[i]))
print('wrote {}'.format(csv_path))

# ---- VTU export: path_coordinate_s, toolpath composition, global_x composition, projection distance ----
all_node_ids = sorted(nodes)
node_id_to_local = {nid: i for i, nid in enumerate(all_node_ids)}
points = np.array([nodes[nid] for nid in all_node_ids])
local_cells = []
for eid in build_eids:
    local_cells += [8] + [node_id_to_local[n] for n in build_node_ids[eid]]
local_cells = np.array(local_cells)
cell_types = np.array([vtk.VTK_HEXAHEDRON] * len(build_eids))

grid = pv.UnstructuredGrid(local_cells, cell_types, points)
grid.cell_data['path_coordinate_s'] = s
grid.cell_data['composition_arc_length'] = composition
grid.cell_data['composition_global_x'] = composition_global_x
grid.cell_data['projection_distance'] = proj_dist
grid.cell_data['best_segment'] = best_seg.astype(float)
vtu_path = os.path.join(args.out_dir, 'composition_diagnostics.vtu')
grid.save(vtu_path)
print('wrote {}'.format(vtu_path))

# ---- seam / repeated-pass / large-neighbor-jump inspection ----
max_jump = np.zeros(grid.n_cells)
for i in range(grid.n_cells):
    nb = grid.cell_neighbors(i, connections='points')
    if len(nb):
        max_jump[i] = np.abs(s[nb] - s[i]).max()
seam_mask = max_jump > args.seam_jump_threshold
n_seam = int(seam_mask.sum())
seam_info = []
if n_seam:
    seam_z = centroids[seam_mask, 2]
    seam_segs = best_seg[seam_mask]
    seam_info.append('  z range of flagged elements: [{:.3f}, {:.3f}]'.format(seam_z.min(), seam_z.max()))
    seam_info.append('  best_segment range of flagged elements: [{}, {}]'.format(seam_segs.min(), seam_segs.max()))

# ---- statistics ----
report_lines = []


def report(line):
    print(line)
    report_lines.append(line)


report('\n{} toolpath_arc_length mapping - validation statistics'.format(os.path.basename(args.dataset)))
report('=' * 60)
report('build elements: {}'.format(len(build_eids)))
report('toolpath: {} waypoints, {} segments, total length {:.4f}'.format(
    len(toolpath_xyz), len(toolpath_xyz) - 1, total_length))
report('')
report('s (normalized path coordinate):')
report('  min={:.6f}  max={:.6f}'.format(s.min(), s.max()))
near0 = (s < 0.05).sum()
near1 = (s > 0.95).sum()
report('  elements with s < 0.05: {} ({:.2f}%)'.format(near0, 100 * near0 / len(s)))
report('  elements with s > 0.95: {} ({:.2f}%)'.format(near1, 100 * near1 / len(s)))
report('')
report('nearest-projection distance (element centroid to toolpath):')
report('  mean={:.4f}  median={:.4f}  max={:.4f}'.format(proj_dist.mean(), float(np.median(proj_dist)), proj_dist.max()))
report('')
n_unique_6dp = len(set(np.round(s, 6)))
n_unique_3dp = len(set(np.round(s, 3)))
report('unique s values: {} of {} elements at 6dp ({:.1f}% unique), {} at 3dp'.format(
    n_unique_6dp, len(s), 100 * n_unique_6dp / len(s), n_unique_3dp))
report('')
n_segments_used = len(np.unique(best_seg))
report('segment assignment: {} of {} segments used by at least one element'.format(
    n_segments_used, len(toolpath_xyz) - 1))
report('')
report('composition field validity:')
report('  any NaN? {}  any inf? {}'.format(bool(np.isnan(composition).any()), bool(np.isinf(composition).any())))
report('  range: [{:.6f}, {:.6f}] (must be within [0,1])'.format(composition.min(), composition.max()))
out_of_range = ((composition < 0) | (composition > 1)).sum()
report('  out-of-[0,1] values: {}'.format(int(out_of_range)))
report('')
report('seam / repeated-pass / large-neighbor-jump inspection (threshold={}):'.format(args.seam_jump_threshold))
report('  flagged elements: {} of {} ({:.3f}%)'.format(n_seam, len(s), 100 * n_seam / len(s)))
for line in seam_info:
    report(line)
report('')
report('comparison vs. global_x composition field:')
diff = np.abs(composition - composition_global_x)
report('  mean abs difference: {:.6f}  max abs difference: {:.6f}'.format(diff.mean(), diff.max()))
report('  correlation: {:.4f}'.format(float(np.corrcoef(composition, composition_global_x)[0, 1])))

report_path = os.path.join(args.out_dir, 'mapping_validation_report.txt')
with open(report_path, 'w') as f:
    f.write('\n'.join(report_lines) + '\n')
print('\nsaved report to {}'.format(report_path))
