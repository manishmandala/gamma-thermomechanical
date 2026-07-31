# Assembles a new .k deck around a synthetic toolpath (from
# generate_raster_toolpath.py) while keeping part004's real mesh
# (*NODE/*ELEMENT_SOLID) and all boundary-condition/material blocks
# completely unchanged. The only things that change from the base .k are:
#   - *TOOL_FILE's path (points at the new toolpath)
#   - *CONTROL_TERMINATION's ENDTIM (matches the new toolpath's total time)
#   - the *DEFINE_CURVE birth-time table (recomputed for the new toolpath via
#     the existing, already-tested gamma.simulator.preprocessor.assign_birth_time,
#     not a new algorithm)
#
# Birth-curve convention (matches gamma.py's parser exactly): gamma.py
# defaults ALL elements to birth=0.0 before reading *DEFINE_CURVE, and the
# curve only lists OVERRIDES for elements with a real nonzero birth time.
# So only build-region elements that got a positive birth time are written
# to the new curve - substrate elements (birth=0) are correctly omitted,
# matching the original file's own convention.

import argparse
import os

import numpy as np
from gamma.simulator.preprocessor import assign_birth_time, load_mesh_file
from gamma.simulator.gamma import load_toolpath

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--base-k', required=True, help='part004\'s original plain 4.k (mesh source, unchanged)')
parser.add_argument('--toolpath', required=True, help='new synthetic .crs file')
parser.add_argument('--toolpath-ref-in-deck', required=True,
                     help='the *TOOL_FILE path to write into the new deck (relative to --input-data-dir at run time)')
parser.add_argument('--radius', type=float, default=1.12, help='beam radius for birth assignment (matches GAUSS_LASER)')
parser.add_argument('--path-resolution', type=float, default=0.05)
parser.add_argument('--never-reached-sentinel', type=float, default=1e5,
                     help='birth time assigned to any element the toolpath never reaches (never activates in-window)')
parser.add_argument('--out', required=True)
args = parser.parse_args()

nodes, elements = load_mesh_file(args.base_k)
ele_nodes = nodes[elements]
ele_ctrl = ele_nodes.sum(axis=1) / 8.0
ele_topz = ele_nodes[:, :, 2].max(axis=1)

toolpath = load_toolpath(args.toolpath)
element_birth = -1.0 * np.ones(elements.shape[0])
element_birth[ele_topz <= 0] = 0.0  # substrate: already active at t=0, same convention as preprocessor.write_birth
n_substrate = int((ele_topz <= 0).sum())
n_build = elements.shape[0] - n_substrate

assign_birth_time(ele_nodes, ele_ctrl, ele_topz, toolpath, element_birth, args.radius, args.path_resolution, 0)

never_reached_mask = (element_birth < 0) & (ele_topz > 0)
n_never_reached = int(never_reached_mask.sum())
if n_never_reached:
    element_birth[never_reached_mask] = args.never_reached_sentinel
print('build elements: {}, substrate elements: {}, never-reached: {} ({:.3f}%)'.format(
    n_build, n_substrate, n_never_reached, 100 * n_never_reached / max(n_build, 1)))
print('birth-time range (build elements): [{:.4f}, {:.4f}]'.format(
    element_birth[ele_topz > 0].min(), element_birth[ele_topz > 0][element_birth[ele_topz > 0] < args.never_reached_sentinel].max()
    if (element_birth[ele_topz > 0] < args.never_reached_sentinel).any() else float('nan')))

element_base = 1  # confirmed: part004's *ELEMENT_SOLID starts at eid=1

with open(args.base_k) as f:
    lines = f.readlines()

tool_file_idx = next(i for i, l in enumerate(lines) if l.strip() == '*TOOL_FILE') + 1
endtim_idx = next(i for i, l in enumerate(lines) if l.strip() == '*CONTROL_TERMINATION') + 2
curve_kw_idx = next(i for i, l in enumerate(lines) if l.strip() == '*DEFINE_CURVE')
node_idx = next(i for i, l in enumerate(lines) if l.strip() == '*NODE')

new_lines = list(lines[:tool_file_idx])  # through and including '*TOOL_FILE'
new_lines.append(args.toolpath_ref_in_deck + '\n')  # replaces the old path line
new_lines += lines[tool_file_idx + 1:endtim_idx]  # GAUSS_LASER ... through ENDTIM's comment line
new_lines.append('   {:.10f}\n'.format(toolpath[-1, 0]))  # new ENDTIM value
new_lines += lines[endtim_idx + 1:curve_kw_idx + 1]  # rest of header, up through and including '*DEFINE_CURVE'
new_lines.append('         1                 1.0       1.0\n')  # curve-ID header, matches original convention
build_idx = np.where(ele_topz > 0)[0]
for idx in build_idx:
    eid = idx + element_base
    new_lines.append('{:20.8f}{:20.8f}\n'.format(element_birth[idx], float(eid)))
new_lines += lines[node_idx:]

with open(args.out, 'w') as f:
    f.writelines(new_lines)
print('wrote {}'.format(args.out))
