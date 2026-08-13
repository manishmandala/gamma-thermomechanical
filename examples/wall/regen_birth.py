# Recomputes element birth (activation) times to match the current
# toolpath.crs and rewrites thinwall_discrete_bands.k's *DEFINE_CURVE block in place.
# Run this any time generate_toolpath.py's schedule changes (SWEEP/DWELL/Z0/
# LAYER_DZ/N_PASSES/Y_CENTER) - the mesh geometry itself is untouched.
#
# RADIUS must be wide enough for a single centered pass (Y_CENTER=0.0) to
# reach the outer wall columns (+/-0.4): 0.45 clears them with margin.
# PATH_RESOLUTION matches the original birth-curve generation (0.2).

import numpy as np
import gamma.simulator.preprocessor as pre

MESH_FILE = 'thinwall_discrete_bands.k'
TOOLPATH_FILE = 'toolpath.crs'
RADIUS = 0.45
PATH_RESOLUTION = 0.2
MODE = 1    # flat head

with open(MESH_FILE) as f:
    lines = f.readlines()

curve_start = next(i for i, l in enumerate(lines) if l.startswith('*DEFINE_CURVE'))
node_start = next(i for i, l in enumerate(lines) if l.startswith('*NODE'))
lines = lines[:curve_start] + lines[node_start:]

with open(MESH_FILE, 'w') as f:
    f.writelines(lines)

nodes, elements = pre.load_mesh_file(MESH_FILE)
ele_nodes = nodes[elements]
ele_ctrl = ele_nodes.sum(axis=1) / 8
ele_topz = ele_nodes[:, :, 2].max(axis=1)
toolpath = pre.load_toolpath(TOOLPATH_FILE)

element_birth = -1 * np.ones(elements.shape[0])
element_birth[ele_topz <= 0] = 0
pre.assign_birth_time(ele_nodes, ele_ctrl, ele_topz, toolpath, element_birth, RADIUS, PATH_RESOLUTION, MODE)

curve_lines = ['*DEFINE_CURVE\n', '         1                 1.0       1.0\n']
for idx in np.argsort(element_birth):
    b = element_birth[idx]
    if b > 0:
        curve_lines.append("%20.8f%20.8f\n" % (b, idx + 1))
    elif b < 0:
        curve_lines.append("%20.8f%20.8f\n" % (100000, idx + 1))

with open(MESH_FILE) as f:
    lines = f.readlines()
insert_at = next(i for i, l in enumerate(lines) if l.startswith('*NODE'))
lines = lines[:insert_at] + curve_lines + lines[insert_at:]
with open(MESH_FILE, 'w') as f:
    f.writelines(lines)

n_never = int((element_birth < 0).sum())
print("Regenerated birth curve: {} elements, {} never activated (outside build region)".format(
    elements.shape[0], n_never))
