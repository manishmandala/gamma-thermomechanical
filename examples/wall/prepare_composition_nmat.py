# General N-material composition-graded .k file generator - the general
# form of prepare_composition.py (exactly 2 fixed materials, single-scalar
# presets that assume [1-frac, frac]). Materials, material count, blend
# shape, and coordinate mode are all CLI-selectable instead of hardcoded:
#
#   --materials TI64,IN718,1018,CPTi   any N materials from MATERIAL_LIBRARY,
#                                       in order along the coordinate
#   --shape parabola|sinusoidal        which per-material blend kernel
#   --coordinate-mode global_x|...     same modes as composition_lib.py
#
# Shape: every material gets a kernel-shaped "zone" of width `--width`
# (default 1/N) centered at its own evenly-spaced position along the
# normalized coordinate s; per-element fractions are each material's kernel
# value normalized to sum to 1 - material k dominates its own zone and
# blends into its neighbors along whatever curve the kernel describes
# (parabola, raised-cosine/sinusoidal, ...). This generalizes the same idea
# composition_lib.py's 2-material `quadratic`/`sinusoidal` presets encode,
# to any N.
#
# Substrate is auto-detected by birth time (all-zero birth) and left
# untouched, same approach as prepare_composition.py.

import argparse
import os
import numpy as np
import cupy as cp
from gamma.simulator.gamma import domain_mgr, load_toolpath
from composition_lib import (compute_centroid, compute_bounds, coordinate_function,
                              filter_deposit_segments, COORDINATE_MODES)

# Known material property sets (density, solidus, liquidus, latent) + Cp/Cond
# curve file paths, relative to examples/wall - the directory every dataset
# in this project resolves curve paths against. Add a material here once and
# it becomes selectable via --materials for any N and any blend shape.
MATERIAL_LIBRARY = {
    'TI64':  dict(density=0.00440, solidus=1878, liquidus=1928, latent=286,
                   cp='../0_properties/TI64_cp.txt', cond='../0_properties/TI64_cond.txt'),
    'IN718': dict(density=0.00819, solidus=1533, liquidus=1609, latent=270,
                   cp='../0_properties/IN718_cp.txt', cond='../0_properties/IN718_cond.txt'),
    '1018':  dict(density=0.00787, solidus=1693, liquidus=1733, latent=270,
                   cp='../data/materials/1018_cp.txt', cond='../data/materials/1018_cond.txt'),
    'CPTi':  dict(density=0.00451, solidus=1933, liquidus=1941, latent=295,
                   cp='../0_properties/CPTi_cp.txt', cond='../0_properties/CPTi_cond.txt'),
    'Al':    dict(density=0.00270, solidus=930, liquidus=933, latent=397,
                   cp='../0_properties/Al_cp.txt', cond='../0_properties/Al_cond.txt'),
    'Cu':    dict(density=0.00896, solidus=1353, liquidus=1358, latent=205,
                   cp='../0_properties/Cu_cp.txt', cond='../0_properties/Cu_cond.txt'),
}

# Per-material blend kernels for the N-material partition of unity: material
# k gets weight kernel(|s-center_k|, width); both kernels here reach exactly
# 0 at distance `width` from their own center, just along a different curve
# on the way there. Add a new kernel here to get a new --shape choice.
BLEND_SHAPES = {
    'parabola': lambda d, width: np.clip(1.0 - (d / width) ** 2, 0.0, None),
    'sinusoidal': lambda d, width: np.clip(0.5 * (1.0 + np.cos(np.pi * np.minimum(d / width, 1.0))), 0.0, None),
}


def n_material_fractions(s, n_materials, shape, width=None):
    """s: (M,) normalized coordinate in [0,1]. Returns (M, n_materials)
    fractions, each row summing to 1. Materials sit at evenly spaced zone
    centers (2k+1)/(2N) along s; default width=1/N makes adjacent zones
    touch zero exactly at their neighbor's center, so every point in [0,1]
    is covered by at least one nonzero kernel."""
    width = width if width is not None else 1.0 / n_materials
    centers = (2 * np.arange(n_materials) + 1) / (2.0 * n_materials)
    d = np.abs(s[:, None] - centers[None, :])
    w = BLEND_SHAPES[shape](d, width)
    return w / w.sum(axis=1, keepdims=True)


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--dataset', default='.', help='dataset folder (default: .)')
parser.add_argument('--kfile', default='thinwall_clean.k', help='input .k filename within --dataset')
parser.add_argument('--toolpath', default='toolpath.crs', help='toolpath filename within --dataset')
parser.add_argument('--materials', required=True,
                     help='comma-separated material names from MATERIAL_LIBRARY ({}), in order along '
                          'the coordinate, e.g. TI64,IN718,1018,CPTi. N = however many are listed.'.format(
                              ','.join(sorted(MATERIAL_LIBRARY))))
parser.add_argument('--shape', choices=sorted(BLEND_SHAPES), default='parabola', help='blend kernel (default: parabola)')
parser.add_argument('--coordinate-mode', choices=COORDINATE_MODES, default='global_x')
parser.add_argument('--width', type=float, default=None, help='blend kernel half-width (default: 1/N)')
parser.add_argument('--local-normalize', action='store_true',
                     help='only meaningful with --coordinate-mode toolpath_arc_length - rescale s to the '
                          'build region\'s own observed min/max instead of the full toolpath (see '
                          'prepare_composition.py\'s flag of the same name for the full rationale)')
parser.add_argument('--out', required=True, help='output .k path - must not already exist')
parser.add_argument('--device', type=int, default=0, help='CUDA device id for the read-only parse (default 0)')
args = parser.parse_args()

if os.path.exists(args.out):
    raise SystemExit('refusing to overwrite existing file: {}'.format(args.out))

material_names = args.materials.split(',')
unknown = [m for m in material_names if m not in MATERIAL_LIBRARY]
if unknown:
    raise SystemExit('unknown material(s) {} - known: {}'.format(unknown, sorted(MATERIAL_LIBRARY)))
n_materials = len(material_names)
if n_materials < 2:
    raise SystemExit('--materials needs at least 2 materials, got {}'.format(n_materials))
materials = [MATERIAL_LIBRARY[m] for m in material_names]

cp.cuda.Device(args.device).use()
src_path = os.path.join(args.dataset, args.kfile)

# --- real parse, read-only, to find build-vs-substrate by birth time ---
d = domain_mgr(filename=src_path, toolpathdir=os.path.join(args.dataset, args.toolpath),
                input_data_dir=args.dataset, verbose=False)
assert len(d.mat_thermal) == 2, 'expected exactly 2 ordinary materials, found {}'.format(len(d.mat_thermal))
assert len(d.mat_graded) == 0, 'source file already has a graded region - not the expected clean 2-material input'

mat_ids = [m[0] for m in d.mat_thermal]
substrate_candidates = [m for m in mat_ids if bool((d.element_birth[d.element_mat == m] == 0).all())]
assert len(substrate_candidates) == 1, 'expected exactly one all-zero-birth material (the substrate)'
substrate_pid = substrate_candidates[0]
build_pid = [m for m in mat_ids if m != substrate_pid][0]
print('auto-detected by birth time: substrate=matID {}, build region=matID {} ({} elements)'.format(
    substrate_pid, build_pid, int((d.element_mat == build_pid).sum())))
print('materials: {} (N={}), shape={}, coordinate-mode={}'.format(
    material_names, n_materials, args.shape, args.coordinate_mode))

with open(src_path) as f:
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
build_element_ids = []
build_element_node_ids = {}
for l in lines[elem_start + 1:elem_end]:
    if l.startswith('$'):
        continue
    text = l.split()
    eid, pid = int(text[0]), int(text[1])
    if pid == build_pid:
        build_element_ids.append(eid)
        build_element_node_ids[eid] = [int(t) for t in text[2:10]]
assert len(build_element_ids) == int((d.element_mat == build_pid).sum())

build_start = next(s for s in (i for i, l in enumerate(lines) if l.startswith('*MAT_THERMAL_ISOTROPIC_TD'))
                    if int(lines[s + 2].split()[0]) == build_pid)
build_end = next(i for i in range(build_start + 1, len(lines)) if lines[i].startswith('*'))

# --- normalized coordinate s along --coordinate-mode ---
centroids = np.array([compute_centroid(nodes, build_element_node_ids[eid]) for eid in build_element_ids])
if args.coordinate_mode == 'toolpath_arc_length':
    toolpath_raw = load_toolpath(os.path.join(args.dataset, args.toolpath))
    toolpath_xyz = filter_deposit_segments(toolpath_raw[:, 1:4], toolpath_raw[:, 4])
    s = coordinate_function(centroids, args.coordinate_mode, toolpath=toolpath_xyz)
    if args.local_normalize:
        s_lo, s_hi = float(s.min()), float(s.max())
        print('--local-normalize: build region occupies global s=[{:.4f}, {:.4f}] - rescaling to [0,1]'.format(
            s_lo, s_hi))
        s = (s - s_lo) / (s_hi - s_lo) if s_hi > s_lo else np.zeros_like(s)
else:
    bounds = compute_bounds(centroids)
    s = coordinate_function(centroids, args.coordinate_mode, bounds=bounds)

fractions = n_material_fractions(s, n_materials, args.shape, width=args.width)

new_mat_block = ['*MAT_THERMAL_GRADED_TD\n',
                  '$HMNAME MATS     {}GRADED_{}MAT_{}_{}\n'.format(
                      build_pid, n_materials, args.shape.upper(), args.coordinate_mode.upper())]
data_line = '      %5d   %5d' % (build_pid, n_materials)
for m in materials:
    data_line += '   %.5f   %.1f   %.1f   %.1f' % (m['density'], m['solidus'], m['liquidus'], m['latent'])
new_mat_block.append(data_line + '\n')
for m in materials:
    new_mat_block.append(m['cp'] + '\n')
    new_mat_block.append(m['cond'] + '\n')

lines = lines[:build_start] + new_mat_block + lines[build_end:]

node_insert_at = next(i for i, l in enumerate(lines) if l.startswith('*NODE'))
comp_lines = ['*ELEMENT_COMPOSITION\n', '%10d\n' % build_pid]
for i, eid in enumerate(build_element_ids):
    row = ''.join('%20.8f' % f for f in fractions[i])
    comp_lines.append('%s%20d\n' % (row, eid))
lines = lines[:node_insert_at] + comp_lines + lines[node_insert_at:]

with open(args.out, 'w') as f:
    f.writelines(lines)

print('wrote {} - {} materials ({}), {} shape over {}, {} graded elements'.format(
    args.out, n_materials, ','.join(material_names), args.shape, args.coordinate_mode, len(build_element_ids)))
