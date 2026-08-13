# Renders presentation-quality images from a COMPLETED (100%-depth) full
# build: final-frame temperature field and final-frame composition field,
# filtered to the build region only (excludes the substrate, which is much
# larger in extent than the build and mostly sits at/near ambient - including
# it drowns out the actual part of interest and renders as a flat black slab
# under most colormaps). Reusable across any dataset's full-build result.

import argparse
import os

import pyvista as pv

pv.OFF_SCREEN = True

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--vtu', required=True, help='final-frame result VTU (e.g. results_.../wall_0149.vtu)')
parser.add_argument('--out-prefix', required=True, help='output path prefix, e.g. .../part004_full')
parser.add_argument('--camera-position', required=True, nargs=3, type=float)
parser.add_argument('--camera-focal-point', required=True, nargs=3, type=float)
parser.add_argument('--camera-up', nargs=3, type=float, default=[0.0, 0.0, 1.0])
parser.add_argument('--zoom', type=float, default=1.3)
parser.add_argument('--build-matid', type=int, default=1)
parser.add_argument('--panel-px', type=int, default=1800)
args = parser.parse_args()

grid = pv.read(args.vtu)
build = grid.extract_cells(grid.cell_data['material'] == args.build_matid)
print('loaded {}: {} build cells, TIME={:.2f}s'.format(
    args.vtu, build.n_cells, float(grid.field_data['TIME'][0])))


def render(mesh, scalar, cmap, clim, out_path):
    p = pv.Plotter(off_screen=True, window_size=(args.panel_px, args.panel_px))
    p.set_background('white')
    p.add_mesh(mesh, scalars=scalar, cmap=cmap, clim=clim, show_edges=False, show_scalar_bar=True,
               silhouette=dict(color='black', line_width=3.0))
    p.camera.position = tuple(args.camera_position)
    p.camera.focal_point = tuple(args.camera_focal_point)
    p.camera.up = tuple(args.camera_up)
    p.camera.zoom(args.zoom)
    p.screenshot(out_path)
    print('wrote {}'.format(out_path))


t = build.point_data['temp']
print('build-region temp range: [{:.1f}, {:.1f}]K, mean {:.1f}K'.format(t.min(), t.max(), t.mean()))
render(build, 'temp', 'inferno', None, args.out_prefix + '_temperature.png')

if 'composition' in build.cell_data:
    render(build, 'composition', 'viridis', [0, 1], args.out_prefix + '_composition.png')
