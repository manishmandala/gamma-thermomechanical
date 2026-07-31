# Same side-by-side composition-mapping figure as
# make_presentation_figures.py's Figure 1, applied to part005 (a distinctly
# different geometry - a rectangular block fused with a cylindrical boss,
# not part002's hexagonal ring) as a generalization check. Same camera-
# locking / shared-crop approach so both panels are guaranteed identical
# scale and centering.

import numpy as np
import pyvista as pv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

pv.OFF_SCREEN = True

DIAG_VTU = '../incoming_dataset/part005_LP1000_SSp8_H1.12_SSt4_LH1.0/toolpath_arc_length_composition/composition_diagnostics.vtu'
OUT_DIR = '../incoming_dataset/part005_LP1000_SSp8_H1.12_SSt4_LH1.0/toolpath_arc_length_composition'
CMAP = 'viridis'
CLIM = [0, 1]

CAMERA_POSITION = (70.0, -90.0, 70.0)
CAMERA_FOCAL_POINT = (0.0, 0.0, -1.5)
CAMERA_UP = (0.0, 0.0, 1.0)
ZOOM = 1.0
PANEL_PX = 1800

grid = pv.read(DIAG_VTU)


def render_panel(scalar_name):
    p = pv.Plotter(off_screen=True, window_size=(PANEL_PX, PANEL_PX))
    p.set_background('white')
    p.add_mesh(grid, scalars=scalar_name, cmap=CMAP, clim=CLIM, show_edges=False, show_scalar_bar=False,
               silhouette=dict(color='black', line_width=3.0))
    p.camera.position = CAMERA_POSITION
    p.camera.focal_point = CAMERA_FOCAL_POINT
    p.camera.up = CAMERA_UP
    p.camera.zoom(ZOOM)
    img = p.screenshot(return_img=True)
    p.close()
    return img


def shared_crop(img_a, img_b, pad_frac=0.06):
    non_white = np.any(img_a[:, :, :3] < 250, axis=2) | np.any(img_b[:, :, :3] < 250, axis=2)
    rows = np.where(non_white.any(axis=1))[0]
    cols = np.where(non_white.any(axis=0))[0]
    r0, r1 = rows.min(), rows.max()
    c0, c1 = cols.min(), cols.max()
    pad_r, pad_c = int((r1 - r0) * pad_frac), int((c1 - c0) * pad_frac)
    r0, r1 = max(0, r0 - pad_r), min(img_a.shape[0], r1 + pad_r)
    c0, c1 = max(0, c0 - pad_c), min(img_a.shape[1], c1 + pad_c)
    return img_a[r0:r1, c0:c1], img_b[r0:r1, c0:c1]


left_img = render_panel('composition_global_x')
right_img = render_panel('composition_arc_length')
left_img, right_img = shared_crop(left_img, right_img)

fig = plt.figure(figsize=(20, 11), facecolor='white')
gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.045], wspace=0.03)

ax = fig.add_subplot(gs[0, 0])
ax.imshow(left_img)
ax.set_title('Global X Gradient', fontsize=30, fontweight='bold', pad=16)
ax.axis('off')

ax = fig.add_subplot(gs[0, 1])
ax.imshow(right_img)
ax.set_title('Toolpath-Following Gradient', fontsize=30, fontweight='bold', pad=16)
ax.axis('off')

cax = fig.add_subplot(gs[0, 2])
norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=CMAP)
cb = fig.colorbar(sm, cax=cax)
cb.set_label('Composition Fraction', fontsize=20, labelpad=14)
cb.ax.tick_params(labelsize=16)

out_path = OUT_DIR + '/global_x_vs_toolpath_gradient_part005.png'
fig.savefig(out_path, dpi=300, facecolor='white', bbox_inches='tight')
plt.close(fig)
print('wrote {}'.format(out_path))
