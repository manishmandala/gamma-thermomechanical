# Presentation-quality side-by-side figure: global_x vs toolpath_arc_length
# composition field on part002 (the hexagonal ring dataset). Renders each
# panel independently with pyvista (off-screen, white background, no
# axes/widgets/UI), then composes them with matplotlib for precise control
# over titles, a single shared colorbar, and export resolution - pyvista's
# own multi-panel compositing doesn't give fine enough typography control
# for a slide-ready figure.

import numpy as np
import pyvista as pv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

pv.OFF_SCREEN = True

DIAG_VTU = '../incoming_dataset/part002_LP800_SSp10_H2.24_SSt3_LH0.9/toolpath_arc_length_composition/part002_composition_diagnostics.vtu'
OUT_DIR = '../incoming_dataset/part002_LP800_SSp10_H2.24_SSt3_LH0.9/toolpath_arc_length_composition/comparison'
CMAP = 'viridis'  # perceptually uniform, colorblind-safe
CLIM = [0.25, 0.75]  # shared range - both fields' actual min/max are ~0.25/0.75

# same camera for both panels, locked to explicit values (not the 'iso' shorthand,
# which pyvista derives from internal state) so both renders are guaranteed identical
CAMERA_POSITION = (160.87343668733345, 160.87343668733345, 155.27343618733346)
CAMERA_FOCAL_POINT = (0.0, 0.0, -5.6000005)
CAMERA_UP = (0.0, 0.0, 1.0)
ZOOM = 1.3
PANEL_PX = 1800  # per-panel render resolution; composed figure comfortably exceeds 3000px wide

grid = pv.read(DIAG_VTU)


def render_panel(scalar_name, outline=False):
    p = pv.Plotter(off_screen=True, window_size=(PANEL_PX, PANEL_PX))
    p.set_background('white')
    p.add_mesh(grid, scalars=scalar_name, cmap=CMAP, clim=CLIM, show_edges=False, show_scalar_bar=False,
               silhouette=dict(color='black', line_width=3.0) if outline else False)
    p.camera.position = CAMERA_POSITION
    p.camera.focal_point = CAMERA_FOCAL_POINT
    p.camera.up = CAMERA_UP
    p.camera.zoom(ZOOM)
    img = p.screenshot(return_img=True)
    p.close()
    return img


def shared_crop(img_a, img_b, pad_frac=0.06):
    """Same idea as autocrop, but computed as the UNION of both images' content
    boxes and applied identically to both - guarantees identical scale/centering
    between panels rather than each cropping independently to slightly different boxes."""
    non_white_a = np.any(img_a[:, :, :3] < 250, axis=2)
    non_white_b = np.any(img_b[:, :, :3] < 250, axis=2)
    non_white = non_white_a | non_white_b
    rows = np.where(non_white.any(axis=1))[0]
    cols = np.where(non_white.any(axis=0))[0]
    r0, r1 = rows.min(), rows.max()
    c0, c1 = cols.min(), cols.max()
    pad_r = int((r1 - r0) * pad_frac)
    pad_c = int((c1 - c0) * pad_frac)
    r0, r1 = max(0, r0 - pad_r), min(img_a.shape[0], r1 + pad_r)
    c0, c1 = max(0, c0 - pad_c), min(img_a.shape[1], c1 + pad_c)
    return img_a[r0:r1, c0:c1], img_b[r0:r1, c0:c1]


def compose(outline, out_path):
    left_img = render_panel('composition_global_x', outline=outline)
    right_img = render_panel('composition_arc_length', outline=outline)
    left_img, right_img = shared_crop(left_img, right_img)

    fig = plt.figure(figsize=(20, 11), facecolor='white')
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.045], wspace=0.03)

    ax_left = fig.add_subplot(gs[0, 0])
    ax_left.imshow(left_img)
    ax_left.set_title('Global X Gradient', fontsize=30, fontweight='bold', pad=16)
    ax_left.axis('off')

    ax_right = fig.add_subplot(gs[0, 1])
    ax_right.imshow(right_img)
    ax_right.set_title('Toolpath Arc-Length Gradient', fontsize=30, fontweight='bold', pad=16)
    ax_right.axis('off')

    cax = fig.add_subplot(gs[0, 2])
    norm = matplotlib.colors.Normalize(vmin=CLIM[0], vmax=CLIM[1])
    sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=CMAP)
    cb = fig.colorbar(sm, cax=cax)
    cb.set_label('Composition Fraction', fontsize=20, labelpad=14)
    cb.ax.tick_params(labelsize=16)

    fig.savefig(out_path, dpi=300, facecolor='white', bbox_inches='tight')
    plt.close(fig)
    print('wrote {} ({}px wide)'.format(out_path, fig.get_size_inches()[0] * 300))


compose(outline=False, out_path=OUT_DIR + '/global_x_vs_toolpath_gradient.png')
compose(outline=True, out_path=OUT_DIR + '/global_x_vs_toolpath_gradient_outlined.png')
