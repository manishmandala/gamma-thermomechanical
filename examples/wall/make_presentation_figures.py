# Three presentation-quality figures built from the completed part002
# results:
#   Figure 1: composition-mapping comparison (global_x vs toolpath_arc_length)
#   Figure 2: thermal validation summary (max diff vs time, peak temp vs time)
#   Figure 3: spatial temperature-difference field at the frame where the
#             372.19K transient maximum occurred (not the final frame)
#
# Renders via pyvista (off-screen, white background, no axes/widgets/UI),
# composites/plots via matplotlib for precise typography and export
# resolution control.

import glob
import os

import numpy as np
import pyvista as pv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

pv.OFF_SCREEN = True

DATASET = '../incoming_dataset/part002_LP800_SSp10_H2.24_SSt3_LH0.9'
COMP_DIR = DATASET + '/toolpath_arc_length_composition'
DIAG_VTU = COMP_DIR + '/part002_composition_diagnostics.vtu'
RESULTS_ARC = COMP_DIR + '/results_arclength_sinusoidal_10pct'
RESULTS_GX = COMP_DIR + '/results_globalx_sinusoidal_10pct'
OUT_DIR = COMP_DIR + '/comparison'

# same explicit camera as the earlier gradient-comparison figure - locked to
# literal values (not the 'iso' shorthand) so every figure in this project
# that shows part002's geometry uses an identical, reproducible viewpoint
CAMERA_POSITION = (160.87343668733345, 160.87343668733345, 155.27343618733346)
CAMERA_FOCAL_POINT = (0.0, 0.0, -5.6000005)
CAMERA_UP = (0.0, 0.0, 1.0)
ZOOM = 1.3
PANEL_PX = 1800

plt.rcParams['font.size'] = 14


def render_mesh(mesh, scalar_name, clim, cmap, outline=True, edges=False):
    p = pv.Plotter(off_screen=True, window_size=(PANEL_PX, PANEL_PX))
    p.set_background('white')
    # edges=True: needed whenever the scalar field is heavily skewed toward one
    # end of the colormap (e.g. a mostly-near-zero difference field under a
    # colormap that goes to black at 0) - without visible facet edges, a
    # nearly-uniformly-dark surface loses all 3D structure and reads as a flat
    # black silhouette instead of recognizable geometry.
    p.add_mesh(mesh, scalars=scalar_name, cmap=cmap, clim=clim,
               show_edges=edges, edge_color='#888888', line_width=0.4,
               show_scalar_bar=False,
               silhouette=dict(color='black', line_width=3.0) if outline else False)
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


def single_crop(img, pad_frac=0.06):
    non_white = np.any(img[:, :, :3] < 250, axis=2)
    rows = np.where(non_white.any(axis=1))[0]
    cols = np.where(non_white.any(axis=0))[0]
    r0, r1 = rows.min(), rows.max()
    c0, c1 = cols.min(), cols.max()
    pad_r, pad_c = int((r1 - r0) * pad_frac), int((c1 - c0) * pad_frac)
    r0, r1 = max(0, r0 - pad_r), min(img.shape[0], r1 + pad_r)
    c0, c1 = max(0, c0 - pad_c), min(img.shape[1], c1 + pad_c)
    return img[r0:r1, c0:c1]


# =========================================================================
# Figure 1: composition-mapping comparison, full [0,1] scale
# =========================================================================
print('=== Figure 1: composition mapping comparison ===')
diag = pv.read(DIAG_VTU)
left_img = render_mesh(diag, 'composition_global_x', clim=[0, 1], cmap='viridis', outline=True)
right_img = render_mesh(diag, 'composition_arc_length', clim=[0, 1], cmap='viridis', outline=True)
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
sm = matplotlib.cm.ScalarMappable(norm=norm, cmap='viridis')
cb = fig.colorbar(sm, cax=cax)
cb.set_label('Composition Fraction', fontsize=20, labelpad=14)
cb.ax.tick_params(labelsize=16)

fig1_path = OUT_DIR + '/global_x_vs_toolpath_gradient_presentation.png'
fig.savefig(fig1_path, dpi=300, facecolor='white', bbox_inches='tight')
plt.close(fig)
print('wrote {}'.format(fig1_path))

# =========================================================================
# Load 10%-depth frames, recompute the time series (not saved to disk
# earlier - only the report text/PNGs were kept) and locate the exact
# frame/time of the 372.19K transient maximum
# =========================================================================
print('\n=== Recomputing 10%-depth time series ===')


def load_frames(folder):
    frames = sorted(glob.glob(os.path.join(folder, 'wall_*.vtu')))
    by_time = {}
    for f in frames:
        g = pv.read(f)
        t = round(float(g.field_data['TIME'][0]), 6)
        by_time[t] = (g, os.path.basename(f))
    return by_time


arc_frames = load_frames(RESULTS_ARC)
gx_frames = load_frames(RESULTS_GX)
common_times = sorted(set(arc_frames) & set(gx_frames))
times = np.array(common_times)

max_diff_hist, peak_arc_hist, peak_gx_hist = [], [], []
for t in common_times:
    ga, _ = arc_frames[t]
    gg, _ = gx_frames[t]
    ta, tg = ga.point_data['temp'], gg.point_data['temp']
    max_diff_hist.append(float(np.abs(ta - tg).max()))
    peak_arc_hist.append(float(ta.max()))
    peak_gx_hist.append(float(tg.max()))
max_diff_hist = np.array(max_diff_hist)
peak_arc_hist = np.array(peak_arc_hist)
peak_gx_hist = np.array(peak_gx_hist)

max_idx = int(np.argmax(max_diff_hist))
max_time = common_times[max_idx]
max_value = max_diff_hist[max_idx]
arc_frame_file = arc_frames[max_time][1]
gx_frame_file = gx_frames[max_time][1]
print('372.19K-scale maximum: index {} of {}, t={:.3f}s, value={:.4f}K'.format(
    max_idx, len(common_times), max_time, max_value))
print('  arc_length frame file: {}'.format(arc_frame_file))
print('  global_x frame file:   {}'.format(gx_frame_file))

# =========================================================================
# Figure 2: thermal validation summary
# =========================================================================
print('\n=== Figure 2: thermal validation summary ===')
COLOR_GX = '#2a78d6'
COLOR_ARC = '#c0392b'

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7.5), facecolor='white')

ax1.plot(times, max_diff_hist, color='#6a4fb0', linewidth=2.5)
ax1.scatter([max_time], [max_value], color='black', zorder=5, s=80)
ax1.annotate('{:.2f} K\nat t={:.1f}s'.format(max_value, max_time),
             xy=(max_time, max_value), xytext=(max_time - 28, max_value - 40),
             fontsize=16, fontweight='bold',
             arrowprops=dict(arrowstyle='->', lw=1.8, color='black'))
ax1.set_xlabel('Simulation Time (s)', fontsize=18)
ax1.set_ylabel('Max |ΔT| (K)', fontsize=18)
ax1.set_title('Maximum Temperature Difference vs. Time', fontsize=20, fontweight='bold', pad=14)
ax1.tick_params(labelsize=14)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.grid(axis='y', color='#e5e5e5', linewidth=0.8)
ax1.set_axisbelow(True)

ax2.plot(times, peak_gx_hist, color=COLOR_GX, linewidth=2.5, label='Global X')
ax2.plot(times, peak_arc_hist, color=COLOR_ARC, linewidth=2.5, label='Toolpath Arc Length')
ax2.set_xlabel('Simulation Time (s)', fontsize=18)
ax2.set_ylabel('Peak Temperature (K)', fontsize=18)
ax2.set_title('Peak Temperature vs. Time', fontsize=20, fontweight='bold', pad=14)
ax2.tick_params(labelsize=14)
ax2.legend(fontsize=16, frameon=False, loc='lower right')
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.grid(axis='y', color='#e5e5e5', linewidth=0.8)
ax2.set_axisbelow(True)

fig.tight_layout()
fig2_path = OUT_DIR + '/thermal_validation_10pct_presentation.png'
fig.savefig(fig2_path, dpi=300, facecolor='white', bbox_inches='tight')
plt.close(fig)
print('wrote {}'.format(fig2_path))

# =========================================================================
# Figure 3: spatial temperature-difference field at the max-diff frame
# =========================================================================
print('\n=== Figure 3: spatial field at max-difference frame ===')
ga_max, _ = arc_frames[max_time]
gg_max, _ = gx_frames[max_time]
diff_grid = ga_max.copy()
diff_grid.point_data['temperature_difference'] = np.abs(ga_max.point_data['temp'] - gg_max.point_data['temp'])

diff_img = render_mesh(diff_grid, 'temperature_difference', clim=[0, max_value], cmap='inferno', outline=True, edges=True)
diff_img = single_crop(diff_img)

fig = plt.figure(figsize=(13, 11), facecolor='white')
gs = fig.add_gridspec(1, 2, width_ratios=[1, 0.045], wspace=0.04)
ax = fig.add_subplot(gs[0, 0])
ax.imshow(diff_img)
ax.set_title('Spatial Temperature Difference at t = {:.2f}s\n(peak transient difference: {:.2f}K)'.format(
    max_time, max_value), fontsize=22, fontweight='bold', pad=16)
ax.axis('off')

cax = fig.add_subplot(gs[0, 1])
norm = matplotlib.colors.Normalize(vmin=0, vmax=max_value)
sm = matplotlib.cm.ScalarMappable(norm=norm, cmap='inferno')
cb = fig.colorbar(sm, cax=cax)
cb.set_label('Absolute Temperature Difference (K)', fontsize=18, labelpad=14)
cb.ax.tick_params(labelsize=14)

fig3_path = OUT_DIR + '/max_transient_temperature_difference.png'
fig.savefig(fig3_path, dpi=300, facecolor='white', bbox_inches='tight')
plt.close(fig)
print('wrote {}'.format(fig3_path))

print('\n=== Summary ===')
print('372.19K-scale max occurred at frame index {} (0-based, of {} matched frames), t={:.3f}s'.format(
    max_idx, len(common_times), max_time))
print('arc_length source frame: {}/{}'.format(RESULTS_ARC, arc_frame_file))
print('global_x source frame:   {}/{}'.format(RESULTS_GX, gx_frame_file))
