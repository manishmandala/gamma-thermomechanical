# 15%-depth matched comparison, toolpath_arc_length vs global_x (both
# sinusoidal) on part005 - generalization check, same script structure as
# compare_composition_modes_part002_10pct.py (part002). Extends compare_composition_modes_part002_1pct.py's pattern
# (matched-by-TIME comparison) with per-frame time series (not just final-
# frame stats), a max-temperature-difference-vs-time plot, and a spatial
# cross-check against the known repeated-pass seam elements (identified by
# CENTROID COORDINATES, not element index/ID - domain_mgr's sort_birth=True
# reorders elements internally by birth time, so the simulation's active-
# element ordering does NOT match the diagnostic script's file-order
# element list; coordinates are the only reliable link between the two).

import argparse
import glob
import os

import numpy as np
import pyvista as pv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--arclength', required=True)
parser.add_argument('--globalx', required=True)
parser.add_argument('--out', required=True)
parser.add_argument('--seam-csv', default=None,
                     help='composition_diagnostics.csv - used to cross-check the repeated-pass '
                          'seam elements against the largest thermal differences')
parser.add_argument('--seam-jump-threshold', type=float, default=0.15,
                     help='s-jump-to-neighbor threshold used to (re)identify seam elements, matching '
                          'the value used during the original streak diagnosis')
args = parser.parse_args()
os.makedirs(args.out, exist_ok=True)
BUILD_PID = 1


def load_frames(folder):
    frames = sorted(glob.glob(os.path.join(folder, 'wall_*.vtu')))
    if not frames:
        raise SystemExit('no wall_*.vtu frames found in {}'.format(folder))
    by_time = {}
    for f in frames:
        g = pv.read(f)
        by_time[round(float(g.field_data['TIME'][0]), 6)] = g
    return by_time


arc_frames = load_frames(args.arclength)
gx_frames = load_frames(args.globalx)
common_times = sorted(set(arc_frames) & set(gx_frames))
if not common_times:
    raise SystemExit('no matching TIME values between the two runs')
print('matched {} frames by TIME'.format(len(common_times)))

times = np.array(common_times)
max_diff_hist, mean_diff_hist, median_diff_hist, std_diff_hist = [], [], [], []
peak_arc_hist, peak_gx_hist = [], []
mean_build_arc_hist, mean_build_gx_hist = [], []
hottest_history = []
frac_gt1, frac_gt10, frac_gt25 = [], [], []

for t in common_times:
    ga, gg = arc_frames[t], gx_frames[t]
    if ga.n_points != gg.n_points:
        raise SystemExit('point count mismatch at t={}'.format(t))
    ta, tg = ga.point_data['temp'], gg.point_data['temp']
    d = np.abs(ta - tg)

    max_diff_hist.append(float(d.max()))
    mean_diff_hist.append(float(d.mean()))
    median_diff_hist.append(float(np.median(d)))
    std_diff_hist.append(float(d.std()))
    frac_gt1.append(float((d > 1.0).mean()))
    frac_gt10.append(float((d > 10.0).mean()))
    frac_gt25.append(float((d > 25.0).mean()))

    peak_arc_hist.append(float(ta.max()))
    peak_gx_hist.append(float(tg.max()))

    mat = ga.cell_data['material']
    build_cells = np.where(mat == BUILD_PID)[0]
    if len(build_cells):
        sub = ga.extract_cells(build_cells)
        build_pt_ids = np.unique(sub.point_data['vtkOriginalPointIds'])
        mean_build_arc_hist.append(float(ta[build_pt_ids].mean()))
        mean_build_gx_hist.append(float(tg[build_pt_ids].mean()))
    else:
        mean_build_arc_hist.append(float('nan'))
        mean_build_gx_hist.append(float('nan'))

    hot_a, hot_g = int(np.argmax(ta)), int(np.argmax(tg))
    hottest_history.append((t, tuple(ga.points[hot_a]), float(ta[hot_a]), tuple(gg.points[hot_g]), float(tg[hot_g])))

max_diff_hist = np.array(max_diff_hist)
mean_diff_hist = np.array(mean_diff_hist)
median_diff_hist = np.array(median_diff_hist)
std_diff_hist = np.array(std_diff_hist)
peak_arc_hist = np.array(peak_arc_hist)
peak_gx_hist = np.array(peak_gx_hist)
frac_gt1, frac_gt10, frac_gt25 = np.array(frac_gt1), np.array(frac_gt10), np.array(frac_gt25)

t_final = common_times[-1]
ga_final, gg_final = arc_frames[t_final], gx_frames[t_final]
final_diff = np.abs(ga_final.point_data['temp'] - gg_final.point_data['temp'])

# ---- where do the largest differences occur, relative to melt pool / deposition path? ----
worst_idx = np.argsort(final_diff)[-50:]  # 50 highest-difference nodes at final frame
worst_pts = ga_final.points[worst_idx]
hot_a_idx = int(np.argmax(ga_final.point_data['temp']))
melt_pool_pt = ga_final.points[hot_a_idx]
dist_to_melt_pool = np.linalg.norm(worst_pts - melt_pool_pt, axis=1)

# ---- seam cross-check (spatial, via coordinates - see module docstring) ----
seam_report_lines = []
if args.seam_csv and os.path.exists(args.seam_csv):
    import csv as csvmod
    rows = list(csvmod.DictReader(open(args.seam_csv)))
    all_s = np.array([float(r['s_arc_length']) for r in rows])
    all_xyz = np.array([[float(r['centroid_x']), float(r['centroid_y']), float(r['centroid_z'])] for r in rows])
    all_eid = np.array([int(r['element_id']) for r in rows])
    # rebuild the same neighbor-jump seam detector used during the original diagnosis
    diag_vtu_path = os.path.join(os.path.dirname(args.seam_csv), 'composition_diagnostics.vtu')
    diag = pv.read(diag_vtu_path)
    s_diag = diag.cell_data['path_coordinate_s']
    max_jump = np.zeros(diag.n_cells)
    for i in range(diag.n_cells):
        nb = diag.cell_neighbors(i, connections='points')
        if len(nb):
            max_jump[i] = np.abs(s_diag[nb] - s_diag[i]).max()
    seam_mask = max_jump > args.seam_jump_threshold
    seam_xyz = diag.cell_centers().points[seam_mask]
    n_seam = seam_mask.sum()

    # nearest seam-element distance for each of the 50 worst-difference nodes
    from scipy.spatial import cKDTree
    tree = cKDTree(seam_xyz)
    dist_to_seam, _ = tree.query(worst_pts)
    close_to_seam = (dist_to_seam < 1.0).sum()  # within ~1 element edge length

    seam_report_lines.append('Repeated-pass seam cross-check:')
    seam_report_lines.append('  {} seam elements identified (s-jump-to-neighbor > {})'.format(
        n_seam, args.seam_jump_threshold))
    seam_report_lines.append('  of the 50 highest final-frame-difference nodes:')
    seam_report_lines.append('    within 1.0 units of a seam element: {} ({:.0f}%)'.format(
        close_to_seam, 100 * close_to_seam / len(worst_pts)))
    seam_report_lines.append('    median distance to nearest seam element: {:.3f} units'.format(
        float(np.median(dist_to_seam))))
    seam_report_lines.append('    median distance to melt pool (hottest node): {:.3f} units'.format(
        float(np.median(dist_to_melt_pool))))
    seam_verdict = ('YES - largest differences are concentrated near seam elements' if close_to_seam > len(worst_pts) * 0.5
                     else 'NO - largest differences are NOT primarily at seam elements (likely melt-pool-driven instead)')
    seam_report_lines.append('  seam artifact creates misleading thermal differences? {}'.format(seam_verdict))
else:
    seam_report_lines.append('(seam cross-check skipped - no --seam-csv provided or file not found)')

# ---- report ----
lines = []
lines.append('15%-depth composition-mode thermal comparison: toolpath_arc_length vs global_x (part005)')
lines.append('=' * 70)
lines.append('matched frames: {}'.format(len(common_times)))
lines.append('')
lines.append('Temperature difference over time (all matched frames):')
lines.append('  max:    min={:.4f}  max={:.4f}  (at t={:.3f})'.format(
    max_diff_hist.min(), max_diff_hist.max(), times[np.argmax(max_diff_hist)]))
lines.append('  mean:   min={:.4f}  max={:.4f}'.format(mean_diff_hist.min(), mean_diff_hist.max()))
lines.append('  median: min={:.4f}  max={:.4f}'.format(median_diff_hist.min(), median_diff_hist.max()))
lines.append('  std:    min={:.4f}  max={:.4f}'.format(std_diff_hist.min(), std_diff_hist.max()))
lines.append('')
lines.append('Peak temperature history:')
lines.append('  arc_length: max={:.2f}K (final={:.2f}K)'.format(peak_arc_hist.max(), peak_arc_hist[-1]))
lines.append('  global_x:   max={:.2f}K (final={:.2f}K)'.format(peak_gx_hist.max(), peak_gx_hist[-1]))
lines.append('  max peak-history difference: {:.4f}K'.format(float(np.abs(peak_arc_hist - peak_gx_hist).max())))
lines.append('')
lines.append('Mean active-build-region temperature (final valid frame):')
valid = ~np.isnan(mean_build_arc_hist)
if valid.any():
    lines.append('  arc_length: {:.2f}K   global_x: {:.2f}K'.format(
        np.array(mean_build_arc_hist)[valid][-1], np.array(mean_build_gx_hist)[valid][-1]))
    lines.append('  fraction of frames with active build region: {:.0f}%'.format(100 * valid.sum() / len(valid)))
lines.append('')
lines.append('Hottest-node location (first, 25%, 50%, 75%, last matched frame):')
for idx in (0, len(hottest_history)//4, len(hottest_history)//2, 3*len(hottest_history)//4, -1):
    t, pa, va, pg, vg = hottest_history[idx]
    lines.append('  t={:.3f}s  arc_length: xyz=({:.2f},{:.2f},{:.2f}) T={:.1f}K   '
                 'global_x: xyz=({:.2f},{:.2f},{:.2f}) T={:.1f}K'.format(t, *pa, va, *pg, vg))
lines.append('')
lines.append('Final-frame (t={:.3f}s) difference thresholds:'.format(t_final))
lines.append('  max={:.4f}  mean={:.4f}  median={:.4f}  std={:.4f} K'.format(
    final_diff.max(), final_diff.mean(), float(np.median(final_diff)), float(final_diff.std())))
for thresh in (1.0, 10.0, 25.0):
    n = int((final_diff > thresh).sum())
    lines.append('  nodes > {:.0f}K: {} ({:.3f}%)'.format(thresh, n, 100 * n / len(final_diff)))
lines.append('')
lines.append('Largest-difference node locations relative to melt pool (final frame, top 50 worst nodes):')
lines.append('  median distance to hottest (melt-pool) node: {:.3f} units'.format(float(np.median(dist_to_melt_pool))))
lines.append('  min/max distance to melt pool: {:.3f} / {:.3f}'.format(dist_to_melt_pool.min(), dist_to_melt_pool.max()))
lines.append('')
lines.extend(seam_report_lines)

report = '\n'.join(lines)
print('\n' + report)
with open(os.path.join(args.out, 'composition_mode_comparison_15pct_report.txt'), 'w') as f:
    f.write(report + '\n')

# ---- comparison VTU (final matched frame) ----
comp_grid = ga_final.copy()
comp_grid.point_data['temperature_arc_length'] = ga_final.point_data['temp']
comp_grid.point_data['temperature_global_x'] = gg_final.point_data['temp']
comp_grid.point_data['temperature_difference'] = ga_final.point_data['temp'] - gg_final.point_data['temp']
comp_grid.cell_data['composition_arc_length'] = ga_final.cell_data['composition']
comp_grid.cell_data['composition_global_x'] = gg_final.cell_data['composition']
diag_vtu = os.path.join(os.path.dirname(args.arclength), 'composition_diagnostics.vtu')
s_field = np.full(comp_grid.n_cells, np.nan)
if os.path.exists(diag_vtu):
    diag = pv.read(diag_vtu)
    if comp_grid.n_cells == diag.n_cells:
        s_field = diag.cell_data['path_coordinate_s']
comp_grid.cell_data['path_coordinate_s'] = s_field
vtu_path = os.path.join(args.out, 'part005_composition_mode_comparison_15pct_final_frame.vtu')
comp_grid.save(vtu_path)
print('wrote {}'.format(vtu_path))

# ---- plots ----
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(times, peak_arc_hist, label='toolpath_arc_length', color='#c0392b')
ax.plot(times, peak_gx_hist, label='global_x', color='#2a78d6')
ax.set_xlabel('sim time (s)'); ax.set_ylabel('peak temperature (K)')
ax.set_title('Peak temperature vs. time (15% depth, part005)')
ax.legend(); ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
fig.tight_layout(); fig.savefig(os.path.join(args.out, 'peak_temperature_vs_time_15pct.png'), dpi=200); plt.close(fig)

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(times, mean_build_arc_hist, label='toolpath_arc_length', color='#c0392b')
ax.plot(times, mean_build_gx_hist, label='global_x', color='#2a78d6')
ax.set_xlabel('sim time (s)'); ax.set_ylabel('mean active-build-region temperature (K)')
ax.set_title('Mean active-build-region temperature vs. time (15% depth, part005)')
ax.legend(); ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
fig.tight_layout(); fig.savefig(os.path.join(args.out, 'mean_build_temperature_vs_time_15pct.png'), dpi=200); plt.close(fig)

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(times, max_diff_hist, color='#6a4fb0')
ax.set_xlabel('sim time (s)'); ax.set_ylabel('max |T_arc_length - T_global_x| (K)')
ax.set_title('Maximum temperature difference vs. time (15% depth, part005)')
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
fig.tight_layout(); fig.savefig(os.path.join(args.out, 'max_temperature_difference_vs_time_15pct.png'), dpi=200); plt.close(fig)

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.hist(final_diff, bins=60, color='#6a4fb0')
ax.set_yscale('log')
ax.set_xlabel('|T_arc_length - T_global_x| (K) at final matched frame')
ax.set_ylabel('node count (log scale)')
ax.set_title('Final-frame temperature-difference distribution (15% depth)')
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
fig.tight_layout(); fig.savefig(os.path.join(args.out, 'final_frame_difference_distribution_15pct.png'), dpi=200); plt.close(fig)

print('wrote plots to {}'.format(args.out))
