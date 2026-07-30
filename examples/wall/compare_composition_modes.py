# Compares two matched thermal runs that differ ONLY in their composition
# field (toolpath_arc_length vs global_x, same profile) - everything else
# (mesh, toolpath, birth curves, laser, output schedule) is identical by
# construction (verified separately, see the .k-file diff done before
# running). Matches frames by TIME field, same pattern as
# compare_endpoint_validation.py.
#
# Reports the numeric comparison requested, exports a comparison-ready VTU
# for the final matched frame, and produces two lightweight PNG plots
# (peak-temp-vs-time for both runs, mean build-region-temp-vs-time).

import argparse
import glob
import os

import numpy as np
import pyvista as pv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--arclength', required=True, help='results folder for the toolpath_arc_length run')
parser.add_argument('--globalx', required=True, help='results folder for the global_x run')
parser.add_argument('--out', required=True, help='output folder for the comparison VTU/plots/report')
args = parser.parse_args()

os.makedirs(args.out, exist_ok=True)


def load_frames(folder):
    frames = sorted(glob.glob(os.path.join(folder, 'wall_*.vtu')))
    if not frames:
        raise SystemExit('no wall_*.vtu frames found in {}'.format(folder))
    by_time = {}
    for f in frames:
        g = pv.read(f)
        t = round(float(g.field_data['TIME'][0]), 6)
        by_time[t] = g
    return by_time


arc_frames = load_frames(args.arclength)
gx_frames = load_frames(args.globalx)
common_times = sorted(set(arc_frames) & set(gx_frames))
if not common_times:
    raise SystemExit('no matching TIME values - runs were not produced with matching --stop-fraction/--n-frames')
print('matched {} frames by TIME'.format(len(common_times)))

BUILD_PID = 1  # auto-detected build-region matID, confirmed for part002 by prepare_composition.py's own printout

max_abs_diff = -1.0
max_abs_diff_time = None
sum_abs_diff, n_vals = 0.0, 0
peak_arc_hist, peak_gx_hist = [], []
mean_build_arc_hist, mean_build_gx_hist = [], []
hottest_history = []  # (time, arc_node_xyz, arc_temp, gx_node_xyz, gx_temp)

for t in common_times:
    ga = arc_frames[t]
    gg = gx_frames[t]
    if ga.n_points != gg.n_points:
        raise SystemExit('point count mismatch at t={}: arc={} globalx={}'.format(t, ga.n_points, gg.n_points))

    ta = ga.point_data['temp']
    tg = gg.point_data['temp']
    d = np.abs(ta - tg)
    if d.max() > max_abs_diff:
        max_abs_diff = float(d.max())
        max_abs_diff_time = t
    sum_abs_diff += float(d.sum())
    n_vals += d.size

    peak_arc_hist.append(float(ta.max()))
    peak_gx_hist.append(float(tg.max()))

    # mean build-region temperature: nodes touched by any build-region (material==BUILD_PID) cell
    mat = ga.cell_data['material']
    build_cells = np.where(mat == BUILD_PID)[0]
    if len(build_cells):
        build_pt_ids = np.unique(ga.extract_cells(build_cells).point_data['vtkOriginalPointIds'])
        mean_build_arc_hist.append(float(ta[build_pt_ids].mean()))
        mean_build_gx_hist.append(float(tg[build_pt_ids].mean()))
    else:
        mean_build_arc_hist.append(float('nan'))
        mean_build_gx_hist.append(float('nan'))

    hot_a = int(np.argmax(ta))
    hot_g = int(np.argmax(tg))
    hottest_history.append((t, tuple(ga.points[hot_a]), float(ta[hot_a]), tuple(gg.points[hot_g]), float(tg[hot_g])))

mean_abs_diff = sum_abs_diff / n_vals
peak_arc_hist = np.array(peak_arc_hist)
peak_gx_hist = np.array(peak_gx_hist)

# ---- final-frame statistics ----
t_final = common_times[-1]
ga_final = arc_frames[t_final]
gg_final = gx_frames[t_final]
final_diff = np.abs(ga_final.point_data['temp'] - gg_final.point_data['temp'])

# ---- report ----
lines = []
lines.append('Composition-mode thermal comparison: toolpath_arc_length vs global_x (both sinusoidal, part002)')
lines.append('=' * 70)
lines.append('matched frames: {}'.format(len(common_times)))
lines.append('')
lines.append('Whole-field temperature difference (all matched frames):')
lines.append('  max absolute difference:  {:.4f} K  (at t={})'.format(max_abs_diff, max_abs_diff_time))
lines.append('  mean absolute difference: {:.4f} K'.format(mean_abs_diff))
lines.append('')
lines.append('Peak temperature history:')
lines.append('  arc_length: min={:.2f} max={:.2f} K (final={:.2f} K)'.format(
    peak_arc_hist.min(), peak_arc_hist.max(), peak_arc_hist[-1]))
lines.append('  global_x:   min={:.2f} max={:.2f} K (final={:.2f} K)'.format(
    peak_gx_hist.min(), peak_gx_hist.max(), peak_gx_hist[-1]))
lines.append('  max peak-history difference: {:.4f} K'.format(float(np.abs(peak_arc_hist - peak_gx_hist).max())))
lines.append('')
lines.append('Hottest-node location over time (first, middle, last matched frame):')
for idx in (0, len(hottest_history) // 2, -1):
    t, pa, va, pg, vg = hottest_history[idx]
    lines.append('  t={:.3f}s  arc_length hottest: xyz=({:.2f},{:.2f},{:.2f}) T={:.1f}K   '
                 'global_x hottest: xyz=({:.2f},{:.2f},{:.2f}) T={:.1f}K'.format(
                     t, pa[0], pa[1], pa[2], va, pg[0], pg[1], pg[2], vg))
lines.append('')
lines.append('Final-frame (t={:.3f}s) temperature difference distribution:'.format(t_final))
lines.append('  max={:.4f}  mean={:.4f}  median={:.4f}  std={:.4f} K'.format(
    final_diff.max(), final_diff.mean(), float(np.median(final_diff)), float(final_diff.std())))
lines.append('  fraction of nodes with diff > 1K:  {:.2f}%'.format(100 * (final_diff > 1.0).mean()))
lines.append('  fraction of nodes with diff > 10K: {:.2f}%'.format(100 * (final_diff > 10.0).mean()))
lines.append('')
visibly_different = max_abs_diff > 5.0  # a few K is well above solver/roundoff noise for this problem scale
lines.append('Visibly different thermal pattern? {} (max whole-field diff {:.2f}K)'.format(
    'YES' if visibly_different else 'NO - fields are thermally very similar', max_abs_diff))

report = '\n'.join(lines)
print('\n' + report)
with open(os.path.join(args.out, 'composition_mode_comparison_report.txt'), 'w') as f:
    f.write(report + '\n')

# ---- comparison VTU (final matched frame) ----
comp_grid = ga_final.copy()
comp_grid.point_data['temperature_arc_length'] = ga_final.point_data['temp']
comp_grid.point_data['temperature_global_x'] = gg_final.point_data['temp']
comp_grid.point_data['temperature_difference'] = ga_final.point_data['temp'] - gg_final.point_data['temp']
comp_grid.cell_data['composition_arc_length'] = ga_final.cell_data['composition']
comp_grid.cell_data['composition_global_x'] = gg_final.cell_data['composition']
# path_coordinate_s: only meaningful on build-region cells; substrate cells get NaN
s_field = np.full(comp_grid.n_cells, np.nan)
mat = ga_final.cell_data['material']
build_mask = (mat == BUILD_PID)
# composition_arc_length here is the TI64 fraction written by prepare_composition.py;
# recover s isn't stored per-cell in the sim output, so approximate via the diagnostic
# CSV/VTU generated earlier for the same mesh (same element ordering) if available -
# otherwise leave as NaN (composition fields above are the authoritative comparison).
diag_vtu = os.path.join(os.path.dirname(args.arclength), 'part002_composition_diagnostics.vtu')
if os.path.exists(diag_vtu) and comp_grid.n_cells == pv.read(diag_vtu).n_cells:
    s_field = pv.read(diag_vtu).cell_data['s_arc_length']
comp_grid.cell_data['path_coordinate_s'] = s_field
comp_vtu_path = os.path.join(args.out, 'part002_composition_mode_comparison_final_frame.vtu')
comp_grid.save(comp_vtu_path)
print('wrote {}'.format(comp_vtu_path))

# ---- plots ----
times = np.array(common_times)

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(times, peak_arc_hist, label='toolpath_arc_length', color='#c0392b')
ax.plot(times, peak_gx_hist, label='global_x', color='#2a78d6')
ax.set_xlabel('sim time (s)')
ax.set_ylabel('peak temperature (K)')
ax.set_title('Peak temperature vs. time (part002, sinusoidal composition)')
ax.legend()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join(args.out, 'peak_temperature_vs_time.png'), dpi=200)
plt.close(fig)

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(times, mean_build_arc_hist, label='toolpath_arc_length', color='#c0392b')
ax.plot(times, mean_build_gx_hist, label='global_x', color='#2a78d6')
ax.set_xlabel('sim time (s)')
ax.set_ylabel('mean build-region temperature (K)')
ax.set_title('Mean build-region temperature vs. time (part002)')
ax.legend()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join(args.out, 'mean_build_temperature_vs_time.png'), dpi=200)
plt.close(fig)

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.hist(final_diff, bins=60, color='#6a4fb0')
ax.set_xlabel('|T_arc_length - T_global_x| (K) at final matched frame')
ax.set_ylabel('node count')
ax.set_title('Final-frame temperature-difference distribution')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join(args.out, 'final_frame_difference_distribution.png'), dpi=200)
plt.close(fig)

print('wrote plots to {}'.format(args.out))
