# Aggregates a completed power (or other single-parameter) sweep across
# multiple sweep values, for both composition modes at each value, into one
# scaling-behavior report + plot set. Reuses the same matched-by-TIME
# per-frame comparison approach as compare_composition_modes_part004.py, but
# loops over sweep values instead of hardcoding one dataset/depth.
#
# Answers the linear-vs-nonlinear-vs-threshold question (5 points is the
# smallest meaningful sweep - enough to see curvature/a knee, not enough for
# a rigorous statistical change-point test, so this reports linear/quadratic
# R^2 and a simple max-local-slope-ratio flag rather than pretending to more
# precision than 5 points support).

import argparse
import glob
import os

import numpy as np
import pyvista as pv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--results-dir', required=True, help='dir containing results_<stem>_LP<value>/ subfolders')
parser.add_argument('--values', required=True, nargs='+', type=float, help='sweep values, e.g. 400 600 800 1000 1200')
parser.add_argument('--arc-stem', required=True, help='deck stem for the arc-length mode, e.g. 4_graded_arclength_sinusoidal')
parser.add_argument('--gx-stem', required=True, help='deck stem for the global_x mode, e.g. 4_graded_globalx_sinusoidal')
parser.add_argument('--liquidus-k', type=float, default=1609.0,
                     help='melt-pool proxy threshold, K (default: IN718 liquidus)')
parser.add_argument('--out-dir', required=True)
parser.add_argument('--param-name', default='Laser Power (W)')
args = parser.parse_args()
os.makedirs(args.out_dir, exist_ok=True)


def load_frames(folder):
    frames = sorted(glob.glob(os.path.join(folder, 'wall_*.vtu')))
    if not frames:
        raise SystemExit('no frames in {}'.format(folder))
    by_time = {}
    for f in frames:
        g = pv.read(f)
        by_time[round(float(g.field_data['TIME'][0]), 6)] = g
    return by_time


BUILD_PID = 1


def _mean_build_temp(grid):
    """Mean temperature restricted to build-region material cells (matches
    compare_composition_modes_part004.py's convention) - NOT a whole-domain
    mean, which would be diluted by not-yet-relevant substrate/ambient nodes."""
    mat = grid.cell_data['material']
    build_cells = np.where(mat == BUILD_PID)[0]
    if not len(build_cells):
        return float('nan')
    sub = grid.extract_cells(build_cells)
    build_pt_ids = np.unique(sub.point_data['vtkOriginalPointIds'])
    return float(grid.point_data['temp'][build_pt_ids].mean())


rows = []
for v in args.values:
    arc_dir = os.path.join(args.results_dir, 'results_{}_LP{:g}'.format(args.arc_stem, v))
    gx_dir = os.path.join(args.results_dir, 'results_{}_LP{:g}'.format(args.gx_stem, v))
    arc_frames = load_frames(arc_dir)
    gx_frames = load_frames(gx_dir)
    common_times = sorted(set(arc_frames) & set(gx_frames))
    if not common_times:
        raise SystemExit('no matching TIME values for sweep value {}'.format(v))

    max_diff_hist, peak_arc_hist, peak_gx_hist = [], [], []
    frac_liquidus_arc, frac_liquidus_gx = [], []
    for t in common_times:
        ga, gg = arc_frames[t], gx_frames[t]
        ta, tg = ga.point_data['temp'], gg.point_data['temp']
        max_diff_hist.append(float(np.abs(ta - tg).max()))
        peak_arc_hist.append(float(ta.max()))
        peak_gx_hist.append(float(tg.max()))
        frac_liquidus_arc.append(float((ta > args.liquidus_k).mean()))
        frac_liquidus_gx.append(float((tg > args.liquidus_k).mean()))
    max_diff_hist = np.array(max_diff_hist)

    t_final = common_times[-1]
    final_diff = np.abs(arc_frames[t_final].point_data['temp'] - gx_frames[t_final].point_data['temp'])

    rows.append(dict(
        value=v,
        peak_temp_arc=max(peak_arc_hist), peak_temp_gx=max(peak_gx_hist),
        transient_max_diff=float(max_diff_hist.max()),
        transient_max_diff_time=common_times[int(np.argmax(max_diff_hist))],
        final_max_diff=float(final_diff.max()), final_mean_diff=float(final_diff.mean()),
        mean_build_temp_arc=_mean_build_temp(arc_frames[t_final]),
        mean_build_temp_gx=_mean_build_temp(gx_frames[t_final]),
        frac_above_liquidus_arc=max(frac_liquidus_arc), frac_above_liquidus_gx=max(frac_liquidus_gx),
        n_frames=len(common_times),
    ))
    print('value={:g}: peak_arc={:.1f}K peak_gx={:.1f}K transient_max_diff={:.3f}K final_max_diff={:.4f}K '
          'frac>liquidus(arc)={:.3f}'.format(v, rows[-1]['peak_temp_arc'], rows[-1]['peak_temp_gx'],
                                              rows[-1]['transient_max_diff'], rows[-1]['final_max_diff'],
                                              rows[-1]['frac_above_liquidus_arc']))

values = np.array([r['value'] for r in rows])
transient = np.array([r['transient_max_diff'] for r in rows])
final_max = np.array([r['final_max_diff'] for r in rows])
peak_arc = np.array([r['peak_temp_arc'] for r in rows])
peak_gx = np.array([r['peak_temp_gx'] for r in rows])
mean_build_arc = np.array([r['mean_build_temp_arc'] for r in rows])
mean_build_gx = np.array([r['mean_build_temp_gx'] for r in rows])
frac_liq = np.array([r['frac_above_liquidus_arc'] for r in rows])

# relative sensitivity: transient composition-mode diff normalized by the
# peak temperature RISE above ambient (300K) at that same sweep value -
# controls for the trivial fact that absolute diffs grow when everything
# gets hotter, isolating whether composition choice becomes DISPROPORTION-
# ATELY more important (e.g. near a phase-change threshold) rather than
# just scaling in step with the overall thermal magnitude.
peak_rise = np.maximum(peak_arc - 300.0, 1e-9)
relative_sensitivity = transient / peak_rise
for r, rs in zip(rows, relative_sensitivity):
    r['relative_sensitivity'] = float(rs)

# ---- CSV ----
csv_path = os.path.join(args.out_dir, 'parameter_sweep_summary.csv')
with open(csv_path, 'w') as f:
    keys = list(rows[0].keys())
    f.write(','.join(keys) + '\n')
    for r in rows:
        f.write(','.join(str(r[k]) for k in keys) + '\n')
print('\nwrote {}'.format(csv_path))


def r_squared(x, y, deg):
    coeffs = np.polyfit(x, y, deg)
    fit = np.polyval(coeffs, x)
    ss_res = np.sum((y - fit) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return 1 - ss_res / ss_tot if ss_tot > 0 else float('nan'), coeffs


lin_r2, lin_coef = r_squared(values, transient, 1)
quad_r2, quad_coef = r_squared(values, transient, 2)
slopes = np.diff(transient) / np.diff(values)
mean_slope = np.mean(slopes)
max_slope_ratio = float(np.max(np.abs(slopes)) / np.mean(np.abs(slopes))) if mean_slope != 0 else float('nan')
knee_idx = int(np.argmax(np.abs(slopes)))

lines = []
lines.append('Parameter sweep scaling analysis ({} values: {})'.format(len(values), list(values)))
lines.append('=' * 70)
lines.append('')
lines.append('Transient max |dT| vs {}:'.format(args.param_name))
lines.append('  values: {}'.format(dict(zip(values.tolist(), transient.tolist()))))
lines.append('  linear fit R^2  = {:.4f}  (slope={:.5f}, intercept={:.3f})'.format(lin_r2, *lin_coef))
lines.append('  quadratic fit R^2 = {:.4f}'.format(quad_r2))
lines.append('  per-interval slopes: {}'.format(np.round(slopes, 5).tolist()))
lines.append('  max/mean interval-slope ratio: {:.2f} (interval [{:g},{:g}])'.format(
    max_slope_ratio, values[knee_idx], values[knee_idx + 1]))
verdict = ('LINEAR (quadratic fit does not meaningfully improve R^2)' if quad_r2 - lin_r2 < 0.05
           else 'NONLINEAR (quadratic fit substantially better than linear)')
if max_slope_ratio > 2.5:
    verdict += '; POSSIBLE THRESHOLD/KNEE near {}={:g}->{:g}'.format(args.param_name, values[knee_idx], values[knee_idx + 1])
lines.append('  verdict: {}'.format(verdict))
lines.append('')
lines.append('')
lines.append('Relative composition sensitivity (transient max |dT| / peak temperature rise above 300K):')
lines.append('  {}'.format(dict(zip(values.tolist(), np.round(relative_sensitivity, 5).tolist()))))
lines.append('  (flat/decreasing => composition sensitivity just tracks overall thermal magnitude;')
lines.append('   increasing => composition choice becomes disproportionately more important at higher {},'.format(args.param_name))
lines.append('   e.g. because the two alloys phase-change at different temperatures)')
lines.append('')
lines.append('Melt-pool proxy (fraction of nodes > {:.0f}K, i.e. above liquidus):'.format(args.liquidus_k))
lines.append('  {}'.format(dict(zip(values.tolist(), np.round(frac_liq, 4).tolist()))))
first_melt_idx = np.argmax(frac_liq > 0) if (frac_liq > 0).any() else None
if first_melt_idx is not None and frac_liq[0] == 0:
    lines.append('  first sweep value with any melting: {}={:g}'.format(args.param_name, values[first_melt_idx]))
else:
    lines.append('  melting present at all sweep values, or absent at all sweep values')

report = '\n'.join(lines)
print('\n' + report)
with open(os.path.join(args.out_dir, 'scaling_analysis_report.txt'), 'w') as f:
    f.write(report + '\n')

# ---- plots (generated for every sweep, per item 6 of the study design) ----
plt.rcParams['font.size'] = 12


def lineplot(x, ys, labels, colors, ylabel, title, fname, logy=False):
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for y, label, color in zip(ys, labels, colors):
        ax.plot(x, y, 'o-', label=label, color=color, linewidth=2, markersize=7)
    ax.set_xlabel(args.param_name)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if logy:
        ax.set_yscale('log')
    if len(labels) > 1:
        ax.legend()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', color='#e5e5e5', linewidth=0.8)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, fname), dpi=200)
    plt.close(fig)


lineplot(values, [peak_arc, peak_gx], ['toolpath_arc_length', 'global_x'], ['#c0392b', '#2a78d6'],
          'Peak Temperature (K)', 'Peak temperature vs. {}'.format(args.param_name), 'peak_temperature_vs_parameter.png')
lineplot(values, [transient], ['transient max |dT|'], ['#6a4fb0'],
          'Max |T_arc - T_gx| (K)', 'Composition-mode transient max difference vs. {}'.format(args.param_name),
          'transient_max_diff_vs_parameter.png')
lineplot(values, [final_max], ['final-frame max |dT|'], ['#6a4fb0'],
          'Max |T_arc - T_gx| (K)', 'Composition-mode final-frame max difference vs. {}'.format(args.param_name),
          'final_max_diff_vs_parameter.png')
lineplot(values, [mean_build_arc, mean_build_gx], ['toolpath_arc_length', 'global_x'], ['#c0392b', '#2a78d6'],
          'Mean Temperature (K)', 'Mean whole-domain temperature vs. {}'.format(args.param_name),
          'mean_build_temperature_vs_parameter.png')
lineplot(values, [frac_liq * 100], ['fraction > liquidus'], ['#e08214'],
          '% nodes above liquidus', 'Melt-pool extent proxy vs. {}'.format(args.param_name),
          'melt_pool_proxy_vs_parameter.png')
lineplot(values, [relative_sensitivity], ['transient max |dT| / peak temperature rise'], ['#1a9850'],
          'Relative composition sensitivity (K/K)',
          'Relative composition-mode sensitivity vs. {}'.format(args.param_name),
          'relative_sensitivity_vs_parameter.png')

print('\nwrote 6 plots + CSV + report to {}'.format(args.out_dir))
