# Endpoint validation for the graded-material solver path: compares a pure-
# material run (legacy *MAT_THERMAL_ISOTROPIC_TD path) against the graded
# path (*MAT_THERMAL_GRADED_TD/*ELEMENT_COMPOSITION) forced to a constant
# composition at that same endpoint. If the graded blend math is correct,
# these should agree to within floating-point precision - composition=[1,0]
# reduces linear_mix's sum(f[...,k]*values[k]) to exactly values[0], so any
# real disagreement beyond numerical noise means something is actually wrong
# with the blend, not just "close enough".
#
# Matches frames by their embedded TIME field (not file index), so this only
# works correctly if both runs were produced with the same --stop-fraction/
# --n-frames against the same endtime - run_wall.py's save schedule is
# deterministic given those, so this holds as long as both runs used
# identical flags.

import argparse
import glob
import os
import sys

import numpy as np
import pyvista as pv

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--pure', required=True, help='results folder for the pure-material run')
parser.add_argument('--constant', required=True, help='results folder for the constant-composition graded run')
parser.add_argument('--label', default='', help='short label for this pair, e.g. "inconel"')
parser.add_argument('--tolerance-k', type=float, default=1e-3,
                     help='max-abs-diff pass threshold in Kelvin (default 1e-3 - this should be near machine '
                          'precision if the blend math is correct; do not loosen this to force a pass)')
parser.add_argument('--out', default=None, help='report file to write (default: printed to stdout only)')
args = parser.parse_args()


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


pure_frames = load_frames(args.pure)
const_frames = load_frames(args.constant)

common_times = sorted(set(pure_frames) & set(const_frames))
missing_pure = sorted(set(const_frames) - set(pure_frames))
missing_const = sorted(set(pure_frames) - set(const_frames))
if not common_times:
    raise SystemExit('no matching TIME values between {} and {} - runs were not produced with matching '
                      '--stop-fraction/--n-frames'.format(args.pure, args.constant))

max_abs_diff = -1.0
max_abs_diff_time = None
sum_abs_diff = 0.0
n_vals = 0
peak_pure_hist = []
peak_const_hist = []

for t in common_times:
    gp = pure_frames[t]
    gc = const_frames[t]
    if gp.n_points != gc.n_points:
        raise SystemExit('point count mismatch at t={}: pure={} constant={} - meshes are not the same run'.format(
            t, gp.n_points, gc.n_points))
    tp = gp.point_data['temp']
    tc = gc.point_data['temp']
    d = np.abs(tp - tc)
    if d.max() > max_abs_diff:
        max_abs_diff = float(d.max())
        max_abs_diff_time = t
    sum_abs_diff += float(d.sum())
    n_vals += d.size
    peak_pure_hist.append(float(tp.max()))
    peak_const_hist.append(float(tc.max()))

mean_abs_diff = sum_abs_diff / n_vals
peak_pure_hist = np.array(peak_pure_hist)
peak_const_hist = np.array(peak_const_hist)
peak_hist_diff = np.abs(peak_pure_hist - peak_const_hist)
max_peak_hist_diff = float(peak_hist_diff.max())
mean_peak_hist_diff = float(peak_hist_diff.mean())

passed = max_abs_diff <= args.tolerance_k

lines = []
lines.append('Endpoint validation: {}'.format(args.label or '{} vs {}'.format(args.pure, args.constant)))
lines.append('  pure run:     {}'.format(args.pure))
lines.append('  constant run: {}'.format(args.constant))
lines.append('  matched frames: {} (by TIME field)'.format(len(common_times)))
if missing_pure or missing_const:
    lines.append('  WARNING: unmatched frames - pure-only: {} constant-only: {}'.format(
        len(missing_const), len(missing_pure)))
lines.append('  max absolute temperature difference:  {:.6e} K  (at t={})'.format(max_abs_diff, max_abs_diff_time))
lines.append('  mean absolute temperature difference: {:.6e} K'.format(mean_abs_diff))
lines.append('  max peak-temperature-history difference:  {:.6e} K'.format(max_peak_hist_diff))
lines.append('  mean peak-temperature-history difference: {:.6e} K'.format(mean_peak_hist_diff))
lines.append('  tolerance: {:.1e} K'.format(args.tolerance_k))
lines.append('  RESULT: {}'.format('PASS' if passed else 'FAIL'))

report = '\n'.join(lines)
print(report)
if args.out:
    with open(args.out, 'w') as f:
        f.write(report + '\n')
    print('\nsaved report to {}'.format(args.out))

sys.exit(0 if passed else 1)
