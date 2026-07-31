# Reusable synthetic boustrophedon (back-and-forth) raster-toolpath generator,
# built for the controlled hatch-spacing study. Writes a .crs file in the
# same 5-column [time, x, y, z, laser_state] format load_toolpath() expects.
#
# Design grounded directly in part004's REAL toolpath (not assumed):
#  - footprint: measured mesh/toolpath bounds, x width 17.92mm, y width 43.50mm
#  - part004's real toolpath already contains a genuine dense raster - 17
#    tracks per layer at EXACTLY 1.12mm spacing (all 16 consecutive gaps
#    measured at 1.12mm), 8 real deposition z-levels (1.0-8.0mm), each pass
#    resetting to the same starting corner - not the sparse "24 passes total"
#    raster an earlier (buggy) direction-reversal analysis mis-reported.
#  - every segment in the real toolpath - deposition tracks, inter-track
#    steps, and the diagonal layer-transition move - runs at the SAME
#    uniform speed (measured exactly 8.0 units/s throughout, including
#    laser-off moves) - this generator reproduces that convention exactly
#    rather than inventing a separate rapid-travel speed.
#
# Layer count is DERIVED from footprint height / layer height, not hardcoded,
# specifically so it can never silently exceed the real mesh's z-extent (a
# real risk: an earlier draft of this study assumed 16 layers, carried over
# from element z-sublayer count rather than real deposition-pass count, which
# would have made the toolpath 16mm tall against an 8mm-tall mesh).

import argparse
import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--x-min', type=float, required=True)
parser.add_argument('--x-max', type=float, required=True)
parser.add_argument('--y-min', type=float, required=True)
parser.add_argument('--y-max', type=float, required=True)
parser.add_argument('--z-max', type=float, required=True, help='total build height - must match the real mesh z-extent')
parser.add_argument('--layer-height', type=float, required=True)
parser.add_argument('--hatch', type=float, required=True, help='spacing between adjacent tracks (x-direction)')
parser.add_argument('--speed', type=float, required=True, help='uniform scan speed, units/s, applied to every segment')
parser.add_argument('--out', required=True)
args = parser.parse_args()

n_layers = args.z_max / args.layer_height
if abs(n_layers - round(n_layers)) > 1e-6:
    raise SystemExit('z_max={} is not an integer multiple of layer_height={} ({} layers)'.format(
        args.z_max, args.layer_height, n_layers))
n_layers = int(round(n_layers))

x_width = args.x_max - args.x_min
n_gaps = x_width / args.hatch
if abs(n_gaps - round(n_gaps)) > 1e-6:
    raise SystemExit('x_width={} is not an integer multiple of hatch={} ({} gaps) - '
                      'tracks would not land exactly on x_max'.format(x_width, args.hatch, n_gaps))
n_tracks = int(round(n_gaps)) + 1
track_x = args.x_min + np.arange(n_tracks) * args.hatch

rows = [(0.0, 0.0, 0.0, 0.0, 0)]  # origin sentinel, matches real part004 toolpath convention
t = 0.0


def move_to(x, y, z, state):
    global t
    x0, y0, z0 = rows[-1][1:4]
    dist = np.sqrt((x - x0) ** 2 + (y - y0) ** 2 + (z - z0) ** 2)
    t += dist / args.speed
    rows.append((t, x, y, z, state))


for layer in range(n_layers):
    z = (layer + 1) * args.layer_height
    # reset move to this layer's first track start (laser off) - matches
    # part004's real convention of restarting at the same corner every layer
    move_to(track_x[0], args.y_min, z, 0)
    for i, x in enumerate(track_x):
        y_end = args.y_max if i % 2 == 0 else args.y_min
        move_to(x, y_end, z, 1)  # deposition track, laser on
        if i < n_tracks - 1:
            move_to(track_x[i + 1], y_end, z, 0)  # inter-track step, laser off

with open(args.out, 'w') as f:
    for row in rows:
        f.write('{:20.8f}{:20.8f}{:20.8f}{:20.8f}{:2d}\n'.format(*row))

total_length = sum(np.sqrt((rows[i][1] - rows[i - 1][1]) ** 2 + (rows[i][2] - rows[i - 1][2]) ** 2 +
                           (rows[i][3] - rows[i - 1][3]) ** 2) for i in range(1, len(rows)))
print('wrote {} : {} waypoints, {} layers, {} tracks/layer, hatch={:g}mm, total length={:.4f}, ENDTIM={:.4f}'.format(
    args.out, len(rows), n_layers, n_tracks, args.hatch, total_length, rows[-1][0]))
