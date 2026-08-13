# Tests for composition_lib.py's toolpath-arc-length coordinate mode
# (build_toolpath_arc_length_table, project_to_toolpath, and
# coordinate_function(mode='toolpath_arc_length')). No pytest in this
# project (checked - not installed) - plain assert-and-report, matching
# test_composition_regression.py's convention. Run standalone; also run
# test_composition_regression.py after any change here, since this module
# is shared with the position-dependent wall-example generators.

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from composition_lib import (build_toolpath_arc_length_table, project_to_toolpath, coordinate_function,
                              filter_deposit_segments)

results = []


def check(label, condition, detail=''):
    results.append((label, bool(condition), detail))


def close(a, b, tol=1e-9):
    return abs(a - b) <= tol


# ---- 1. straight toolpath: arc-length s should match normalized global position ----
straight = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0], [4, 0, 0]], dtype=float)
arc = build_toolpath_arc_length_table(straight)
check('straight: arc table', np.allclose(arc, [0, 1, 2, 3, 4]), str(arc))

centroids = np.array([[0.5, 0, 0], [2.0, 0, 0], [3.5, 0, 0]])
s = coordinate_function(centroids, 'toolpath_arc_length', toolpath=straight)
expected = np.array([0.125, 0.5, 0.875])
check('straight: s matches normalized global position', np.allclose(s, expected), 'got {}'.format(s))

# a point exactly on the path should have essentially zero projection distance
arc_at, dist = project_to_toolpath(np.array([2.0, 0, 0]), straight, arc)
check('straight: on-path point has ~zero projection distance', close(dist, 0.0, 1e-9), 'dist={}'.format(dist))

# ---- 2. L-shaped toolpath with known expected projections ----
L = np.array([[0, 0, 0], [10, 0, 0], [10, 10, 0]], dtype=float)  # two 10-unit segments, total 20
arc_L = build_toolpath_arc_length_table(L)
check('L-shape: arc table', np.allclose(arc_L, [0, 10, 20]), str(arc_L))

# hand-computed: point (10,5,0) sits exactly on the vertical leg, s = (10+5)/20
s_vert = coordinate_function(np.array([10, 5, 0]), 'toolpath_arc_length', toolpath=L)
check('L-shape: on-vertical-leg point', close(s_vert, 0.75), 's={}'.format(s_vert))

# hand-computed: point (5,0,0) sits exactly on the horizontal leg, s = 5/20
s_horiz = coordinate_function(np.array([5, 0, 0]), 'toolpath_arc_length', toolpath=L)
check('L-shape: on-horizontal-leg point', close(s_horiz, 0.25), 's={}'.format(s_horiz))

# hand-computed tie: (9,1,0) is distance 1 from both segments (verified by hand);
# numpy argmin breaks ties toward the first (horizontal) segment deterministically
s_corner = coordinate_function(np.array([9, 1, 0]), 'toolpath_arc_length', toolpath=L)
check('L-shape: near-corner tie resolves deterministically', close(s_corner, 0.45), 's={}'.format(s_corner))

# ---- 3. curved / multi-segment synthetic path (quarter circle, 100 segments) ----
theta = np.linspace(0, np.pi / 2, 101)
circle = np.stack([10 * np.cos(theta), 10 * np.sin(theta), np.zeros_like(theta)], axis=1)
arc_c = build_toolpath_arc_length_table(circle)
true_len = 10 * np.pi / 2
check('curved path: total arc length matches analytic quarter-circle length',
      close(arc_c[-1], true_len, tol=1e-3), 'got {:.6f}, expected {:.6f}'.format(arc_c[-1], true_len))

# centroids placed exactly ON the curve at 10 evenly-spaced angles should
# produce a MONOTONICALLY INCREASING, evenly-spaced s (this is the actual
# "not stair-stepped" property - a fine-grained curve shouldn't quantize)
probe_theta = np.linspace(0.02, np.pi / 2 - 0.02, 10)
probes = np.stack([10 * np.cos(probe_theta), 10 * np.sin(probe_theta), np.zeros_like(probe_theta)], axis=1)
s_probes = coordinate_function(probes, 'toolpath_arc_length', toolpath=circle)
check('curved path: s is monotonically increasing along the curve',
      bool(np.all(np.diff(s_probes) > 0)), 's={}'.format(s_probes))
check('curved path: s spacing is roughly even (not stair-stepped)',
      np.std(np.diff(s_probes)) < 0.02, 'diffs={}'.format(np.diff(s_probes)))

# ---- 4. points near segment boundaries ----
# a point placed exactly at a shared vertex between two straight segments
boundary = np.array([[0, 0, 0], [5, 0, 0], [10, 0, 0]], dtype=float)  # collinear, vertex at (5,0,0)
arc_b = build_toolpath_arc_length_table(boundary)
s_at_vertex = coordinate_function(np.array([5, 0, 0]), 'toolpath_arc_length', toolpath=boundary)
check('segment boundary: point exactly at shared vertex', close(s_at_vertex, 0.5), 's={}'.format(s_at_vertex))
# points just before/after the boundary should be continuous, not jump
s_before = coordinate_function(np.array([4.999, 0, 0]), 'toolpath_arc_length', toolpath=boundary)
s_after = coordinate_function(np.array([5.001, 0, 0]), 'toolpath_arc_length', toolpath=boundary)
check('segment boundary: continuous across the boundary (no jump)',
      abs(s_after - s_before) < 1e-3, 'before={} after={}'.format(s_before, s_after))

# ---- 5. repeated / zero-length waypoints ----
dup = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0], [5, 0, 0]], dtype=float)
arc_dup = build_toolpath_arc_length_table(dup)
check('repeated waypoints: arc table has no NaN/inf', np.all(np.isfinite(arc_dup)), str(arc_dup))
check('repeated waypoints: arc table stays index-aligned (len matches input)',
      len(arc_dup) == len(dup), 'len={}'.format(len(arc_dup)))
s_dup = coordinate_function(np.array([2.5, 0, 0]), 'toolpath_arc_length', toolpath=dup)
check('repeated waypoints: still projects correctly onto the real segment', close(s_dup, 0.5), 's={}'.format(s_dup))

# all-identical toolpath (zero total length) - must not crash or produce NaN
allzero = np.array([[3, 3, 3], [3, 3, 3], [3, 3, 3]], dtype=float)
s_zero = coordinate_function(np.array([[3, 3, 3], [100, 100, 100]]), 'toolpath_arc_length', toolpath=allzero)
check('zero-length toolpath: no crash, no NaN', np.all(np.isfinite(s_zero)), 's={}'.format(s_zero))
check('zero-length toolpath: falls back to s=0 everywhere', np.allclose(s_zero, 0.0), 's={}'.format(s_zero))

# ---- 6. explicit edge-case errors (empty / one-point toolpaths) ----
try:
    build_toolpath_arc_length_table(np.zeros((0, 3)))
    check('empty toolpath raises ValueError', False, 'did not raise')
except ValueError:
    check('empty toolpath raises ValueError', True)

try:
    coordinate_function(np.array([1, 1, 1]), 'toolpath_arc_length', toolpath=np.array([[0, 0, 0]]))
    check('one-point toolpath raises ValueError', False, 'did not raise')
except ValueError:
    check('one-point toolpath raises ValueError', True)

# ---- 7. before-first / after-last clamping ----
s_before_all = coordinate_function(np.array([-100, 0, 0]), 'toolpath_arc_length', toolpath=straight)
s_after_all = coordinate_function(np.array([100, 0, 0]), 'toolpath_arc_length', toolpath=straight)
check('point far before first waypoint clamps to s=0', close(s_before_all, 0.0), 's={}'.format(s_before_all))
check('point far after last waypoint clamps to s=1', close(s_after_all, 1.0), 's={}'.format(s_after_all))

# ---- 8. outputs always within [0,1] - fuzz test with random far-flung points ----
rng = np.random.default_rng(0)
random_pts = rng.uniform(-1000, 1000, size=(500, 3))
s_random = coordinate_function(random_pts, 'toolpath_arc_length', toolpath=circle)
check('fuzz test (500 random points): all s in [0,1]',
      bool(np.all((s_random >= 0.0) & (s_random <= 1.0))),
      'min={} max={}'.format(s_random.min(), s_random.max()))
check('fuzz test: no NaN/inf', bool(np.all(np.isfinite(s_random))))

# ---- 9. chunking doesn't change results (chunk_size=1 vs default) ----
s_default = coordinate_function(probes, 'toolpath_arc_length', toolpath=circle, chunk_size=2000)
s_tiny_chunks = coordinate_function(probes, 'toolpath_arc_length', toolpath=circle, chunk_size=1)
check('chunk_size does not affect results', np.allclose(s_default, s_tiny_chunks),
      'default={} tiny={}'.format(s_default, s_tiny_chunks))

# ---- 10. raster/self-folding path: travel segments must not win the search ----
# Reproduces examples/wall's own toolpath.crs structure: 3 deposit layers
# (state=1) 0.2 apart in Z, connected by travel moves (state=0), PLUS an
# end-of-build "return to origin" travel move that spans the entire Z range
# in one segment sitting right at the raster's edge (x=7) - this is exactly
# the segment that hijacked project_to_toolpath's nearest-segment search
# before filter_deposit_segments existed (see that function's docstring).
raster = np.array([
    [-7, 0, 0.0],                  # 0: start position
    [-7, 0, 0.2], [7, 0, 0.2],     # 1-2: travel up, then deposit layer 0 (state 0,1)
    [7, 0, 0.4], [-7, 0, 0.4],     # 3-4: travel up, then deposit layer 1 (state 0,1)
    [-7, 0, 0.6], [7, 0, 0.6],     # 5-6: travel up, then deposit layer 2 (state 0,1)
    [7, 0, 0.0],                   # 7: end-of-build return to origin (state 0) -
    #    a single travel segment spanning the ENTIRE Z range (0.6 -> 0.0),
    #    sitting exactly at x=7 where the last layer's pass also ends.
], dtype=float)
raster_state = np.array([0, 0, 1, 0, 1, 0, 1, 0])

filtered = filter_deposit_segments(raster, raster_state)
check('raster: filtered toolpath drops the 2 non-deposit-adjacent points',
      len(filtered) == 6, 'got {} points: {}'.format(len(filtered), filtered.tolist()))
check('raster: filtered toolpath keeps no travel-only segment spanning multiple layers',
      float(filtered[:, 2].max() - filtered[:, 2].min()) <= 0.6 + 1e-9,
      'z range {}'.format((filtered[:, 2].min(), filtered[:, 2].max())))

# two points 0.4 apart along layer 0's own pass (both near its far edge,
# x=6.9 and x=6.5, at z=0.2) must land close together in s - WITHOUT
# filtering, x=6.9 snaps onto the end-of-build return segment (which also
# sits at x=7, spanning all Z) while x=6.5 doesn't, producing a large jump.
near_edge_close = coordinate_function(np.array([6.5, 0, 0.2]), 'toolpath_arc_length', toolpath=filtered)
near_edge_far = coordinate_function(np.array([6.9, 0, 0.2]), 'toolpath_arc_length', toolpath=filtered)
check('raster: two nearby points on the same layer stay close in s (no travel-segment hijack)',
      abs(near_edge_far - near_edge_close) < 0.05,
      's(x=6.5)={:.4f} s(x=6.9)={:.4f} diff={:.4f}'.format(
          near_edge_close, near_edge_far, abs(near_edge_far - near_edge_close)))

# on examples/wall's real toolpath.crs (20 layers, Z 0.4..4.0, so the
# end-of-build return-to-origin segment spans a much larger Z range than
# this 3-layer toy example) the unfiltered jump is large and directly
# confirmed (see project memory) - this toy example is deliberately small
# just to keep filter_deposit_segments' output easy to hand-verify above.

try:
    filter_deposit_segments(np.array([[0, 0, 0], [1, 0, 0]], dtype=float), np.array([0, 0]))
    check('all-travel toolpath raises ValueError', False, 'did not raise')
except ValueError:
    check('all-travel toolpath raises ValueError', True)

print('Toolpath arc-length tests\n' + '=' * 60)
all_passed = True
for label, passed, *detail in results:
    status = 'PASS' if passed else 'FAIL'
    d = detail[0] if detail else ''
    print('[{}] {}{}'.format(status, label, ' - ' + d if d else ''))
    all_passed = all_passed and passed
print('=' * 60)
print('RESULT: {}'.format('ALL PASS' if all_passed else 'FAILURES PRESENT'))

import sys
sys.exit(0 if all_passed else 1)
