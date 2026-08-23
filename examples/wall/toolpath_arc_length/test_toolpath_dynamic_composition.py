# Tests for composition_lib.py's dynamic-composition additions:
# simulate_first_order_composition, build_toolpath_composition_table, and
# project_to_toolpath's extra_tables interpolation. No pytest in this project
# (see test_toolpath_arc_length.py) - plain assert-and-report, same convention.

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from composition_lib import (build_toolpath_arc_length_table, project_to_toolpath,
                              simulate_first_order_composition, build_toolpath_composition_table,
                              build_target_field_from_mesh)

results = []


def check(label, condition, detail=''):
    results.append((label, bool(condition), detail))


def close(a, b, tol=1e-9):
    return abs(a - b) <= tol


# ---- 1. tau=0: output must equal command exactly (infinitely fast system) ----
times = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
phi_cmd = np.array([0.0, 0.0, 1.0, 1.0, 0.5])
phi_out = simulate_first_order_composition(times, phi_cmd, tau=0.0)
check('tau=0: output equals command exactly', np.allclose(phi_out, phi_cmd), 'got {}'.format(phi_out))

# ---- 2. step response matches the analytic 1-exp(-t/tau) curve ----
# fine time grid; command is 0 at the very first sample (t=0, sets the initial
# condition) then jumps to 1 and holds - a true step, not a constant command
t_fine = np.linspace(0, 10, 1001)
step_cmd = np.ones_like(t_fine)
step_cmd[0] = 0.0
tau = 2.0
phi_step = simulate_first_order_composition(t_fine, step_cmd, tau=tau)
t1 = t_fine[1]  # step "happens" at the 2nd sample, ZOH holds cmd[0]=0 until then
analytic = np.where(t_fine < t1, 0.0, 1 - np.exp(-(t_fine - t1) / tau))
check('step response matches analytic 1-exp(-t/tau) (tau=2)',
      np.max(np.abs(phi_step - analytic)) < 1e-3,
      'max abs err = {:.6f}'.format(np.max(np.abs(phi_step - analytic))))

# ---- 3. large tau relative to build time -> heavy smoothing, output changes slowly ----
phi_slow = simulate_first_order_composition(times, phi_cmd, tau=1000.0)
check('tau >> build duration: output barely moves from its initial value',
      np.max(np.abs(phi_slow - phi_slow[0])) < 0.01,
      'phi_slow={}'.format(phi_slow))

# ---- 4. delay shifts the response start, doesn't change its shape ----
phi_nodelay = simulate_first_order_composition(t_fine, step_cmd, tau=tau, delay=0.0)
phi_delay = simulate_first_order_composition(t_fine, step_cmd, tau=tau, delay=3.0)
# before the delay elapses, output should stay at the initial command value
early = t_fine < 3.0
check('delay: output flat at initial value before the delay elapses',
      np.allclose(phi_delay[early], step_cmd[0], atol=1e-9), 'phi_delay[early]={}'.format(phi_delay[early][:5]))
# both curves approach the same steady state (1.0); by t=10 the delayed one is
# simply 3s "behind" (only 7s of settling vs 10s), so they're close but not
# identical - a loose tolerance confirms convergence without over-specifying
check('delay: output eventually approaches the same steady-state as no delay',
      close(phi_delay[-1], phi_nodelay[-1], tol=0.05), 'delayed={} undelayed={}'.format(phi_delay[-1], phi_nodelay[-1]))

# ---- 5. monotonic command + positive tau -> monotonic, non-overshooting output ----
ramp_cmd = np.linspace(0, 1, 50)
ramp_t = np.linspace(0, 20, 50)
phi_ramp = simulate_first_order_composition(ramp_t, ramp_cmd, tau=1.5)
check('monotonic command + first-order lag: output stays monotonic (no overshoot)',
      bool(np.all(np.diff(phi_ramp) >= -1e-12)), 'diffs min={}'.format(np.diff(phi_ramp).min()))
check('lag output always within command bounds [0,1]',
      bool(np.all((phi_ramp >= -1e-9) & (phi_ramp <= 1 + 1e-9))), 'phi_ramp={}'.format(phi_ramp))

# ---- 6. output always lags behind (or equals, never leads) an increasing command ----
check('output never exceeds the running max of the command (no lead/overshoot)',
      bool(np.all(phi_ramp <= np.maximum.accumulate(ramp_cmd) + 1e-9)))

# ---- 7. edge-case errors ----
try:
    simulate_first_order_composition(np.array([0.0, 1.0]), np.array([0.0]), tau=1.0)
    check('mismatched lengths raises ValueError', False, 'did not raise')
except ValueError:
    check('mismatched lengths raises ValueError', True)

try:
    simulate_first_order_composition(times, phi_cmd, tau=-1.0)
    check('negative tau raises ValueError', False, 'did not raise')
except ValueError:
    check('negative tau raises ValueError', True)

try:
    simulate_first_order_composition(np.zeros(0), np.zeros(0), tau=1.0)
    check('empty input raises ValueError', False, 'did not raise')
except ValueError:
    check('empty input raises ValueError', True)

# ---- 8. build_toolpath_composition_table wires xyz + time + target field together ----
straight = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0], [4, 0, 0]], dtype=float)
straight_time = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
phi_table = build_toolpath_composition_table(
    straight, straight_time, phi_target_fn=lambda x, y, z: x / 4.0, tau=0.0)
check('composition table with tau=0 matches target field evaluated at each waypoint',
      np.allclose(phi_table, [0.0, 0.25, 0.5, 0.75, 1.0]), 'got {}'.format(phi_table))

# ---- 9. extra_tables interpolates exactly like arc length (same segment/t weights) ----
arc = build_toolpath_arc_length_table(straight)
centroids = np.array([[0.5, 0, 0], [2.0, 0, 0], [3.5, 0, 0]])
arc_out, dist_out, extra = project_to_toolpath(centroids, straight, arc, extra_tables={'arc_copy': arc})
check('extra_tables reproduces arc length exactly when given the arc table itself',
      np.allclose(arc_out, extra['arc_copy']), 'arc_out={} extra={}'.format(arc_out, extra['arc_copy']))

arc_out2, dist_out2, extra2 = project_to_toolpath(centroids, straight, arc, extra_tables={'phi': phi_table})
check('extra_tables interpolates phi consistently with arc-length fraction on a straight path',
      np.allclose(extra2['phi'], arc_out2 / arc[-1]), 'phi={} arc_frac={}'.format(extra2['phi'], arc_out2 / arc[-1]))

# ---- 10. extra_tables leaves the original return shape untouched when omitted ----
plain = project_to_toolpath(centroids, straight, arc)
check('omitting extra_tables returns the original 2-tuple', len(plain) == 2, 'len={}'.format(len(plain)))
with_seg = project_to_toolpath(centroids, straight, arc, return_segment=True)
check('return_segment without extra_tables returns the original 3-tuple', len(with_seg) == 3, 'len={}'.format(len(with_seg)))
with_both = project_to_toolpath(centroids, straight, arc, return_segment=True, extra_tables={'phi': phi_table})
check('return_segment + extra_tables returns a 4-tuple, extra dict last', len(with_both) == 4, 'len={}'.format(len(with_both)))
check('extra dict is the last element and has the requested key',
      isinstance(with_both[-1], dict) and 'phi' in with_both[-1], 'last={}'.format(with_both[-1]))

# ---- 11. single-centroid input returns scalars, including for extra_tables ----
single_arc, single_dist, single_extra = project_to_toolpath(
    np.array([2.0, 0, 0]), straight, arc, extra_tables={'phi': phi_table})
check('single-centroid extra_tables output is a plain float, not an array',
      isinstance(single_extra['phi'], float), 'type={}'.format(type(single_extra['phi'])))
check('single-centroid extra_tables value matches batch result at the same point',
      close(single_extra['phi'], extra2['phi'][1]), 'single={} batch={}'.format(single_extra['phi'], extra2['phi'][1]))

# ---- 12. extra_tables length mismatch raises ValueError ----
try:
    project_to_toolpath(centroids, straight, arc, extra_tables={'bad': np.array([1.0, 2.0])})
    check('extra_tables length mismatch raises ValueError', False, 'did not raise')
except ValueError:
    check('extra_tables length mismatch raises ValueError', True)

# ---- 13. build_target_field_from_mesh: nearest-neighbor lookup on a mesh field ----
mesh_points_1d = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0], [4, 0, 0]], dtype=float)
mesh_values_1d = np.array([0.0, 0.0, 0.0, 1.0, 1.0])  # step at x=2.5
phi_mesh = build_target_field_from_mesh(mesh_points_1d, mesh_values_1d)
check('mesh field: query exactly at a mesh point returns that point\'s value',
      close(phi_mesh(2.0, 0, 0), 0.0), 'got {}'.format(phi_mesh(2.0, 0, 0)))
check('mesh field: query closer to x=3 than x=2 returns x=3\'s value',
      close(phi_mesh(2.6, 0, 0), 1.0), 'got {}'.format(phi_mesh(2.6, 0, 0)))
check('mesh field: query closer to x=2 than x=3 returns x=2\'s value',
      close(phi_mesh(2.4, 0, 0), 0.0), 'got {}'.format(phi_mesh(2.4, 0, 0)))
check('mesh field: query far off-axis still snaps to nearest mesh point',
      close(phi_mesh(0.1, 5.0, 0), 0.0), 'got {}'.format(phi_mesh(0.1, 5.0, 0)))

try:
    build_target_field_from_mesh(np.zeros((3, 3)), np.zeros(2))
    check('mesh field: mismatched lengths raises ValueError', False, 'did not raise')
except ValueError:
    check('mesh field: mismatched lengths raises ValueError', True)

try:
    build_target_field_from_mesh(np.zeros((0, 3)), np.zeros(0))
    check('mesh field: empty mesh raises ValueError', False, 'did not raise')
except ValueError:
    check('mesh field: empty mesh raises ValueError', True)

# brute-force cross-check against a scattered 3D mesh (independent of cKDTree's
# internals - confirms build_target_field_from_mesh's lookup is really nearest-
# neighbor, not some other ordering)
rng = np.random.default_rng(1)
scatter_points = rng.uniform(-10, 10, size=(200, 3))
scatter_values = rng.uniform(0, 1, size=200)
phi_scatter = build_target_field_from_mesh(scatter_points, scatter_values)
probe_pts = rng.uniform(-10, 10, size=(20, 3))
brute_force_ok = True
for qx, qy, qz in probe_pts:
    d = np.linalg.norm(scatter_points - np.array([qx, qy, qz]), axis=1)
    expected_val = scatter_values[np.argmin(d)]
    if not close(phi_scatter(qx, qy, qz), expected_val, tol=1e-9):
        brute_force_ok = False
        break
check('mesh field: matches brute-force nearest-neighbor on scattered 3D points', brute_force_ok)

# ---- 14. end-to-end: mesh target field -> toolpath dynamics -> per-element assignment ----
# design mesh: 5 points along the build's X extent with a linear target gradient
design_mesh_pts = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0], [4, 0, 0]], dtype=float)
design_mesh_vals = np.array([0.0, 0.25, 0.5, 0.75, 1.0])  # linear x/4, matches earlier closed-form test
phi_from_mesh = build_target_field_from_mesh(design_mesh_pts, design_mesh_vals)

toolpath_e2e = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0], [4, 0, 0]], dtype=float)
toolpath_time_e2e = np.array([0.0, 1.0, 2.0, 3.0, 4.0])

# tau=0: infinitely fast system, so the per-waypoint table should reproduce the
# mesh's nearest-neighbor field exactly at each waypoint (waypoints coincide
# with mesh points here, so this is an exact roundtrip, not just "close")
table_instant = build_toolpath_composition_table(toolpath_e2e, toolpath_time_e2e, phi_from_mesh, tau=0.0)
check('end-to-end (tau=0): per-waypoint table matches the mesh field exactly at coincident points',
      np.allclose(table_instant, design_mesh_vals), 'got {}'.format(table_instant))

# tau>0: lag should still leave the assignment monotonic and within [0,1],
# same sanity properties checked in isolation above, now through the full
# mesh -> command -> lag -> per-element chain
table_lagged = build_toolpath_composition_table(toolpath_e2e, toolpath_time_e2e, phi_from_mesh, tau=1.5)
check('end-to-end (tau>0): lagged table stays monotonic (mesh gradient is monotonic)',
      bool(np.all(np.diff(table_lagged) >= -1e-12)), 'table={}'.format(table_lagged))
check('end-to-end (tau>0): lagged table stays within [0,1]',
      bool(np.all((table_lagged >= -1e-9) & (table_lagged <= 1 + 1e-9))), 'table={}'.format(table_lagged))
check('end-to-end (tau>0): lag makes the table strictly less "advanced" than tau=0 at each interior point',
      bool(np.all(table_lagged[1:-1] <= table_instant[1:-1] + 1e-9)), 'lagged={} instant={}'.format(
          table_lagged[1:-1], table_instant[1:-1]))

# finally, hand the lagged table to project_to_toolpath exactly as a real
# pipeline would, to assign composition to simulation element centroids
arc_e2e = build_toolpath_arc_length_table(toolpath_e2e)
sim_element_centroids = np.array([[0.5, 0, 0], [2.0, 0, 0], [3.5, 0, 0]])
_, _, elem_extra = project_to_toolpath(sim_element_centroids, toolpath_e2e, arc_e2e,
                                        extra_tables={'phi': table_lagged})
check('end-to-end: simulation elements get monotonically increasing composition along the build',
      bool(np.all(np.diff(elem_extra['phi']) > 0)), 'phi={}'.format(elem_extra['phi']))
check('end-to-end: assigned element composition stays within [0,1]',
      bool(np.all((elem_extra['phi'] >= 0) & (elem_extra['phi'] <= 1))), 'phi={}'.format(elem_extra['phi']))

print('Toolpath dynamic-composition tests\n' + '=' * 60)
all_passed = True
for label, passed, *detail in results:
    status = 'PASS' if passed else 'FAIL'
    d = detail[0] if detail else ''
    print('[{}] {}{}'.format(status, label, ' - ' + d if d else ''))
    all_passed = all_passed and passed
print('=' * 60)
print('RESULT: {}'.format('ALL PASS' if all_passed else 'FAILURES PRESENT'))

sys.exit(0 if all_passed else 1)
