# Shared composition-generation engine, used by every script that computes
# a per-element *ELEMENT_COMPOSITION fraction (gradient_material_continuous_TI64_IN718.py,
# gradient_material_continuous_TI64_Cu.py, prepare_composition.py). Three
# deliberately separate stages, matching the target architecture agreed on
# for the curved-path (toolpath-arc-length) composition work:
#
#   compute_centroid(nodes, node_ids)            -> (cx, cy, cz)
#   coordinate_function(centroid, mode, ...)     -> normalized scalar s in [0,1]
#   composition_function(x_norm, z_norm, mode)   -> fraction in [0,1]
#
# Before this module existed, every script above computed its own centroid,
# normalized it inline, and called a hardcoded composition function - three
# independent copies of the same three concerns tangled together. Separating
# them means a new coordinate mode (toolpath_arc_length, still not
# implemented - see coordinate_function's docstring) only ever needs a new
# branch in ONE function, and every existing composition preset keeps
# working against it completely unchanged, since none of them know or care
# where their input came from.
#
# REGRESSION-CRITICAL: coordinate_function's global_x/global_y/global_z
# behavior and every function in COMPOSITION_PRESETS must stay byte-for-byte
# equivalent to what gradient_material_continuous_TI64_IN718.py computed inline before
# this refactor - see test_composition_regression.py, which proves this
# against the actual committed thinwall_graded.k/thinwall_graded_cu.k output.

import numpy as np

COORDINATE_MODES = ('global_x', 'global_y', 'global_z', 'toolpath_arc_length')
_AXIS_INDEX = {'global_x': 0, 'global_y': 1, 'global_z': 2}


def compute_centroid(nodes, node_ids):
    """Element centroid = mean of its corner node coordinates.

    nodes: dict {node_id: (x, y, z)}
    node_ids: this element's corner node ids (8, for a HEX8 element)
    """
    n = len(node_ids)
    cx = sum(nodes[i][0] for i in node_ids) / n
    cy = sum(nodes[i][1] for i in node_ids) / n
    cz = sum(nodes[i][2] for i in node_ids) / n
    return (cx, cy, cz)


def compute_bounds(centroids):
    """Min/max per global axis across a collection of (cx,cy,cz) centroids -
    the normalization range coordinate_function's global_* modes need.
    Computed once per element set, not per element.
    """
    arr = np.asarray(centroids, dtype=float)
    return dict(
        x_min=float(arr[:, 0].min()), x_max=float(arr[:, 0].max()),
        y_min=float(arr[:, 1].min()), y_max=float(arr[:, 1].max()),
        z_min=float(arr[:, 2].min()), z_max=float(arr[:, 2].max()),
    )


def build_toolpath_arc_length_table(toolpath_xyz):
    """Cumulative arc length per toolpath waypoint.

    toolpath_xyz: (N,3) waypoints, already in deposition/time order (this
    project's .crs toolpath rows are always strictly increasing in time -
    confirmed during inspection, not re-sorted here or anywhere else).

    Returns an (N,) float64 array, arc[0]=0.0, monotonically NON-decreasing
    (not strictly increasing - repeated/zero-length consecutive waypoints,
    e.g. the laser pausing in place, produce equal consecutive values rather
    than being dropped, so the result always stays index-aligned with the
    input: len(arc) == len(toolpath_xyz) always).
    """
    toolpath_xyz = np.asarray(toolpath_xyz, dtype=float)
    if len(toolpath_xyz) == 0:
        raise ValueError("build_toolpath_arc_length_table: toolpath has zero points")
    if len(toolpath_xyz) == 1:
        # a single point defines no segments; arc length is trivially 0 there
        return np.zeros(1)
    seglen = np.linalg.norm(np.diff(toolpath_xyz, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(seglen)])


def simulate_first_order_composition(times, phi_command, tau, delay=0.0):
    """First-order-plus-dead-time (FOPDT) model of a feeder/mixing system's
    response to a commanded composition signal:

        tau * dphi/dt + phi(t) = phi_command(t - delay)

    times: (N,) real deposition time per sample, strictly the same convention
    as this project's .crs time column (non-decreasing; see generate_toolpath.py).
    phi_command: (N,) commanded composition fraction at each time sample -
    typically a part-design target field phi(x,y,z) evaluated at the toolpath's
    position at that time (see build_toolpath_composition_table).
    tau: response time constant, same units as `times`. tau=0 means an
    infinitely fast system - output tracks (delayed) command exactly. Must be
    >= 0.
    delay: pure transport delay (dead time), same units as `times`. Applied by
    resampling phi_command at (times - delay) via linear interpolation, holding
    phi_command[0] for any t < times[0] - delay (system hasn't received a
    command yet, so it's assumed to start at the initial command value rather
    than undefined/zero). Must be >= 0.

    Returns phi_actual, an (N,) array aligned index-for-index with `times` -
    the composition the system actually produces once its own dynamics are
    accounted for, always lagging behind (never leading) phi_command.

    Uses the exact zero-order-hold discretization, not an Euler approximation:
    over each interval [times[i-1], times[i]], phi_command is held constant at
    its value at times[i-1] (this matches how the .crs toolpath already works -
    a state/position holds until the NEXT waypoint changes it), so the closed-
    form solution of the linear ODE over that interval is exact for any
    tau > 0:

        phi_actual[i] = phi_actual[i-1]*exp(-dt/tau) + phi_command[i-1]*(1-exp(-dt/tau))

    phi_actual[0] is initialized to phi_command[0] (delayed) - the system is
    assumed to already be at steady-state with the build's starting command
    before deposition begins, rather than starting from an arbitrary/undefined
    state.
    """
    times = np.asarray(times, dtype=float)
    phi_command = np.asarray(phi_command, dtype=float)
    if len(times) != len(phi_command):
        raise ValueError("times length ({}) doesn't match phi_command ({})".format(
            len(times), len(phi_command)))
    if len(times) == 0:
        raise ValueError("simulate_first_order_composition: got zero samples")
    if tau < 0:
        raise ValueError("tau must be >= 0, got {}".format(tau))
    if delay < 0:
        raise ValueError("delay must be >= 0, got {}".format(delay))

    if delay > 0:
        phi_cmd = np.interp(times - delay, times, phi_command, left=phi_command[0])
    else:
        phi_cmd = phi_command

    if tau == 0:
        # infinitely fast system: output IS the (delayed) command, no lag
        return phi_cmd.copy()

    phi_actual = np.empty_like(phi_cmd)
    phi_actual[0] = phi_cmd[0]
    for i in range(1, len(times)):
        dt = times[i] - times[i - 1]
        if dt <= 0:
            # simultaneous/out-of-order timestamps: no time elapsed, no change
            phi_actual[i] = phi_actual[i - 1]
            continue
        decay = np.exp(-dt / tau)
        phi_actual[i] = phi_actual[i - 1] * decay + phi_cmd[i - 1] * (1 - decay)
    return phi_actual


def build_target_field_from_mesh(mesh_points, mesh_values):
    """Wraps a mesh that already carries a per-entity target composition
    value into the phi_target_fn(x, y, z) callable build_toolpath_composition_table
    expects - this is the "the input is the mesh" case: the part designer's
    intent arrives as a field already baked onto a mesh (e.g. an externally
    supplied design mesh's cell_data, same shape as
    external_mesh/external_mesh_composition.py's sol_100.vtu cell_data['mat'];
    or per-element compute_centroid() output paired with a composition array
    computed some other way) - not a closed-form function of position.

    mesh_points: (K,3) array - one point per mesh entity that carries a
    composition value (element centroids, node coordinates, whichever
    `mesh_values` is aligned to).
    mesh_values: (K,) array - the target composition fraction at each of
    those points, index-aligned with mesh_points.

    Returns phi_target_fn(x, y, z) -> float: nearest-neighbor lookup via a
    KD-tree (same approach already used for spatial lookups in this
    directory's compare_composition_modes_part*.py) - returns mesh_values[i]
    for whichever mesh_points[i] is closest to the query point. Deliberately
    nearest-neighbor, not interpolation: the target field IS the mesh's data,
    verbatim, not a smoothed version of it - same rationale as
    project_to_toolpath's nearest-SEGMENT (not blended) search.

    The KD-tree is built once, up front (O(K log K)); each call to the
    returned phi_target_fn is then a single O(log K) query - cheap even for
    a large design mesh, since build_toolpath_composition_table calls it
    once per TOOLPATH WAYPOINT (typically tens to low thousands), not once
    per simulation element.
    """
    from scipy.spatial import cKDTree
    mesh_points = np.asarray(mesh_points, dtype=float)
    mesh_values = np.asarray(mesh_values, dtype=float)
    if len(mesh_points) != len(mesh_values):
        raise ValueError("mesh_points length ({}) doesn't match mesh_values ({})".format(
            len(mesh_points), len(mesh_values)))
    if len(mesh_points) == 0:
        raise ValueError("build_target_field_from_mesh: got an empty mesh")
    tree = cKDTree(mesh_points)

    def phi_target_fn(x, y, z):
        _dist, idx = tree.query([x, y, z])
        return float(mesh_values[idx])

    return phi_target_fn


def build_toolpath_composition_table(toolpath_xyz, toolpath_time, phi_target_fn, tau, delay=0.0):
    """Converts a part-design target composition field into the composition
    actually achievable along the toolpath, given feeder/mixing dynamics.

    toolpath_xyz: (N,3) waypoints, toolpath_time: (N,) real deposition time per
    waypoint (this project's .crs time column), both already in deposition
    order and index-aligned (toolpath_time[i] is when the head was at
    toolpath_xyz[i]).
    phi_target_fn: callable (x, y, z) -> composition fraction in [0,1] - the
    part designer's spatial spec, evaluated at each waypoint's real position.
    This is intentionally left as a plain callable rather than a fixed format
    (e.g. closed-form function, region/label lookup, interpolated field from a
    CAD tool) - any of those can be wrapped in one.
    tau, delay: see simulate_first_order_composition.

    Returns phi_actual, an (N,) array index-aligned with toolpath_xyz - the
    composition the system actually produces at each waypoint once its
    response dynamics are accounted for. Pass this into project_to_toolpath's
    `extra_tables` (e.g. extra_tables={'phi': phi_actual}) to assign each
    element the composition actually deposited at its nearest point on the
    path, using the exact same per-segment interpolation as arc length.
    """
    toolpath_xyz = np.asarray(toolpath_xyz, dtype=float)
    toolpath_time = np.asarray(toolpath_time, dtype=float)
    if len(toolpath_time) != len(toolpath_xyz):
        raise ValueError("toolpath_time length ({}) doesn't match toolpath_xyz ({})".format(
            len(toolpath_time), len(toolpath_xyz)))
    phi_command = np.array([phi_target_fn(x, y, z) for x, y, z in toolpath_xyz], dtype=float)
    return simulate_first_order_composition(toolpath_time, phi_command, tau, delay=delay)


def filter_deposit_segments(toolpath_xyz, state):
    """Reduces a raw toolpath down to only the segments where material was
    actually being deposited, dropping travel/repositioning moves.

    toolpath_xyz: (N,3) waypoints. state: (N,) array, this project's .crs
    convention - state[i]==1 means the segment ENDING at waypoint i (i.e.
    toolpath_xyz[i-1] -> toolpath_xyz[i]) was a deposit move with the laser
    on; state[i]==0 means it was a travel move (laser off) - repositioning
    between passes, or the end-of-build return to a park position.

    Why this matters for project_to_toolpath: its nearest-segment search
    has no notion of "deposit" vs "travel" - every segment is a candidate,
    travel moves included. On a layer-by-layer raster build this is a real
    problem: travel segments include the vertical reposition between each
    layer, and - critically - a single end-of-build "return to origin" move
    that can span the ENTIRE build height in one segment sitting right at
    the raster's edge. Any element near that X position, at ANY layer, can
    end up geometrically closer to that one all-heights-spanning travel
    segment than to its own layer's actual deposit pass, snapping its
    arc-length coordinate to nonsense (confirmed on examples/wall's own
    raster toolpath.crs: elements 0.4mm apart at the same layer computed
    arc-length coordinates 0.29 apart, because one of them matched the
    end-of-build travel segment instead of its own layer's pass - see
    project memory). Restricting the search to deposit-only segments fixes
    this at the source instead of trying to special-case travel segments
    inside the search itself.

    Returns an (M,3) array (M <= N) of exactly the waypoints that are an
    endpoint of at least one deposit segment, in original order - directly
    usable with build_toolpath_arc_length_table / project_to_toolpath /
    coordinate_function exactly like a raw toolpath array (two consecutive
    deposit passes that aren't adjacent in the original array become
    directly connected here, which is correct: that connecting distance is
    real distance traveled between two deposits, just not itself a deposit).
    Raises ValueError if no deposit segments exist at all.
    """
    toolpath_xyz = np.asarray(toolpath_xyz, dtype=float)
    state = np.asarray(state)
    if len(state) != len(toolpath_xyz):
        raise ValueError("state length ({}) doesn't match toolpath_xyz ({})".format(
            len(state), len(toolpath_xyz)))
    if len(toolpath_xyz) < 2:
        raise ValueError("filter_deposit_segments needs at least 2 toolpath points, got {}".format(
            len(toolpath_xyz)))
    deposit_end = (state[1:] == 1)   # (N-1,) - True where segment i->i+1 is a deposit move
    if not deposit_end.any():
        raise ValueError("filter_deposit_segments: no deposit (state==1) segments found in toolpath")
    keep = np.zeros(len(toolpath_xyz), dtype=bool)
    keep[:-1] |= deposit_end   # start point of each deposit segment
    keep[1:] |= deposit_end    # end point of each deposit segment
    return toolpath_xyz[keep]


def project_to_toolpath(centroids, toolpath_xyz, cumulative_arc_length, chunk_size=2000, return_segment=False,
                         extra_tables=None):
    """Projects each centroid onto the closest point on the toolpath's
    polyline - nearest SEGMENT, not nearest waypoint (see this module's
    header / coordinate_function's docstring for why nearest-waypoint alone
    is inadequate: waypoints can sit far apart relative to element size, so
    snapping to the nearest one alone produces a visibly staircased field).

    centroids: (M,3) array of element centroids, or a single (3,) centroid.
    toolpath_xyz: (N,3) waypoints, N>=2 (need at least one segment).
    cumulative_arc_length: (N,) from build_toolpath_arc_length_table(toolpath_xyz).

    Returns (arc_at_projection, distance), each shape (M,) (or scalars, if
    a single centroid was passed) - the RAW (not yet normalized to [0,1])
    cumulative arc length at each centroid's closest projected point, and
    the Euclidean distance to that projection. If return_segment=True,
    additionally returns which segment index (0-based, into
    toolpath_xyz[:-1]) each centroid mapped to - (kept for diagnostics - see
    diagnose_part002_mapping.py).

    extra_tables: optional dict {name: (N,) array} of additional per-waypoint
    scalar values - e.g. a phi_actual table from build_toolpath_composition_table
    - to interpolate at each centroid's projected point using the EXACT same
    (segment, t) as arc length itself: value = table[seg] + t*(table[seg+1] -
    table[seg]). When supplied, one more dict {name: (M,) array} (or {name:
    float} for single-centroid input) is appended to the return tuple, after
    whatever return_segment already adds. Omit for the original 2/3-tuple
    return shape - this is purely additive, every existing call site is
    unaffected.

    Algorithm, per centroid: for every segment (p_i -> p_{i+1}), compute the
    projection fraction t = dot(centroid - p_i, p_{i+1} - p_i) / |p_{i+1} - p_i|^2,
    clamp t to [0,1] (so a centroid beyond a segment's end snaps to that
    endpoint rather than extrapolating off the path), take the resulting
    projected point, and keep whichever segment gives the smallest distance.
    This is an exact nearest-point-on-polyline search, not an approximation -
    clamping per segment and taking a global min across all segments also
    correctly handles centroids that fall before the first waypoint or
    after the last one, with no special-casing needed (they simply project
    onto the first or last segment's clamped endpoint).

    Memory: processed in chunks of `chunk_size` centroids (each chunk builds
    one (chunk_size, n_segments) distance matrix, then discards it) instead
    of one (M, n_segments) matrix for every centroid at once. On the largest
    mesh in this project (part003: 70,690 elements x 1,799 segments), an
    unchunked matrix would be ~127M floats (~1GB) - chunking bounds peak
    memory to chunk_size x n_segments regardless of total element count, at
    zero accuracy cost (this is exact chunking, not a spatial/approximate
    filter - every centroid still checks every segment, just chunk_size at
    a time).
    """
    single = (np.ndim(centroids) == 1)
    pts = np.atleast_2d(np.asarray(centroids, dtype=float))
    toolpath_xyz = np.asarray(toolpath_xyz, dtype=float)
    cumulative_arc_length = np.asarray(cumulative_arc_length, dtype=float)

    if len(toolpath_xyz) < 2:
        raise ValueError(
            "project_to_toolpath needs at least 2 toolpath points (>=1 segment), got {}".format(
                len(toolpath_xyz)))
    if len(cumulative_arc_length) != len(toolpath_xyz):
        raise ValueError("cumulative_arc_length length ({}) doesn't match toolpath_xyz ({})".format(
            len(cumulative_arc_length), len(toolpath_xyz)))

    p0 = toolpath_xyz[:-1]                # (S,3) segment start points
    p1 = toolpath_xyz[1:]                 # (S,3) segment end points
    v = p1 - p0                           # (S,3) segment vectors
    seg_len_sq = np.sum(v * v, axis=1)    # (S,) - zero for repeated/zero-length waypoints

    n = len(pts)
    arc_out = np.empty(n)
    dist_out = np.empty(n)
    seg_out = np.empty(n, dtype=int)

    extra_in = {}
    extra_out = {}
    if extra_tables:
        for name, values in extra_tables.items():
            values = np.asarray(values, dtype=float)
            if len(values) != len(toolpath_xyz):
                raise ValueError("extra_tables[{!r}] length ({}) doesn't match toolpath_xyz ({})".format(
                    name, len(values), len(toolpath_xyz)))
            extra_in[name] = values
            extra_out[name] = np.empty(n)

    for start in range(0, n, chunk_size):
        chunk = pts[start:start + chunk_size]                          # (Mc,3)
        w = chunk[:, None, :] - p0[None, :, :]                         # (Mc,S,3)
        t_num = np.sum(w * v[None, :, :], axis=2)                      # (Mc,S)
        # safe division: zero-length segments get t=0 - harmless, since
        # v=0 there too, so the "projected point" is just p0 regardless of t
        t = np.divide(t_num, seg_len_sq[None, :], out=np.zeros_like(t_num),
                       where=(seg_len_sq[None, :] > 0))
        t = np.clip(t, 0.0, 1.0)                                       # (Mc,S)
        proj = p0[None, :, :] + t[:, :, None] * v[None, :, :]          # (Mc,S,3)
        dist_sq = np.sum((chunk[:, None, :] - proj) ** 2, axis=2)      # (Mc,S)
        best_seg = np.argmin(dist_sq, axis=1)                          # (Mc,)
        rows = np.arange(len(chunk))
        best_t = t[rows, best_seg]
        arc_out[start:start + chunk_size] = (
            cumulative_arc_length[best_seg]
            + best_t * (cumulative_arc_length[best_seg + 1] - cumulative_arc_length[best_seg]))
        dist_out[start:start + chunk_size] = np.sqrt(dist_sq[rows, best_seg])
        seg_out[start:start + chunk_size] = best_seg
        for name, values in extra_in.items():
            extra_out[name][start:start + chunk_size] = (
                values[best_seg] + best_t * (values[best_seg + 1] - values[best_seg]))

    if single:
        base = (float(arc_out[0]), float(dist_out[0]), int(seg_out[0])) if return_segment \
            else (float(arc_out[0]), float(dist_out[0]))
        if extra_tables:
            return base + ({name: float(arr[0]) for name, arr in extra_out.items()},)
        return base
    base = (arc_out, dist_out, seg_out) if return_segment else (arc_out, dist_out)
    if extra_tables:
        return base + (extra_out,)
    return base


def coordinate_function(centroid, mode, bounds=None, toolpath=None, toolpath_arc=None, chunk_size=2000):
    """Maps element centroid(s) to a normalized scalar s in [0,1].

    centroid: a single (x,y,z) tuple, OR an (M,3) array/list of centroids
    (batch mode - use this for toolpath_arc_length on any non-trivial mesh;
    see project_to_toolpath's memory note). Returns a single float for
    single-centroid input, or an (M,) array for batch input.

    mode='global_x' | 'global_y' | 'global_z': linear position along that
    global axis, normalized against `bounds` (see compute_bounds). Exactly
    what gradient_material_continuous_TI64_IN718.py's inline
    `(cx - x_min) / (x_max - x_min)` computed before this module existed.

    mode='toolpath_arc_length': normalized distance traveled along the
    deposition toolpath, via project_to_toolpath (nearest-segment
    projection, not nearest-waypoint - see that function's docstring).
    Requires `toolpath` (an (N,3) array of waypoints, N>=2, already in
    deposition order). `toolpath_arc` (from build_toolpath_arc_length_table)
    is computed automatically if not supplied; pass a precomputed one in to
    avoid rebuilding it on repeated calls. If the toolpath's total length is
    exactly zero (every waypoint identical - the laser never moved), every
    centroid gets s=0.0 rather than dividing by zero, since there's no
    meaningful spatial variation to encode in that case.
    """
    is_batch = (np.ndim(centroid) == 2)
    pts = np.atleast_2d(np.asarray(centroid, dtype=float))

    if mode in _AXIS_INDEX:
        if bounds is None:
            raise ValueError("mode={!r} requires `bounds` (see compute_bounds)".format(mode))
        axis_name = mode.split('_')[1]  # 'global_x' -> 'x'
        lo, hi = bounds[axis_name + '_min'], bounds[axis_name + '_max']
        v = pts[:, _AXIS_INDEX[mode]]
        s = (v - lo) / (hi - lo) if hi > lo else np.zeros(len(pts))
    elif mode == 'toolpath_arc_length':
        if toolpath is None:
            raise ValueError("mode='toolpath_arc_length' requires `toolpath` (an (N,3) waypoint array)")
        toolpath = np.asarray(toolpath, dtype=float)
        if len(toolpath) < 2:
            raise ValueError(
                "mode='toolpath_arc_length' needs a toolpath with at least 2 points to define "
                "a segment, got {}".format(len(toolpath)))
        arc = toolpath_arc if toolpath_arc is not None else build_toolpath_arc_length_table(toolpath)
        total = arc[-1]
        if total <= 0:
            s = np.zeros(len(pts))
        else:
            arc_at, _dist = project_to_toolpath(pts, toolpath, arc, chunk_size=chunk_size)
            s = np.atleast_1d(arc_at) / total
    else:
        raise ValueError("unknown coordinate mode: {!r} (expected one of {})".format(mode, COORDINATE_MODES))

    # floating-point safety net: t was already clamped per-segment above, but
    # accumulated FP error (division, the global_* branch's own subtraction)
    # can still push a value a hair outside [0,1] - clip defensively.
    s = np.clip(s, 0.0, 1.0)
    return s if is_batch else float(s[0])


# ---- composition presets ----------------------------------------------
# Ported verbatim (same math, same defaults) from gradient_material_continuous_TI64_IN718.py.
# x_norm/z_norm follow that script's original convention: x_norm runs 0 (first
# endpoint material) -> 1 (second); z_norm is a second, independent axis that
# only `sinusoidal` actually uses - every preset shares one call signature so
# composition_function doesn't need to know which axes a given preset cares
# about. z_norm defaults to 0.0 so single-axis callers (e.g. the constant-
# composition endpoint-validation modes) don't need to supply it.

def linear(x_norm, z_norm=0.0):
    return x_norm


def sinusoidal(x_norm, z_norm=0.0, cycles_x=4.0, cycles_z=1.5):
    # additive x-wave + z-wave, not multiplicative - see
    # gradient_material_continuous_TI64_IN718.py's original header note for why
    # (a product term lets z's factor silently suppress x's variation).
    return 0.5 + 0.25 * np.sin(2*np.pi * cycles_x * x_norm) + 0.25 * np.sin(2*np.pi * cycles_z * z_norm)


def step(x_norm, z_norm=0.0, n_steps=4):
    return np.floor(x_norm * n_steps) / (n_steps - 1)


def sigmoid(x_norm, z_norm=0.0, steepness=10, midpoint=0.5):
    return 1 / (1 + np.exp(-steepness * (x_norm - midpoint)))


def constant(x_norm=0.0, z_norm=0.0, value=0.5):
    """Ignores both coordinates entirely - used by the endpoint-validation
    modes (prepare_composition.py) to force a uniform composition regardless
    of position, without needing a separate code path from the
    position-dependent presets."""
    return value


def quadratic(x_norm, z_norm=0.0, strength=1.0):
    """Symmetric parabola: 1.0 (fully material A) at both ends (x_norm=0
    and x_norm=1), dipping to (1-strength) at the midpoint, with a smooth
    (continuous, differentiable) gradient in between - not a step change.
    strength=1.0 -> pure material B at the midpoint; smaller strength ->
    a shallower dip. Intended for a "reinforced ends, weak middle" stress-
    test composition (e.g. --coordinate-mode global_x on a wall, so the two
    physical tips of the build are one material and the center is graded
    toward the other)."""
    return 1.0 - strength * 4 * x_norm * (1 - x_norm)


COMPOSITION_PRESETS = {
    'linear': linear,
    'sinusoidal': sinusoidal,
    'step': step,
    'sigmoid': sigmoid,
    'constant': constant,
    'quadratic': quadratic,
    # 'piecewise' - planned, deliberately not implemented yet
}


def composition_function(x_norm, z_norm, mode, **params):
    """Evaluates the named composition preset and clips to [0,1] (presets
    like sinusoidal can overshoot slightly outside that range)."""
    if mode not in COMPOSITION_PRESETS:
        raise ValueError(
            "unknown or not-yet-implemented composition mode: {!r} (available: {})".format(
                mode, sorted(COMPOSITION_PRESETS)))
    fn = COMPOSITION_PRESETS[mode]
    return float(np.clip(fn(x_norm, z_norm, **params), 0.0, 1.0))
