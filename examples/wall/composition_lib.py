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


def project_to_toolpath(centroids, toolpath_xyz, cumulative_arc_length, chunk_size=2000, return_segment=False):
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

    if single:
        if return_segment:
            return float(arc_out[0]), float(dist_out[0]), int(seg_out[0])
        return float(arc_out[0]), float(dist_out[0])
    if return_segment:
        return arc_out, dist_out, seg_out
    return arc_out, dist_out


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


COMPOSITION_PRESETS = {
    'linear': linear,
    'sinusoidal': sinusoidal,
    'step': step,
    'sigmoid': sigmoid,
    'constant': constant,
    # 'quadratic', 'piecewise' - planned, deliberately not implemented yet
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
