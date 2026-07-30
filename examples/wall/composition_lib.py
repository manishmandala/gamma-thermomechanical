# Shared composition-generation engine, used by every script that computes
# a per-element *ELEMENT_COMPOSITION fraction (gradient_material_continuous.py,
# gradient_material_continuous_cu.py, prepare_composition.py). Three
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
# equivalent to what gradient_material_continuous.py computed inline before
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


def coordinate_function(centroid, mode, bounds=None, toolpath=None):
    """Maps one element's centroid to a normalized scalar s in [0,1].

    mode='global_x' | 'global_y' | 'global_z': linear position along that
    global axis, normalized against `bounds` (see compute_bounds). This is
    the only behavior in production today - exactly what
    gradient_material_continuous.py's inline
    `(cx - x_min) / (x_max - x_min)` computed before this refactor.

    mode='toolpath_arc_length': NOT YET IMPLEMENTED. Composition should vary
    by distance traveled along the deposition toolpath rather than raw
    global position. Requires projecting each centroid onto the nearest
    toolpath *segment* (not just the nearest waypoint - on the recommended
    curved test dataset, toolpath waypoints are ~18x farther apart than a
    single element, so nearest-waypoint snapping alone would produce a
    visibly staircased gradient) and interpolating cumulative arc length
    within that segment. Raises NotImplementedError until built.
    """
    if mode in _AXIS_INDEX:
        if bounds is None:
            raise ValueError("mode={!r} requires `bounds` (see compute_bounds)".format(mode))
        axis_name = mode.split('_')[1]  # 'global_x' -> 'x'
        lo, hi = bounds[axis_name + '_min'], bounds[axis_name + '_max']
        v = centroid[_AXIS_INDEX[mode]]
        return (v - lo) / (hi - lo) if hi > lo else 0.0
    elif mode == 'toolpath_arc_length':
        raise NotImplementedError(
            "toolpath_arc_length is not implemented yet - see this function's docstring")
    else:
        raise ValueError("unknown coordinate mode: {!r} (expected one of {})".format(mode, COORDINATE_MODES))


# ---- composition presets ----------------------------------------------
# Ported verbatim (same math, same defaults) from gradient_material_continuous.py.
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
    # gradient_material_continuous.py's original header note for why
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
