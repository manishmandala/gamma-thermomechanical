# Grades the DEPOSITED WALL (laser-built material) as a TI64/IN718 blend along
# X: one material band per element column (mesh x-resolution). The blend
# fraction at each column comes from COMPOSITION_FN below (linear, sinusoidal,
# step, sigmoid, or your own function of normalized position) rather than a
# hardcoded ramp, and each property is blended via MIXING_RULES, independently
# overridable per property rather than one hardcoded linear rule for
# everything. The base plate the wall is built on top of is left as a single
# TI64 material.
#
# Only touches the *ELEMENT_SOLID pid column, *PART/*MAT blocks, and adds
# blended property-curve files - node geometry, birth times, and the
# toolpath are untouched, so the laser schedule is unaffected.
#
# NOTE: thinwall.k's *PART block names are misleading - the part LABELED
# "Substrate" (pid 1) is actually the above-ground deposited wall (z > 0),
# and the part labeled "Build" (pid 2) is actually the below-ground base
# plate (z < 0). TARGET_PID below is chosen by mesh geometry, asserted
# before touching anything so a relabeling can't silently break this again.
#
# IMPORTANT: gamma.py's critical-timestep calc (update_dt) looks up
# thermal_TD curves via `thermal_TD[i+1]` - i.e. it assumes material IDs
# are sequential (1, 2, 3, ...) in the exact order their *MAT_THERMAL_
# ISOTROPIC_TD blocks appear in the file. Band pids below are therefore
# assigned sequentially with no gaps: pid 1 = base plate, pid 2..N+1 =
# gradient bands in increasing-x order.
#
# TI64/IN718 endpoint properties match examples/clad/clad.k, the project's
# existing TI64+IN718 pairing.
#
# CAVEAT: intermediate-band properties are linear rule-of-mixtures blends of
# the two endpoint materials - an approximation, not measured/CALPHAD data for
# actual TI64/IN718 mixtures. Density and Cp are reasonably well-behaved under
# linear mixing; solidus/liquidus are the least trustworthy, since dissimilar
# Ti + Ni-Fe superalloy mixtures are known to form brittle intermetermetallic
# phases (Laves, TiFe2, Ni3Ti) with eutectic-like behavior - the true solidus
# of an intermediate composition can sit well below both pure endpoints rather
# than on a straight line between them. Treat this gradient as a numerical
# convenience for toolpath/thermal studies, not validated material behavior.
#
# Band count is capped by mesh resolution: one band per distinct element
# column in x (70 in the current mesh, spacing 0.2), so steps land near ~1.4%
# each rather than exact 1% - there aren't enough columns to realize every
# integer percentage. Re-mesh the wall with ~100+ x-columns for true 1% steps.

import os
import numpy as np

MESH_FILE = 'thinwall.k'
PROP_DIR = '../0_properties'
GRAD_DIR = os.path.join(PROP_DIR, 'graded_wall')
TARGET_PID = 1   # the deposited wall (despite the *PART block calling it "Substrate")
BASE_PLATE_PID_OLD = 2

TI64 = dict(density=0.00440, solidus=1878, liquidus=1928, latent=286,
            cp=os.path.join(PROP_DIR, 'TI64_cp.txt'),
            cond=os.path.join(PROP_DIR, 'TI64_cond.txt'))
IN718 = dict(density=0.00819, solidus=1533, liquidus=1609, latent=270,
             cp=os.path.join(PROP_DIR, 'IN718_cp.txt'),
             cond=os.path.join(PROP_DIR, 'IN718_cond.txt'))


# ---- composition profile: fraction of IN718 as a function of normalized x ----
# x_norm runs 0 (TI64 end of the wall) -> 1 (IN718 end). Swap COMPOSITION_FN
# for any preset below, or write your own - it just needs to map x_norm in
# [0,1] to a fraction; output is clipped to [0,1], so overshoot is safe.

def linear(x_norm):
    return x_norm


def sinusoidal(x_norm, cycles=1.5):
    """Oscillates between the two materials `cycles` times across the wall."""
    return 0.5 - 0.5 * np.cos(np.pi * cycles * x_norm)


def step(x_norm, n_steps=4):
    """`n_steps` discrete composition plateaus instead of a smooth ramp."""
    return np.floor(x_norm * n_steps) / (n_steps - 1)


def sigmoid(x_norm, steepness=10, midpoint=0.5):
    """Mostly TI64, mostly IN718, with a sharper transition band in between."""
    return 1 / (1 + np.exp(-steepness * (x_norm - midpoint)))


COMPOSITION_FN = linear   # <- swap this to try a different composition profile


# ---- mixing rule: how each property blends between TI64 (f=0) and IN718
# (f=1) at a given composition fraction. Every property defaults to linear
# rule-of-mixtures (matches this script's original behavior) but can be
# overridden independently - solidus/liquidus are the most likely candidates
# once real data justifies it, since dissimilar Ti + Ni-Fe mixtures can dip
# below both endpoints (eutectic-like behavior) rather than sitting on a
# straight line between them (see CAVEAT above).

def linear_mix(a, b, f):
    return (1 - f) * a + f * b


MIXING_RULES = {
    'density': linear_mix,
    'solidus': linear_mix,
    'liquidus': linear_mix,
    'latent': linear_mix,
    'cp': linear_mix,
    'cond': linear_mix,
}

os.makedirs(GRAD_DIR, exist_ok=True)

with open(MESH_FILE) as f:
    lines = f.readlines()

part_start = next(i for i, l in enumerate(lines) if l.startswith('*PART'))
mat_starts = [i for i, l in enumerate(lines) if l.startswith('*MAT_THERMAL_ISOTROPIC_TD')]
node_start = next(i for i, l in enumerate(lines) if l.startswith('*NODE'))
elem_start = next(i for i, l in enumerate(lines) if l.startswith('*ELEMENT_SOLID'))
elem_end = next(i for i in range(elem_start + 1, len(lines)) if lines[i].startswith('*'))

# --- read node coordinates ---
nodes = {}
for l in lines[node_start + 1:]:
    if l.startswith('*'):
        break
    if l.startswith('$'):
        continue
    text = l.split()
    nodes[int(text[0])] = (float(text[1]), float(text[2]), float(text[3]))

# --- read wall elements, sanity-check TARGET_PID is the above-ground wall ---
elem_recs = []   # (eid, node_ids, cx)
target_z = []
for l in lines[elem_start + 1:elem_end]:
    if l.startswith('$'):
        continue
    text = l.split()
    eid, pid = int(text[0]), int(text[1])
    node_ids = [int(t) for t in text[2:10]]
    if pid == TARGET_PID:
        cx = sum(nodes[n][0] for n in node_ids) / 8.0
        cz = sum(nodes[n][2] for n in node_ids) / 8.0
        target_z.append(cz)
        elem_recs.append((eid, cx))

assert target_z and min(target_z) > -1.0, (
    "TARGET_PID={} looks like the base plate (z below ground), not the deposited "
    "wall - check the mesh before proceeding".format(TARGET_PID))

# --- one graded band per distinct element-column x position ---
columns = sorted(set(round(cx, 6) for _, cx in elem_recs))
n_bands = len(columns)
col_index = {cx: j for j, cx in enumerate(columns)}
x_min, x_max = columns[0], columns[-1]
x_norm = [(cx - x_min) / (x_max - x_min) for cx in columns]
fractions = [float(np.clip(COMPOSITION_FN(xn), 0.0, 1.0)) for xn in x_norm]
band_pid = [j + 2 for j in range(n_bands)]                # pid 1 reserved for base plate
cx_by_eid = dict(elem_recs)

# --- rewrite element lines: base plate pid 2 -> 1, wall pid 1 -> graded band pid ---
new_elem_lines = [lines[elem_start]]
for l in lines[elem_start + 1:elem_end]:
    if l.startswith('$'):
        new_elem_lines.append(l)
        continue
    text = l.split()
    eid, pid = int(text[0]), int(text[1])
    node_ids = [int(t) for t in text[2:10]]
    if pid == TARGET_PID:
        pid = band_pid[col_index[round(cx_by_eid[eid], 6)]]
    elif pid == BASE_PLATE_PID_OLD:
        pid = 1
    fields = [eid, pid] + node_ids
    new_elem_lines.append(''.join('%8d' % v for v in fields) + '\n')

lines = lines[:elem_start] + new_elem_lines + lines[elem_end:]

# --- re-find block boundaries after edits above didn't move them (elements are last) ---
part_end = next(i for i in range(part_start + 1, len(lines)) if lines[i].startswith('*'))
mat_block_bounds = []
for s in mat_starts:
    e = next(i for i in range(s + 1, len(lines)) if lines[i].startswith('*'))
    mat_block_bounds.append((s, e))
mat_first_start = mat_block_bounds[0][0]
mat_last_end = mat_block_bounds[-1][1]


def blend_curve(path_a, path_b, f, mix):
    a = np.loadtxt(path_a)
    b = np.loadtxt(path_b)
    grid = np.union1d(a[:, 0], b[:, 0])
    va = np.interp(grid, a[:, 0], a[:, 1])
    vb = np.interp(grid, b[:, 0], b[:, 1])
    return grid, mix(va, vb, f)


def write_curve(path, grid, values):
    with open(path, 'w') as f:
        for T, v in zip(grid, values):
            f.write('%-10g%10.6g\n' % (T, v))


# --- build new *PART and *MAT_THERMAL_ISOTROPIC_TD blocks, pid 1..n_bands+1 sequential ---
new_part_lines = ['*PART\n']
new_mat_lines = []

# pid 1: base plate, unchanged TI64
new_part_lines += [
    '$HWCOLOR COMPS       1       3\n',
    'Base_Plate\n',
    '         1         0         1\n',
]
new_mat_lines += [
    '*MAT_THERMAL_ISOTROPIC_TD\n',
    '$HMNAME MATS       1MATT1_1\n',
    '         1   %.5f   %d   %d   %d\n' % (TI64['density'], TI64['solidus'], TI64['liquidus'], TI64['latent']),
    TI64['cp'] + '\n',
    TI64['cond'] + '\n',
]

for j in range(n_bands):
    pid = band_pid[j]
    f = fractions[j]
    density = MIXING_RULES['density'](TI64['density'], IN718['density'], f)
    solidus = MIXING_RULES['solidus'](TI64['solidus'], IN718['solidus'], f)
    liquidus = MIXING_RULES['liquidus'](TI64['liquidus'], IN718['liquidus'], f)
    latent = MIXING_RULES['latent'](TI64['latent'], IN718['latent'], f)

    cp_grid, cp_vals = blend_curve(TI64['cp'], IN718['cp'], f, MIXING_RULES['cp'])
    cond_grid, cond_vals = blend_curve(TI64['cond'], IN718['cond'], f, MIXING_RULES['cond'])
    cp_path = os.path.join(GRAD_DIR, 'grad_%02d_cp.txt' % j)
    cond_path = os.path.join(GRAD_DIR, 'grad_%02d_cond.txt' % j)
    write_curve(cp_path, cp_grid, cp_vals)
    write_curve(cond_path, cond_grid, cond_vals)

    pct_in718 = round(f * 100)
    new_part_lines += [
        '$HWCOLOR COMPS      %3d       5\n' % pid,
        'Wall_Grad_%02d_TI%d_IN%d\n' % (j, 100 - pct_in718, pct_in718),
        '      %5d         0     %5d\n' % (pid, pid),
    ]
    new_mat_lines += [
        '*MAT_THERMAL_ISOTROPIC_TD\n',
        '$HMNAME MATS      %3dMATT1_%d\n' % (pid, pid),
        '      %5d   %.5f   %.1f   %.1f   %.1f\n' % (pid, density, solidus, liquidus, latent),
        cp_path + '\n',
        cond_path + '\n',
    ]

# --- splice: replace old *PART block and old *MAT blocks with the new ones ---
lines = (lines[:mat_first_start] + new_mat_lines + lines[mat_last_end:part_start]
          + new_part_lines + lines[part_end:])

with open(MESH_FILE, 'w') as f:
    f.writelines(lines)

print('Graded {} wall elements across {} bands (pid 2..{}), base plate -> pid 1, composition_fn={}'.format(
    len(elem_recs), n_bands, n_bands + 1, COMPOSITION_FN.__name__))
print('fraction range achieved: {:.3f} .. {:.3f}'.format(min(fractions), max(fractions)))
