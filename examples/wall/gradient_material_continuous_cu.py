# Grades the DEPOSITED WALL (laser-built material) as a CONTINUOUS TI64/Cu
# blend across the wall's x-z face (travel direction and build height) - one
# composition value per element (not per discrete band), evaluated straight
# from COMPOSITION_FN at that element's own centroid (x, z).
# Properties are blended live, at runtime, by gamma.py itself (see
# *MAT_THERMAL_GRADED_TD / *ELEMENT_COMPOSITION handling in
# src/gamma/simulator/gamma.py) - this script only writes the composition
# field, it does not pre-blend curves or bake discrete materials the way
# gradient_material.py (the 70-band predecessor) does. No fixed step count:
# however many wall elements the mesh has, that's how many distinct
# compositions you get - nothing needs to divide evenly into anything.
#
# Reads thinwall_clean.k - the pre-gradient mesh (2 plain TI64 materials,
# matches commit 3776a6d's thinwall.k before gradient_material.py ever ran) -
# not the committed 70-band thinwall.k, and writes to a separate output file
# so both variants stay available for comparison.
#
# Only touches the *ELEMENT_SOLID pid column and adds *MAT_THERMAL_GRADED_TD
# / *ELEMENT_COMPOSITION blocks - node geometry, birth times, and the
# toolpath are untouched, so the laser schedule is unaffected.
#
# NOTE: thinwall.k's *PART block names are misleading - the part LABELED
# "Substrate" (pid 1) is actually the above-ground deposited wall (z > 0),
# and the part labeled "Build" (pid 2) is actually the below-ground base
# plate (z < 0). TARGET_PID below is chosen by mesh geometry, asserted
# before touching anything so a relabeling can't silently break this again.
#
# TI64 endpoint properties match examples/clad/clad.k and gradient_material.py.
# Cu endpoint is a much more drastic thermal contrast than IN718 (~60x higher
# conductivity, ~550K lower melting point) - used to make the composition
# gradient's effect on temperature obvious, since the TI64-vs-IN718 pair (both
# poor conductors, similar melting points) barely differed. Not a
# metallurgically realistic DED pairing - a numerical demo of the blending
# mechanism, not a claim about real alloy compatibility.
#
# CAVEAT (same as gradient_material.py): blending is linear rule-of-mixtures
# by default - this script/format fixes the *quantization* problem (no more
# discrete bands), not the *mixing-model* problem. Solidus/liquidus in
# particular are still an approximation; see gradient_material.py's header
# for the full caveat.

import os
import numpy as np

MESH_FILE_IN = 'thinwall_clean.k'        # pre-gradient mesh (2 plain materials)
MESH_FILE_OUT = 'thinwall_graded_cu.k'   # new file; thinwall_graded.k (TI64/IN718) is left untouched
PROP_DIR = '../0_properties'
TARGET_PID = 1   # the deposited wall (despite the *PART block calling it "Substrate")
BASE_PLATE_PID_OLD = 2
GRAD_ID = 102     # pid used for graded wall elements; 101 is already taken by the TI64/IN718 variant

TI64 = dict(density=0.00440, solidus=1878, liquidus=1928, latent=286,
            cp=os.path.join(PROP_DIR, 'TI64_cp.txt'),
            cond=os.path.join(PROP_DIR, 'TI64_cond.txt'))
Cu = dict(density=0.00896, solidus=1353, liquidus=1358, latent=205,
          cp=os.path.join(PROP_DIR, 'Cu_cp.txt'),
          cond=os.path.join(PROP_DIR, 'Cu_cond.txt'))


# ---- composition profile: fraction of Cu as a function of normalized
# position ---- x_norm runs 0 (TI64 end of the wall) -> 1 (Cu end) along
# the build/travel direction (sideways); z_norm runs 0 -> 1 up the build
# height (40 deposited layers, z = 0.05 .. 3.95, upwards). Every preset takes
# both so they share one call signature - linear/step/sigmoid ignore z_norm
# (1D profiles, identical at every layer); sinusoidal is a true 2D ripple -
# both axes are directly visible on the wall's exterior face without needing
# to slice through the (thin) y thickness.

def linear(x_norm, z_norm):
    return x_norm


def sinusoidal(x_norm, z_norm, cycles_x=4.0, cycles_z=1.5):
    # additive x-wave + z-wave: each axis contributes its own independent,
    # always-present variation (a multiplicative cos(x)*cos(z) form was
    # tried first, but a product term means the z-factor can only scale
    # the x-wave's amplitude down - it never shows up as its own visible
    # effect, so the wall still reads as "x-only" wherever z's factor is
    # near 1). z gets the LOW cycle count (dominant, bottom-to-top sweep)
    # and x the HIGH one (secondary ripple): the mesh only has 40 element-
    # rows through z vs 70 columns along x, so a high cycle count on the
    # coarser z axis reads as choppy/blocky under flat per-cell shading -
    # x has plenty of resolution to carry the finer variation smoothly.
    return 0.5 + 0.25 * np.sin(2*np.pi * cycles_x * x_norm) + 0.25 * np.sin(2*np.pi * cycles_z * z_norm)


def step(x_norm, z_norm, n_steps=4):
    return np.floor(x_norm * n_steps) / (n_steps - 1)


def sigmoid(x_norm, z_norm, steepness=10, midpoint=0.5):
    return 1 / (1 + np.exp(-steepness * (x_norm - midpoint)))


COMPOSITION_FN = sinusoidal   # <- swap this to try a different composition profile

with open(MESH_FILE_IN) as f:
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
elem_recs = []   # (eid, cx, cz)
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
        elem_recs.append((eid, cx, cz))

assert target_z and min(target_z) > -1.0, (
    "TARGET_PID={} looks like the base plate (z below ground), not the deposited "
    "wall - check the mesh before proceeding".format(TARGET_PID))

assert GRAD_ID not in {int(lines[s+2].split()[0]) for s in mat_starts}, (
    "GRAD_ID={} collides with an existing ordinary material ID in {} - pick a different GRAD_ID".format(
        GRAD_ID, MESH_FILE_IN))

# --- composition fraction per element, evaluated directly at that element's own (x, z) - no bands ---
x_vals = [cx for _, cx, _ in elem_recs]
z_vals = [cz for _, _, cz in elem_recs]
x_min, x_max = min(x_vals), max(x_vals)
z_min, z_max = min(z_vals), max(z_vals)
fractions = {eid: float(np.clip(COMPOSITION_FN((cx - x_min) / (x_max - x_min),
                                                (cz - z_min) / (z_max - z_min)), 0.0, 1.0))
             for eid, cx, cz in elem_recs}

# --- rewrite element lines: base plate pid 2 -> 1, wall pid 1 -> GRAD_ID ---
new_elem_lines = [lines[elem_start]]
for l in lines[elem_start + 1:elem_end]:
    if l.startswith('$'):
        new_elem_lines.append(l)
        continue
    text = l.split()
    eid, pid = int(text[0]), int(text[1])
    node_ids = [int(t) for t in text[2:10]]
    if pid == TARGET_PID:
        pid = GRAD_ID
    elif pid == BASE_PLATE_PID_OLD:
        pid = 1
    fields = [eid, pid] + node_ids
    new_elem_lines.append(''.join('%8d' % v for v in fields) + '\n')

lines = lines[:elem_start] + new_elem_lines + lines[elem_end:]

# --- drop every ordinary *MAT_THERMAL_ISOTROPIC_TD block except the first ---
# (base plate elements were remapped to pid 1 above, and the first block's data
# line already reads pid 1 unchanged - keeping it as-is and dropping the rest
# avoids leaving a dead, unreferenced material sitting in the file)
assert lines[mat_starts[0] + 2].split()[0] == '1', (
    "expected the first *MAT_THERMAL_ISOTROPIC_TD block to be pid 1 already - "
    "check {} hasn't changed shape".format(MESH_FILE_IN))
mat_first_end = next(i for i in range(mat_starts[0] + 1, len(lines)) if lines[i].startswith('*'))
mat_last_end = next(i for i in range(mat_starts[-1] + 1, len(lines)) if lines[i].startswith('*'))
lines = lines[:mat_first_end] + lines[mat_last_end:]
part_start = next(i for i, l in enumerate(lines) if l.startswith('*PART'))

# --- new *MAT_THERMAL_GRADED_TD block, appended right after the surviving pid-1 block ---
# N-material format: gradID, N, then 4 properties per material (N=2 here),
# followed by one (Cp_file, Cond_file) line pair per material.
new_mat_lines = [
    '*MAT_THERMAL_GRADED_TD\n',
    '$HMNAME MATS     {}GRADED_TI64_CU\n'.format(GRAD_ID),
    '      %5d   %5d   %.5f   %.1f   %.1f   %.1f   %.5f   %.1f   %.1f   %.1f\n' % (
        GRAD_ID, 2, TI64['density'], TI64['solidus'], TI64['liquidus'], TI64['latent'],
        Cu['density'], Cu['solidus'], Cu['liquidus'], Cu['latent']),
    TI64['cp'] + '\n',
    TI64['cond'] + '\n',
    Cu['cp'] + '\n',
    Cu['cond'] + '\n',
]
lines = lines[:mat_first_end] + new_mat_lines + lines[mat_first_end:]

# --- new *PART entry for the graded region (cosmetic only - gamma.py never reads *PART) ---
part_end = next(i for i in range(part_start + 1, len(lines)) if lines[i].startswith('*'))
new_part_lines = [
    '$HWCOLOR COMPS     {}       5\n'.format(GRAD_ID),
    'Wall_Graded_TI64_CU\n',
    '      %5d         0     %5d\n' % (GRAD_ID, GRAD_ID),
]
lines = lines[:part_end] + new_part_lines + lines[part_end:]

# --- *ELEMENT_COMPOSITION block: one fraction per wall element ---
# MUST be inserted AFTER *DEFINE_CURVE's own data block, not before: gamma.py's
# second-pass reader tests '*DEFINE_CURVE' / '*ELEMENT_COMPOSITION' / '*END' as
# sequential `if`s sharing one `line` variable (not `elif`). If this block sat
# before *DEFINE_CURVE, *DEFINE_CURVE's inner loop would terminate by reading
# this block's own header line, and the '*DEFINE_CURVE' check (which already
# ran earlier in that same outer iteration, against the OLD line value) would
# never re-fire - silently skipping every birth time in the file. Inserting
# after *DEFINE_CURVE's block (right before *NODE, where it already naturally
# terminates today) avoids that entirely: *NODE isn't checked for anything in
# this pass, so replacing it with *ELEMENT_COMPOSITION here is safe.
node_start = next(i for i, l in enumerate(lines) if l.startswith('*NODE'))
comp_lines = ['*ELEMENT_COMPOSITION\n', '%10d\n' % GRAD_ID]
for eid, _, _ in elem_recs:
    frac_B = fractions[eid]
    comp_lines.append('%20.8f%20.8f%20d\n' % (1.0 - frac_B, frac_B, eid))
lines = lines[:node_start] + comp_lines + lines[node_start:]

with open(MESH_FILE_OUT, 'w') as f:
    f.writelines(lines)

frac_vals = list(fractions.values())
print('Wrote {} with {} continuously-graded wall elements (GRAD_ID={}), composition_fn={}'.format(
    MESH_FILE_OUT, len(elem_recs), GRAD_ID, COMPOSITION_FN.__name__))
print('fraction range achieved: {:.4f} .. {:.4f}, {} distinct values'.format(
    min(frac_vals), max(frac_vals), len(set(frac_vals))))
