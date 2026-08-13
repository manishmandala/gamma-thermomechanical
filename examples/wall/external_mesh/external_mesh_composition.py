# Generates sol_100_control_graded.k - a control file that imports the
# externally-provided mesh (sol_100.vtu) via *EXTERNAL_MESH and applies its
# OWN per-element composition (not a synthetic function like
# gradient_material_continuous_TI64_IN718.py/_cu.py use for the wall
# examples) as a *MAT_THERMAL_GRADED_TD / *ELEMENT_COMPOSITION blend.
#
# sol_100.vtu's cell_data['mat'] is a true 4-material composition simplex
# (50336 elements x 4 fractions, each row sums to 1.0) - NOT a 2-material
# blend. Per an old code comment from when this file was first explored
# (external_mesh_import_check.py), the 4 columns are, in order: IN625,
# SS316, HA25, MCrAlY. This ordering is INHERITED, not confirmed with the
# mesh's original author - flagged here so whoever revisits this knows it's
# an assumption.
#
# Columns 0 and 1 (IN625, SS316) are the only ones that are ever the
# dominant/argmax material for any element (verified: argmax is never 2 or 3,
# min col0+col1 across all elements is 0.65) - columns 2/3 are always minor
# admixtures (HA25/MCrAlY, plausibly thin coating layers). This script
# collapses the 4-way field down to a 2-way blend between columns 0 and 1,
# discarding 2/3 entirely, to reuse the solver's existing 2-material graded
# mechanism unmodified rather than extending it to true N-way blending.
#
# Neither IN625 nor SS316 has real property curves in this repo. IN718 (Ni-
# based superalloy, like IN625) and 1018 carbon steel (like SS316) are used
# as proxies instead - not physically exact, but not invented either:
# examples/PSED/wall.k already pairs exactly these two materials as a
# substrate/build pair, so this is an existing precedent, not a new guess.

import os
import meshio
import numpy as np

MESH_FILE = 'external_mesh/sol_100.vtu'
MESH_FILE_OUT = 'external_mesh/sol_100_control_graded.k'
PROP_DIR_IN718 = '../0_properties'
PROP_DIR_1018 = '../data/materials'
GRAD_ID = 201   # pid used for the (single) graded region; 1 is used by the pure-material control files

IN718 = dict(density=0.00819, solidus=1533, liquidus=1609, latent=270,
             cp=os.path.join(PROP_DIR_IN718, 'IN718_cp.txt'),
             cond=os.path.join(PROP_DIR_IN718, 'IN718_cond.txt'))
STEEL_1018 = dict(density=0.00787, solidus=1693, liquidus=1733, latent=270,
                   cp=os.path.join(PROP_DIR_1018, '1018_cp.txt'),
                   cond=os.path.join(PROP_DIR_1018, '1018_cond.txt'))

mesh = meshio.read(MESH_FILE)
mat = mesh.cell_data['mat'][0]   # (n_elements, 4), single hex block, confirmed 1:1 with *EXTERNAL_MESH import order
assert mat.shape[1] == 4, "expected a 4-material composition field, got shape {}".format(mat.shape)

frac_B = mat[:, 1] / (mat[:, 0] + mat[:, 1])   # fraction of SS316-proxy (1018); B endpoint in the graded material card
n_elements = mat.shape[0]

header = """*KEYWORD_ID
DED
*PARAMETER
Rboltz    5.6704E-14
*PARAMETER
Rambient       300.0
*PARAMETER
Rabszero         0.0
*TOOL_FILE
toolpath_off.crs
*GAUSS_LASER
250.0 0.5 0.4
*SCALAR_OUT
temp
solid_rate
theta_hist
nid_true
*CONTROL_TERMINATION
$$  ENDTIM    ENDCYC     DTMIN    ENDENG    ENDMAS
   20.0
*CONTROL_TIMESTEP
$$  DTINIT    TSSFAC      ISDO    TSLIMT     DT2MS      LCTM     EROD       E     MSIST
    5.0E-4       1.0
*CONTROL_SOLUTION
$$    SOLN
         0
*DATABASE_NODOUT
$$      DT    BINARY      LCUR      IOPT      DTHF     BINHF
    10.000         0
"""

mat_block = (
    '*MAT_THERMAL_GRADED_TD\n'
    '$HMNAME MATS     {}GRADED_IN718proxyIN625_1018proxySS316\n'.format(GRAD_ID) +
    '      %5d   %5d   %.5f   %.1f   %.1f   %.1f   %.5f   %.1f   %.1f   %.1f\n' % (
        GRAD_ID, 2, IN718['density'], IN718['solidus'], IN718['liquidus'], IN718['latent'],
        STEEL_1018['density'], STEEL_1018['solidus'], STEEL_1018['liquidus'], STEEL_1018['latent']) +
    IN718['cp'] + '\n' +
    IN718['cond'] + '\n' +
    STEEL_1018['cp'] + '\n' +
    STEEL_1018['cond'] + '\n'
)

part_block = (
    '*PART\n'
    '$HWCOLOR COMPS     {}       5\n'.format(GRAD_ID) +
    'Imported_Part_Graded_ExternalMesh_Composition\n' +
    '      %5d         0     %5d\n' % (GRAD_ID, GRAD_ID)
)

external_mesh_block = '*EXTERNAL_MESH\n{}       {}\n'.format(MESH_FILE, GRAD_ID)

load_node_set_block = """*LOAD_NODE_SET
$HMNAME LOADCOLS       1LoadNode_1
$	 Radiation
         2         4                 0.2
$	 convection
         2         3             0.00002
"""

comp_lines = ['*ELEMENT_COMPOSITION\n', '%10d\n' % GRAD_ID]
for eid in range(1, n_elements + 1):
    fb = frac_B[eid - 1]
    comp_lines.append('%20.8f%20.8f%20d\n' % (1.0 - fb, fb, eid))

with open(MESH_FILE_OUT, 'w') as f:
    f.write(header)
    f.write(mat_block)
    f.write(part_block)
    f.write(external_mesh_block)
    f.write(load_node_set_block)
    f.writelines(comp_lines)
    f.write('*END\n')

print('Wrote {} with {} elements (GRAD_ID={})'.format(MESH_FILE_OUT, n_elements, GRAD_ID))
print('frac_B (SS316-proxy fraction) range: {:.4f} .. {:.4f}, mean {:.4f}'.format(
    frac_B.min(), frac_B.max(), frac_B.mean()))
