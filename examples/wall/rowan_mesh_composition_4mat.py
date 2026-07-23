# Generates sol_100_control_graded_4mat.k - the true N=4 counterpart of
# rowan_mesh_composition.py. That earlier script collapsed Rowan's real
# 4-material composition simplex (sol_100.vtu's cell_data['mat'], columns
# IN625/SS316/HA25/MCrAlY per the same inherited-not-confirmed ordering) down
# to a 2-material blend, discarding columns 2/3 entirely, because gamma.py's
# graded-material solver only supported exactly 2 endpoints at the time.
#
# Now that gamma.py supports N materials, this writes all 4 columns straight
# through - no collapsing, no discarding. Same IN718/1018 proxies as before
# for columns 0/1 (IN625/SS316). Columns 2/3 (HA25, MCrAlY) have no property
# curves anywhere in this repo, so they get placeholder proxies from the
# existing property library: CPTi (commercially-pure titanium) for HA25,
# Al (aluminum) for MCrAlY. These are NOT physically matched to the real
# alloys - they're picked only to be numerically distinct from each other and
# from IN718/1018, so a comparison run can prove all 4 columns are actually
# being read and blended rather than silently defaulting to one material.
# Since HA25/MCrAlY are always minor admixtures in this mesh (never the
# dominant material in any element - min(col0+col1)=0.65 across all elements,
# per rowan_mesh_composition.py's original analysis), the physical inaccuracy
# of these two proxies has limited impact on the overall thermal result.

import os
import meshio
import numpy as np

MESH_FILE = 'sol_100.vtu'
MESH_FILE_OUT = 'sol_100_control_graded_4mat.k'
PROP_DIR_IN718 = '../0_properties'
PROP_DIR_1018 = '../data/materials'
PROP_DIR_CPTI = '../0_properties'
PROP_DIR_AL = '../0_properties'
GRAD_ID = 202   # pid for this graded region; 201 already used by the 2-mat collapsed version

IN718 = dict(density=0.00819, solidus=1533, liquidus=1609, latent=270,
             cp=os.path.join(PROP_DIR_IN718, 'IN718_cp.txt'),
             cond=os.path.join(PROP_DIR_IN718, 'IN718_cond.txt'))
STEEL_1018 = dict(density=0.00787, solidus=1693, liquidus=1733, latent=270,
                   cp=os.path.join(PROP_DIR_1018, '1018_cp.txt'),
                   cond=os.path.join(PROP_DIR_1018, '1018_cond.txt'))
CPTI = dict(density=0.00451, solidus=1933, liquidus=1941, latent=295,
            cp=os.path.join(PROP_DIR_CPTI, 'CPTi_cp.txt'),
            cond=os.path.join(PROP_DIR_CPTI, 'CPTi_cond.txt'))
AL = dict(density=0.00270, solidus=930, liquidus=933, latent=397,
          cp=os.path.join(PROP_DIR_AL, 'Al_cp.txt'),
          cond=os.path.join(PROP_DIR_AL, 'Al_cond.txt'))

# order matches sol_100.vtu's cell_data['mat'] columns: IN625, SS316, HA25, MCrAlY
MATERIALS = [IN718, STEEL_1018, CPTI, AL]
N_MAT = len(MATERIALS)

mesh = meshio.read(MESH_FILE)
mat = mesh.cell_data['mat'][0]   # (n_elements, 4), single hex block, confirmed 1:1 with *EXTERNAL_MESH import order
assert mat.shape[1] == N_MAT, "expected a {}-material composition field, got shape {}".format(N_MAT, mat.shape)

# normalize defensively - sol_100.vtu's rows already sum to ~1.0, but this
# guarantees it regardless of floating-point noise in the source file
row_sums = mat.sum(axis=1, keepdims=True)
fractions = mat / row_sums
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

mat_data_line = '      %5d   %5d' % (GRAD_ID, N_MAT)
for m in MATERIALS:
    mat_data_line += '   %.5f   %.1f   %.1f   %.1f' % (m['density'], m['solidus'], m['liquidus'], m['latent'])
mat_data_line += '\n'

mat_block = (
    '*MAT_THERMAL_GRADED_TD\n'
    '$HMNAME MATS     {}GRADED_4MAT_IN625_SS316_HA25_MCrAlY\n'.format(GRAD_ID) +
    mat_data_line +
    ''.join(m['cp'] + '\n' + m['cond'] + '\n' for m in MATERIALS)
)

part_block = (
    '*PART\n'
    '$HWCOLOR COMPS     {}       5\n'.format(GRAD_ID) +
    'Imported_Part_Graded_Rowan_4Material_Composition\n' +
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
    row = fractions[eid - 1]
    comp_lines.append(''.join('%20.8f' % v for v in row) + '%20d\n' % eid)

with open(MESH_FILE_OUT, 'w') as f:
    f.write(header)
    f.write(mat_block)
    f.write(part_block)
    f.write(external_mesh_block)
    f.write(load_node_set_block)
    f.writelines(comp_lines)
    f.write('*END\n')

print('Wrote {} with {} elements (GRAD_ID={}, N_MAT={})'.format(MESH_FILE_OUT, n_elements, GRAD_ID, N_MAT))
for i, m in enumerate(['IN625proxy(IN718)', 'SS316proxy(1018)', 'HA25proxy(CPTi)', 'MCrAlYproxy(Al)']):
    col = fractions[:, i]
    print('  col {} {}: range {:.4f} .. {:.4f}, mean {:.4f}'.format(i, m, col.min(), col.max(), col.mean()))
