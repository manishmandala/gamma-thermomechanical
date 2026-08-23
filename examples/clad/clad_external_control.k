*KEYWORD_ID
DED
*PARAMETER
Rboltz    5.6704E-14
*PARAMETER
Rambient       300.0
*PARAMETER
Rabszero         0.0
*TOOL_FILE
toolpath.crs
*GAUSS_LASER
600 0.8 0.4
*SCALAR_OUT
temp
solid_rate
theta_hist
nid_true
*CONTROL_TERMINATION
$$  ENDTIM    ENDCYC     DTMIN    ENDENG    ENDMAS
   60
*CONTROL_TIMESTEP
$$  DTINIT    TSSFAC      ISDO    TSLIMT     DT2MS      LCTM     EROD       E     MSIST
    1.0E-2       1.0
*CONTROL_SOLUTION
$$    SOLN
         0
*DATABASE_NODOUT
$$      DT    BINARY      LCUR      IOPT      DTHF     BINHF
    10.000         0
*MAT_THERMAL_ISOTROPIC_TD
$HMNAME MATS       1MATT1_1
         1   0.00819   1533   1609     270
../0_properties/IN718_cp.txt
../0_properties/IN718_cond.txt
*PART
$HWCOLOR COMPS       1       3
Imported_Clad
         1         0         1
*EXTERNAL_MESH
clad_export.vtu       1
*LOAD_NODE_SET
$HMNAME LOADCOLS       1LoadNode_1
$	 Radiation
         2         4                 0.2
$	 convection
         2         3             0.00002
*END
