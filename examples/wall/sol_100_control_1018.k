*KEYWORD_ID
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
*MAT_THERMAL_ISOTROPIC_TD
$HMNAME MATS       1MATT1_1_1018_PROXY_FOR_SS316
         1   0.00787   1693   1733   270
../data/materials/1018_cp.txt
../data/materials/1018_cond.txt
*PART
$HWCOLOR COMPS       1       3
Imported_Part_Pure_1018
         1         0         1
*EXTERNAL_MESH
sol_100.vtu       1
*LOAD_NODE_SET
$HMNAME LOADCOLS       1LoadNode_1
$	 Radiation
         2         4                 0.2
$	 convection
         2         3             0.00002
*END
