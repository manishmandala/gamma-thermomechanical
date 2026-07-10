# Back-and-forth raster: each layer is deposited as TWO offset tracks -
# forward sweep (-7->7) lays down one strip (Y_FWD), backward sweep (7->-7)
# lays down a different, overlapping strip (Y_BACK) - so both directions
# visibly deposit new material instead of the return trip just reheating
# ground the wall is only laid down once. The head hops up to the next
# layer height (and realigns Y back to Y_FWD) while stationary and OFF.
# Columns: time  X  Y  Z  laser(0/1)
#
# The wall cross-section is 5 element-columns wide in Y (centers at
# -0.4,-0.2,0,0.2,0.4). Y_FWD/Y_BACK + the birth radius (0.3, see
# gen_birth.py) are chosen so forward covers columns {-0.4,-0.2,0} and
# backward covers {0,0.2,0.4} - full coverage, overlapping at the middle.
#
# NOTE: element "birth" (activation) times are pre-baked into thinwall.k's
# *DEFINE_CURVE from this exact schedule (see gen_birth.py). If you change
# SWEEP/DWELL/Z0/DZ/N/Y_FWD/Y_BACK here, you MUST re-run gen_birth.py
# afterwards or the laser and the deposited material will fall out of sync.

X_L, X_R = -7.0, 7.0
Y_FWD, Y_BACK = -0.2, 0.2    # offset tracks so each direction covers different columns
Z0, DZ, N = 0.4, 0.4, 10    # 10 layers, 0.4 apart -> matches thinwall.k (z 0.4 to 4.0)
SWEEP = 1.0                  # seconds per single-direction pass
DWELL = 10.0                 # cooling pause between layers (also doubles as the hop time)
HOP   = 0.1                  # time for the very first vertical move (avoids duplicate timestamps)

rows = []
t = 0.0
rows.append((t, X_L, Y_FWD, 0.0, 0))       # park at left, laser OFF, ground level

for i in range(N):
    z = Z0 + i*DZ
    dwell = HOP if i == 0 else DWELL
    t += dwell
    rows.append((t, X_L, Y_FWD, z, 0))    # hop up to this layer (and realign to Y_FWD), laser OFF
    t += SWEEP
    rows.append((t, X_R, Y_FWD, z, 1))    # sweep out on the Y_FWD track, laser ON (deposits)
    rows.append((t, X_R, Y_BACK, z, 0))   # instantaneous Y realignment to Y_BACK, laser OFF
    t += SWEEP
    rows.append((t, X_L, Y_BACK, z, 1))   # sweep back on the Y_BACK track, laser ON (deposits)

rows.append((t + DWELL, X_L, Y_FWD, 0.0, 0))    # final park, laser OFF

with open('toolpath.crs', 'w') as f:
    for (tt, x, y, z, l) in rows:
        f.write("{:16.8f}{:16.8f}{:15.8f}{:16.8f} {}\n".format(tt, x, y, z, l))

print("wrote toolpath.crs:", len(rows), "rows | layers =", N, "| total time = {:.2f}s".format(t))
