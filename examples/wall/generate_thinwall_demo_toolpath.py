# Back-and-forth raster: EVERY single-direction pass deposits its own full,
# solid layer across the whole wall width, then the head steps up before the
# next pass. Forward (-7->7) lays down one layer; backward (7->-7) lays down
# the next layer above it - so both directions build new height, instead of
# the old scheme where a forward+backward pair shared one height (split by Y
# column, so the return trip was just finishing that same layer).
# Columns: time  X  Y  Z  laser(0/1)
#
# The wall cross-section is 5 element-columns wide in Y (centers at
# -0.4,-0.2,0,0.2,0.4). Since each pass now covers the FULL width, every pass
# travels a single centered Y track; the birth radius (see regen_birth.py)
# must be wide enough to reach the outer columns from center in one pass.
#
# NOTE: element "birth" (activation) times are pre-baked into thinwall.k's
# *DEFINE_CURVE from this exact schedule. If you change SWEEP/DWELL/Z0/
# LAYER_DZ/N_PASSES/Y_CENTER here, you MUST re-run regen_birth.py afterwards
# or the laser and the deposited material will fall out of sync.

X_L, X_R = -7.0, 7.0
Y_CENTER = 0.0               # single centered track - each pass covers the full width
Z0, LAYER_DZ, N_PASSES = 0.4, 0.2, 19   # 19 single-pass layers, 0.2 apart -> ends at 4.0, matches thinwall.k
SWEEP = 1.0                  # seconds per single-direction pass
DWELL = 1.0                  # cooling pause between layers (also doubles as the hop time) - shortened for fast testing
HOP   = 0.1                  # time for the very first vertical move (avoids duplicate timestamps)

rows = []
t = 0.0
rows.append((t, X_L, Y_CENTER, 0.0, 0))    # park at left, laser OFF, ground level

x_from, x_to = X_L, X_R
for i in range(N_PASSES):
    z = Z0 + i*LAYER_DZ
    dwell = HOP if i == 0 else DWELL
    t += dwell
    rows.append((t, x_from, Y_CENTER, z, 0))   # hop up to this layer's height, laser OFF
    t += SWEEP
    rows.append((t, x_to, Y_CENTER, z, 1))     # sweep across, laser ON - deposits this whole layer
    x_from, x_to = x_to, x_from                # flip direction for the next pass

rows.append((t + DWELL, x_from, Y_CENTER, 0.0, 0))    # final park, laser OFF

with open('toolpath.crs', 'w') as f:
    for (tt, x, y, z, l) in rows:
        f.write("{:16.8f}{:16.8f}{:15.8f}{:16.8f} {}\n".format(tt, x, y, z, l))

print("wrote toolpath.crs:", len(rows), "rows | passes =", N_PASSES, "| total time = {:.2f}s".format(t + DWELL))
