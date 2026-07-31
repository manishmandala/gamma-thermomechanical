# Generic single-field sweep-deck generator, reusable across future controlled
# studies (laser power, absorptivity - both live on the *GAUSS_LASER line).
# Scan-speed and hatch-spacing sweeps are NOT single-field edits (scan speed is
# implicit in toolpath.crs's time column; hatch spacing requires a different
# toolpath/geometry entirely) - out of scope for this script by design, not an
# oversight. See docstring at bottom for how a scan-speed sweep would differ.
#
# Guarantees isolation automatically: after writing each variant, diffs it
# line-by-line against the base deck and asserts EXACTLY the target line
# changed - the whole point of a controlled sweep is proving nothing else
# moved, not just assuming it.

import argparse
import os

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--base-k', required=True, help='source .k file (already graded/composition-baked)')
parser.add_argument('--keyword', default='*GAUSS_LASER',
                     help='keyword whose following data line contains the target field (default *GAUSS_LASER)')
parser.add_argument('--field-index', type=int, default=0,
                     help='0-indexed whitespace-separated field on the keyword\'s data line '
                          '(*GAUSS_LASER: 0=power, 1=beam radius, 2=absorptivity)')
parser.add_argument('--values', required=True, nargs='+', type=float, help='sweep values, e.g. 400 600 800 1000 1200')
parser.add_argument('--label', default='LP', help='filename tag prefix for each generated variant (default LP)')
parser.add_argument('--out-dir', required=True)
args = parser.parse_args()
os.makedirs(args.out_dir, exist_ok=True)

with open(args.base_k) as f:
    base_lines = f.readlines()

kw_idx = next(i for i, l in enumerate(base_lines) if l.strip() == args.keyword)
data_idx = kw_idx + 1
base_fields = base_lines[data_idx].split()
base_value = float(base_fields[args.field_index])
print('base {} line: {}'.format(args.keyword, base_lines[data_idx].strip()))
print('base field[{}] = {}'.format(args.field_index, base_value))

stem = os.path.splitext(os.path.basename(args.base_k))[0]
generated = []
for v in args.values:
    fields = list(base_fields)
    fields[args.field_index] = repr(v) if v != int(v) else str(v)
    if float(fields[args.field_index]) == v:
        fields[args.field_index] = ('{:g}'.format(v))
    new_line = ' '.join(fields) + '\n'

    lines = list(base_lines)
    lines[data_idx] = new_line
    out_name = '{}_{}{:g}.k'.format(stem, args.label, v)
    out_path = os.path.join(args.out_dir, out_name)
    with open(out_path, 'w') as f:
        f.writelines(lines)

    # ---- isolation check: exactly one line may differ from the base ----
    diff_lines = [i for i in range(len(base_lines)) if lines[i] != base_lines[i]]
    if diff_lines != [data_idx]:
        raise SystemExit('ISOLATION VIOLATION for value {}: expected only line {} to change, '
                          'but lines {} differ'.format(v, data_idx, diff_lines))
    print('wrote {} (value={:g}, isolation check passed: only line {} differs)'.format(
        out_path, v, data_idx))
    generated.append(out_path)

print('\n{} decks generated in {}, each verified to differ from the base in exactly one line '
      '({})'.format(len(generated), args.out_dir, args.keyword))

# Note on other sweep types for future reuse:
#  - absorptivity sweep: same script, --field-index 2
#  - scan-speed sweep: NOT a field edit - requires rescaling toolpath.crs's
#    time column (t' = t / speed_factor, keeping x/y/z/laser_state
#    unchanged) so waypoints are reached faster/slower while covering the
#    identical spatial path. Needs a separate small script operating on the
#    .crs file, not this one.
#  - hatch-spacing sweep: requires a genuinely different toolpath (and
#    possibly different mesh/geometry) - not reproducible by editing an
#    existing dataset's files at all.
