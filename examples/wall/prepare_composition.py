# Generates one of four material variants of a native .k file (real
# *NODE/*ELEMENT_SOLID geometry, exactly 2 plain materials, no existing
# graded region) for endpoint validation of the graded-material solver path:
#
#   pure_inconel       - build region -> ordinary IN718 material (legacy path)
#   pure_titanium      - build region -> ordinary TI64 material (legacy path)
#   constant_inconel   - build region -> *MAT_THERMAL_GRADED_TD/*ELEMENT_COMPOSITION,
#                         every element forced to composition [1.0, 0.0] (100% IN718)
#   constant_titanium  - same graded path, forced to [0.0, 1.0] (100% TI64)
#
# The point: constant_inconel should reproduce pure_inconel exactly (and
# constant_titanium / pure_titanium likewise), proving the new graded-blend
# code path collapses to the old plain-material path at its endpoints. This
# is the only thing this script builds - position-dependent composition
# (linear/toolpath_arc_length/etc.) is a deliberately separate, later step.
#
# WHICH material ID is "the build region" is determined by BIRTH TIME, not
# by the file's own *PART names - this project has already hit this exact
# trap once before (see gradient_material_continuous_cu.py's header note on
# thinwall.k) and it recurred independently on the new OneDrive datasets
# (part002's *PART block calls pid 1 "Substrate" and pid 2 "Build", but pid
# 1 is the one with real, progressive birth times - i.e. actually the
# deposited region - and pid 2 is birth=0 everywhere - i.e. actually the
# fixed substrate). Trusting the text label would silently vary the wrong
# material. This script auto-detects instead: whichever ordinary material ID
# has EVERY element born at t=0 is the substrate and is left untouched;
# the other one is the build region and gets rewritten.
#
# Substrate is never touched in any mode - this isolates the thermal effect
# of build-region composition alone, per the endpoint-validation goal.

import argparse
import os
import cupy as cp
cp.cuda.Device(0).use()
from gamma.simulator.gamma import domain_mgr

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--dataset', required=True, help='dataset folder, e.g. ../incoming_dataset/part002_.../')
parser.add_argument('--kfile', required=True, help='input .k filename within --dataset, e.g. 2.k')
parser.add_argument('--toolpath', default='toolpath.crs', help='toolpath filename within --dataset')
parser.add_argument('--mode', required=True,
                     choices=['pure_inconel', 'pure_titanium', 'constant_inconel', 'constant_titanium'])
parser.add_argument('--out', required=True, help='output .k path - must not already exist')
args = parser.parse_args()

if os.path.exists(args.out):
    raise SystemExit('refusing to overwrite existing file: {}'.format(args.out))

src_path = os.path.join(args.dataset, args.kfile)

# --- real parse, read-only, to find build-vs-substrate by birth time ---
d = domain_mgr(filename=src_path, toolpathdir=os.path.join(args.dataset, args.toolpath),
                input_data_dir=args.dataset, verbose=False)
assert len(d.mat_thermal) == 2, 'expected exactly 2 ordinary materials, found {}'.format(len(d.mat_thermal))
assert len(d.mat_graded) == 0, 'source file already has a graded region - not the expected clean 2-material input'

mat_ids = [m[0] for m in d.mat_thermal]
substrate_candidates = [m for m in mat_ids if bool((d.element_birth[d.element_mat == m] == 0).all())]
assert len(substrate_candidates) == 1, (
    'expected exactly one material with all-zero birth (the substrate), found {}: {}'.format(
        len(substrate_candidates), substrate_candidates))
substrate_pid = substrate_candidates[0]
build_pid = [m for m in mat_ids if m != substrate_pid][0]
print('auto-detected by birth time: substrate=matID {}, build region=matID {} ({} elements)'.format(
    substrate_pid, build_pid, int((d.element_mat == build_pid).sum())))

# --- pull each material's real property values + curve-file paths straight out of the
# source file's own two blocks (density is a robust discriminator: IN718 ~0.00819,
# TI64 ~0.00440 - no path/property values are hardcoded here) ---
with open(src_path) as f:
    lines = f.readlines()

elem_start = next(i for i, l in enumerate(lines) if l.startswith('*ELEMENT_SOLID'))
elem_end = next(i for i in range(elem_start + 1, len(lines)) if lines[i].startswith('*'))
build_element_ids = []
for l in lines[elem_start + 1:elem_end]:
    if l.startswith('$'):
        continue
    text = l.split()
    if int(text[1]) == build_pid:
        build_element_ids.append(int(text[0]))
assert len(build_element_ids) == int((d.element_mat == build_pid).sum()), (
    'element count mismatch between text parse ({}) and domain_mgr ({})'.format(
        len(build_element_ids), int((d.element_mat == build_pid).sum())))

mat_starts = [i for i, l in enumerate(lines) if l.startswith('*MAT_THERMAL_ISOTROPIC_TD')]
block_by_pid = {}
for s in mat_starts:
    data_line = lines[s + 2].split()
    pid = int(data_line[0])
    block_by_pid[pid] = dict(
        density=float(data_line[1]), solidus=float(data_line[2]),
        liquidus=float(data_line[3]), latent=float(data_line[4]),
        cp=lines[s + 3].strip(), cond=lines[s + 4].strip(),
    )

IN718 = min(block_by_pid.values(), key=lambda b: abs(b['density'] - 0.00819))
TI64 = min(block_by_pid.values(), key=lambda b: abs(b['density'] - 0.00440))
assert IN718 is not TI64, 'could not distinguish IN718 vs TI64 by density in {}'.format(src_path)

build_start = next(s for s in mat_starts if int(lines[s + 2].split()[0]) == build_pid)
build_end = next(i for i in range(build_start + 1, len(lines)) if lines[i].startswith('*'))

if args.mode in ('pure_inconel', 'pure_titanium'):
    props = IN718 if args.mode == 'pure_inconel' else TI64
    new_block = [
        lines[build_start],
        '$HMNAME MATS     {}MATT1_{}\n'.format(build_pid, build_pid),
        '      %5d   %.5f   %.1f   %.1f   %.1f\n' % (
            build_pid, props['density'], props['solidus'], props['liquidus'], props['latent']),
        props['cp'] + '\n',
        props['cond'] + '\n',
    ]
    lines = lines[:build_start] + new_block + lines[build_end:]

else:  # constant_inconel / constant_titanium
    frac = (1.0, 0.0) if args.mode == 'constant_inconel' else (0.0, 1.0)
    new_mat_block = [
        '*MAT_THERMAL_GRADED_TD\n',
        '$HMNAME MATS     {}GRADED_CONSTANT\n'.format(build_pid),
        '      %5d   %5d   %.5f   %.1f   %.1f   %.1f   %.5f   %.1f   %.1f   %.1f\n' % (
            build_pid, 2, IN718['density'], IN718['solidus'], IN718['liquidus'], IN718['latent'],
            TI64['density'], TI64['solidus'], TI64['liquidus'], TI64['latent']),
        IN718['cp'] + '\n', IN718['cond'] + '\n',
        TI64['cp'] + '\n', TI64['cond'] + '\n',
    ]
    lines = lines[:build_start] + new_mat_block + lines[build_end:]

    # *ELEMENT_COMPOSITION MUST be inserted after *DEFINE_CURVE's own block (before
    # *NODE) - gamma.py's second-pass reader tests *DEFINE_CURVE / *ELEMENT_COMPOSITION
    # / *END as sequential `if`s sharing one `line` variable, not `elif`; putting this
    # block before *DEFINE_CURVE would make *DEFINE_CURVE's inner loop terminate on
    # THIS block's own header, silently skipping every birth time in the file. Same
    # gotcha documented in gradient_material_continuous_cu.py. build_element_ids was
    # captured before any line-shifting edits above, so it's still valid here.
    node_start = next(i for i, l in enumerate(lines) if l.startswith('*NODE'))
    comp_lines = ['*ELEMENT_COMPOSITION\n', '%10d\n' % build_pid]
    for eid in build_element_ids:
        comp_lines.append('%20.8f%20.8f%20d\n' % (frac[0], frac[1], eid))
    lines = lines[:node_start] + comp_lines + lines[node_start:]

with open(args.out, 'w') as f:
    f.writelines(lines)

print('wrote {} (mode={})'.format(args.out, args.mode))
