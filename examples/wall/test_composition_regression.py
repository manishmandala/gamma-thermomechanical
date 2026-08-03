# Regression test for the compute_centroid / coordinate_function /
# composition_function refactor (composition_lib.py). Proves the refactored
# pipeline produces byte-identical *ELEMENT_COMPOSITION output to what was
# committed BEFORE the refactor, for every script that generates composition
# data:
#
#   gradient_material_continuous_TI64_IN718.py     -> thinwall_graded.k    (TI64/IN718, sinusoidal)
#   gradient_material_continuous_TI64_Cu.py  -> thinwall_graded_cu.k (TI64/Cu, sinusoidal)
#   prepare_composition.py              -> the 4 endpoint-validation .k decks (constant mode)
#
# "Byte-identical to git HEAD" is a strong, unambiguous baseline: it doesn't
# rely on a hand-copied snapshot going stale, and it fails loudly (with a
# real diff) the moment anyone changes composition_lib.py's math in a way
# that actually changes output - which is exactly the failure mode this test
# exists to catch. Run this after any change to composition_lib.py or any
# script that imports it, before trusting new output.
#
# No pytest in this project (checked - not installed, no other test_*.py
# convention beyond plain print-and-exit-code scripts like
# compare_endpoint_validation.py) - this follows that same pattern.

import filecmp
import os
import subprocess
import sys
import tempfile

WALL_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(WALL_DIR, '..', '..'))

results = []  # (label, passed, detail)


def check_generator_script(script, output_filename):
    """Runs a generator script (writes output_filename in WALL_DIR as a
    side effect) and diffs the result against what's committed at HEAD."""
    before = subprocess.run(['git', 'show', 'HEAD:examples/wall/{}'.format(output_filename)],
                             cwd=REPO_ROOT, capture_output=True, text=True)
    if before.returncode != 0:
        results.append((script, False, '{} is not committed at HEAD - nothing to regress against'.format(
            output_filename)))
        return

    proc = subprocess.run([sys.executable, script], cwd=WALL_DIR, capture_output=True, text=True)
    if proc.returncode != 0:
        results.append((script, False, 'script failed: {}'.format(proc.stderr.strip()[-500:])))
        return

    with open(os.path.join(WALL_DIR, output_filename)) as f:
        after = f.read()
    if after == before.stdout:
        results.append((script, True, '{} byte-identical to git HEAD'.format(output_filename)))
    else:
        results.append((script, False, '{} differs from git HEAD - composition_lib.py changed behavior'.format(
            output_filename)))


def check_prepare_composition(dataset_rel, kfile, mode):
    """Regenerates one prepare_composition.py variant into a temp dir and
    diffs it against the version committed at HEAD."""
    committed_rel = 'examples/incoming_dataset/{}/endpoint_validation/{}_{}.k'.format(
        os.path.basename(dataset_rel), kfile.split('.')[0], mode)
    before = subprocess.run(['git', 'show', 'HEAD:{}'.format(committed_rel)],
                             cwd=REPO_ROOT, capture_output=True, text=True)
    if before.returncode != 0:
        results.append(('prepare_composition.py --mode {}'.format(mode), False,
                         '{} is not committed at HEAD - nothing to regress against'.format(committed_rel)))
        return

    with tempfile.TemporaryDirectory() as tmp:
        out_path = os.path.join(tmp, 'out.k')
        proc = subprocess.run(
            [sys.executable, 'prepare_composition.py', '--dataset', dataset_rel, '--kfile', kfile,
             '--mode', mode, '--out', out_path],
            cwd=WALL_DIR, capture_output=True, text=True)
        if proc.returncode != 0:
            results.append(('prepare_composition.py --mode {}'.format(mode), False,
                             'script failed: {}'.format(proc.stderr.strip()[-500:])))
            return
        with open(out_path) as f:
            after = f.read()

    if after == before.stdout:
        results.append(('prepare_composition.py --mode {}'.format(mode), True,
                         '{} byte-identical to git HEAD'.format(mode)))
    else:
        results.append(('prepare_composition.py --mode {}'.format(mode), False,
                         '{} differs from git HEAD'.format(mode)))


check_generator_script('gradient_material_continuous_TI64_IN718.py', 'thinwall_graded.k')
check_generator_script('gradient_material_continuous_TI64_Cu.py', 'thinwall_graded_cu.k')

for mode in ('pure_inconel', 'pure_titanium', 'constant_inconel', 'constant_titanium'):
    check_prepare_composition('../incoming_dataset/part002_LP800_SSp10_H2.24_SSt3_LH0.9', '2.k', mode)

print('Composition regression test\n' + '=' * 60)
all_passed = True
for label, passed, detail in results:
    status = 'PASS' if passed else 'FAIL'
    print('[{}] {} - {}'.format(status, label, detail))
    all_passed = all_passed and passed

print('=' * 60)
print('RESULT: {}'.format('ALL PASS' if all_passed else 'FAILURES PRESENT'))
sys.exit(0 if all_passed else 1)
