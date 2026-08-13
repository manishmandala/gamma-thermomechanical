# Thin orchestrator around run_wall.py for a set of already-generated sweep
# decks (see generate_parameter_sweep.py). Runs every deck in --decks-dir
# with IDENTICAL --stop-fraction/--n-frames/--toolpath/--input-data-dir, so
# the only thing that differs run-to-run is whatever generate_parameter_sweep.py
# varied. Reusable for any sweep (power, absorptivity, ...), not power-specific.

import argparse
import glob
import os
import subprocess
import sys
import time

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--decks-dir', required=True, help='folder of *.k decks from generate_parameter_sweep.py')
parser.add_argument('--pattern', default='*.k', help='glob pattern within --decks-dir (default *.k = all)')
parser.add_argument('--input-data-dir', required=True)
parser.add_argument('--toolpath', default='toolpath.crs')
parser.add_argument('--stop-fraction', type=float, required=True)
parser.add_argument('--n-frames', type=int, required=True)
parser.add_argument('--out-dir', required=True, help='base output dir; each deck gets its own results_<stem> subfolder')
parser.add_argument('--device', type=int, default=0)
args = parser.parse_args()
os.makedirs(args.out_dir, exist_ok=True)

decks = sorted(glob.glob(os.path.join(args.decks_dir, args.pattern)))
if not decks:
    raise SystemExit('no decks found matching {}/{}'.format(args.decks_dir, args.pattern))
print('found {} decks to run, identical stop_fraction={} n_frames={}'.format(
    len(decks), args.stop_fraction, args.n_frames))

results = []
for k_path in decks:
    stem = os.path.splitext(os.path.basename(k_path))[0]
    out_dir = os.path.join(args.out_dir, 'results_{}'.format(stem))
    if os.path.exists(out_dir) and os.listdir(out_dir):
        print('SKIP {} - output already exists at {} (no overwrite)'.format(stem, out_dir))
        results.append((stem, out_dir, 'skipped'))
        continue

    cmd = [sys.executable, 'run_wall.py',
           '--k', k_path,
           '--out', out_dir,
           '--toolpath', args.toolpath,
           '--input-data-dir', args.input_data_dir,
           '--stop-fraction', str(args.stop_fraction),
           '--n-frames', str(args.n_frames),
           '--device', str(args.device)]
    print('\n=== running {} ==='.format(stem))
    print(' '.join(cmd))
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    dt = time.time() - t0
    if proc.returncode != 0:
        print(proc.stdout[-3000:])
        print(proc.stderr[-3000:])
        raise SystemExit('run failed for {} (exit {})'.format(stem, proc.returncode))
    n_written = len([l for l in proc.stdout.splitlines() if 'frames written' in l])
    print('done in {:.1f}s'.format(dt))
    results.append((stem, out_dir, 'ok ({:.1f}s)'.format(dt)))

print('\n=== sweep run summary ===')
for stem, out_dir, status in results:
    print('  {:45s} {:10s} {}'.format(stem, status, out_dir))
