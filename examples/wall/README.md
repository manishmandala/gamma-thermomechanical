# examples/wall — quickstart

This folder has a lot of files in it. You need **two** of them to run a
simulation. Everything else is either historical, or built for one
specific past experiment. This page tells you which is which.

## Run your first simulation

From inside this directory:

```
python3 run_wall.py --k thinwall.k --out results_demo --stop-fraction 0.1
```

That's a real thermal simulation — the plain, ungraded demo wall, run to
10% of its build. Frames land in `results_demo/` as `.vtu` files you can
open in ParaView. Increase `--stop-fraction` toward `1.0` for a complete
build (full builds take much longer — see the runtime note below).

Run `python3 run_wall.py --help` to see every option. The two you'll use
most: `--k` (which `.k` deck to simulate) and `--out` (where to write
results). Everything else has a sensible default.

## The two tools you actually need

- **`run_wall.py`** — runs any `.k` deck through the solver. This is the
  only script that starts a simulation. Point it at a deck, get results.
- **`prepare_composition.py`** — if you want a *graded* (multi-material)
  version of a dataset instead of the plain mesh, run this first to
  generate the graded `.k` deck, then feed that deck to `run_wall.py`.
  Example:

  ```
  python3 prepare_composition.py --dataset ../incoming_dataset/part004_LP400_SSp8_H1.12_SSt1_LH1.0 \
    --kfile 4.k --mode graded --coordinate-mode toolpath_arc_length \
    --composition-mode sinusoidal --out my_graded_deck.k

  python3 run_wall.py --k my_graded_deck.k --out results_graded \
    --input-data-dir ../incoming_dataset/part004_LP400_SSp8_H1.12_SSt1_LH1.0 \
    --stop-fraction 0.15
  ```

That's the entire workflow. Everything below is context you can ignore
until you actually need it.

## Everything else in this folder, briefly

- **`legacy_main_wall_*.py`** — the *old* way of running a simulation,
  from before `run_wall.py` existed: one hand-copied Python file per
  material/case, with the deck filename hardcoded inside the source
  instead of passed as an argument. Kept for historical reference only.
  **Don't use these for new work** — `run_wall.py` does everything they
  did, with no code editing required.
- **`composition_lib.py`** — the engine behind graded compositions
  (`compute_centroid` → `coordinate_function` → `composition_function`).
  You don't need to read this to *use* the tools above; it matters if
  you're extending composition logic itself (e.g. adding a new profile
  shape).
- **`diagnose_composition_mapping.py`**, **`compare_endpoint_validation.py`**
  — validation/diagnostic tools, not part of the basic run workflow. Use
  these if you're checking a *new* dataset's composition field before
  trusting it, not for routine runs.
- **`generate_parameter_sweep.py` / `run_parameter_sweep.py` /
  `analyze_parameter_sweep.py`** — infrastructure for running a controlled
  sweep of a single `*GAUSS_LASER` field (power or absorptivity) across
  many values at once, with automatic isolation checking. Only relevant if
  you're setting up a new parameter study.
- **`generate_raster_toolpath.py` / `build_synthetic_deck.py` /
  `validate_synthetic_toolpath.py`** — infrastructure for generating a
  synthetic toolpath (e.g. for a hatch-spacing study) instead of using a
  real dataset's own toolpath. Only relevant if you're building a new
  synthetic experiment.
- **`compare_composition_modes_part00N_*pct.py`,
  `make_gradient_comparison_figure_part00N.py`, `render_full_build.py`,
  `make_presentation_figures.py`** — one-off analysis/figure scripts
  written for specific past experiments on specific datasets, kept for
  reproducibility of already-published results. Not templates you need to
  understand to run something new — copy the *pattern* if useful, but
  don't feel obligated to read them.
- **`test_composition_regression.py`, `test_toolpath_arc_length.py`** —
  automated tests. Run these if you've changed anything in
  `composition_lib.py` or `prepare_composition.py`, to check you haven't
  broken existing behavior: `python3 test_composition_regression.py`.

## A note on runtime

Simulation cost scales with mesh size and how much of the build you
simulate (`--stop-fraction`). As a rough guide from this project's own
datasets: a ~14,000-element mesh takes a few minutes at 15% depth and
under an hour at full (100%) depth; a ~40,000-element mesh can take
several hours at full depth. Start with a small `--stop-fraction`
(0.05-0.15) to sanity-check a new deck before committing to a full run.

## Want the bigger picture?

- `../../RESEARCH_SUMMARY.md` — short summary of the toolpath-arc-length
  composition research project this codebase supports.
- `../../TECHNICAL_REVIEW.md` — full technical write-up: the problem, the
  algorithm, how it was validated, and what was learned.

Neither is required reading to run a simulation — they're there if you
want to understand *why* the tools work the way they do.
