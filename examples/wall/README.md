# examples/wall

A thin-wall DED (directed energy deposition) thermal simulation demo, plus
several related sub-projects for studying multi-material composition
grading. This README documents every script and data file in this folder
so anyone new to the codebase can find the right tool without reading
source first.

## Quickstart

From inside this directory (`examples/wall/`):

```
python3 run_wall.py --k thinwall_clean.k --out results/demo --stop-fraction 0.1
```

That's a real thermal simulation - the plain, ungraded, 2-material (base
plate + build wall, both TI64) demo, run to 10% of its build. Frames land
in `results/demo/` as `.vtu` files, viewable in ParaView. Increase
`--stop-fraction` toward `1.0` for a complete build (see **Runtime** below
- full builds take much longer). All simulation output lives under
`results/` (gitignored), one subfolder per run, named for what the run is.

Run `python3 run_wall.py --help` for every option. The two you need most:
`--k` (which `.k` deck to simulate) and `--out` (where to write results).

### Graded (multi-material) runs

To grade the wall's material composition instead of using a plain 1- or
2-material deck, generate a graded `.k` file first, then feed it to
`run_wall.py`:

**2 materials** (`prepare_composition.py` - also does pure/constant-composition
endpoint validation, and works on any dataset under `../incoming_dataset/`,
not just this folder's own decks):

```
python3 prepare_composition.py --dataset . --kfile thinwall_clean.k \
  --mode graded --coordinate-mode global_x --composition-mode sinusoidal \
  --out my_graded_deck.k

python3 run_wall.py --k my_graded_deck.k --out results/my_graded_run --stop-fraction 0.15
```

**N materials** (`prepare_composition_nmat.py` - any material count, any
blend shape, from `thinwall_clean.k` only):

```
python3 prepare_composition_nmat.py --materials TI64,IN718,1018,CPTi \
  --shape parabola --coordinate-mode global_x --out my_4mat_deck.k

python3 run_wall.py --k my_4mat_deck.k --out results/my_4mat_run --stop-fraction 0.15
```

Before committing to a long run with an unfamiliar material combination,
sanity-check the stable timestep first (see **Picking materials without
blowing up the timestep** below) - some materials (Cu, Al) force a much
smaller `dt` than others and can make a run take hours longer than expected.

## Core tools (top level)

| File | What it does |
|---|---|
| `run_wall.py` | **The only simulation runner.** Takes any `.k` deck + output folder via CLI args and runs it through the GPU thermal(-mechanical) solver, writing `.vtu`/`.vtk` frames. |
| `composition_lib.py` | The engine behind graded compositions: `compute_centroid` -> `coordinate_function` (position -> normalized `s` in [0,1], modes `global_x`/`global_y`/`global_z`/`toolpath_arc_length`) -> `composition_function` (single-scalar 2-material presets: `linear`, `sinusoidal`, `step`, `sigmoid`, `constant`, `quadratic`). Used by `prepare_composition.py` and the `toolpath_arc_length/gradient_material_continuous_*.py` scripts. You only need to read this if you're adding a new 2-material composition shape. |
| `prepare_composition.py` | Generates a 2-material graded (or pure/constant, for endpoint validation) `.k` deck from any clean 2-material source deck, via `composition_lib.py`'s presets. Regression-tested against committed output (`toolpath_arc_length/test_composition_regression.py`) - don't change its output format without re-running that test. |
| `prepare_composition_nmat.py` | General **N-material** version of the above: any material count (from `MATERIAL_LIBRARY`), any blend shape (from `BLEND_SHAPES` - currently `parabola`, `sinusoidal`), any coordinate mode. Each material gets a kernel-shaped zone centered at an evenly-spaced position along the coordinate; per-element fractions are the kernel values normalized to sum to 1. Add a material to `MATERIAL_LIBRARY` or a shape to `BLEND_SHAPES` to extend it. |
| `generate_toolpath.py` | Generates `toolpath.crs` for the thin-wall demo: a back-and-forth raster where every single-direction pass deposits one full layer. Element birth times in the mesh must match this schedule exactly - see next row. |
| `regen_birth.py` | Recomputes element birth (activation) times to match the *current* `toolpath.crs` and rewrites `thinwall_discrete_bands.k`'s `*DEFINE_CURVE` block in place. **Run this after any change to `generate_toolpath.py`'s schedule**, or the laser and the deposited material fall out of sync. |
| `find_base_center_node.py` | One-off utility: finds the node ID at the center of the wall demo's substrate base, for use as a `--log-node` target or manual sanity check. Not part of any pipeline. |

## Input decks (`.k` files, top level)

| File | What it is | Produced results |
|---|---|---|
| `thinwall_clean.k` | The plain, pre-gradient 2-material mesh (substrate + build, both TI64) - the canonical starting point for `prepare_composition.py`/`prepare_composition_nmat.py`. | `results/demo` |
| `thinwall_discrete_bands.k` | The **same mesh**, with its build region split into 70 discrete material bands (one per element column in X) instead of a continuous blend, via `toolpath_arc_length/gradient_material_discrete_bands_TI64_IN718.py`. Named for what it currently is - it used to be called `thinwall.k` before that script rewrote it in place; the git history for `thinwall.k` still applies to this file. | `results/discrete_bands` |
| `thinwall_IN718.k` / `thinwall_Cu.k` | `thinwall_clean.k` with the build region set to a single ordinary material (IN718 / Cu) instead of TI64 - pure-material control runs. | `results/pure_IN718`, `results/pure_Cu` |
| `thinwall_graded.k` | Continuous TI64/IN718 blend (sinusoidal, 2 materials), generated by `toolpath_arc_length/gradient_material_continuous_TI64_IN718.py`. | `results/graded`, `results/demo_cpu` (CPU-vs-GPU reference check) |
| `thinwall_graded_cu.k` | Continuous TI64/Cu blend (sinusoidal) - a higher-thermal-contrast counterpart to the above, generated by `toolpath_arc_length/gradient_material_continuous_TI64_Cu.py`. | `results/graded_cu` |
| `thinwall_graded_linear.k` | Continuous TI64/IN718 blend, `linear` composition mode instead of `sinusoidal`. | `results/graded_linear` |
| `thinwall_graded_4mat_parabola.k` | 4-material blend (TI64/IN718/1018/CPTi) over `global_x`, `parabola` shape - generated by `prepare_composition_nmat.py` (see **Picking materials** below for why these 4). | `results/thinwall_4mat_parabola` |
| `wall_graded_arclength_sinusoidal.k` / `wall_graded_quadratic_stresstest.k` | Composition-graded variants of the wall geometry using the `toolpath_arc_length` coordinate mode instead of `global_x`, sinusoidal and quadratic shapes respectively. | `results/wall_arclength_sinusoidal_20pct`, `results/wall_quadratic_stresstest_20pct` |
| `toolpath.crs` | The demo wall's deposition toolpath (back-and-forth raster), generated by `generate_toolpath.py`. Every deck above uses this. |
| `toolpath_off.crs` | Same geometry, laser held off - used for pure-relaxation runs (`external_mesh/` import checks). |

**Every `results/` subfolder is reproducible**: run the deck in the table
above through `run_wall.py` with a matching `--stop-fraction`/`--n-frames`.
`results/` itself is gitignored - nothing there is checked in.

## Picking materials without blowing up the timestep

`gamma.py`'s stable timestep is bounded by `min_Cp * density / max_Cond`
for whichever material in a blend has the highest thermal diffusivity - one
"fast" material anywhere in an N-material mix drags the *entire*
simulation's timestep down, even if it's a minor component everywhere.
Roughly, from this project's material library (higher = safer, larger dt):

| Material | dt proxy |
|---|---|
| IN718 | 0.120 |
| CPTi | 0.101 |
| 1018 | 0.073 |
| TI64 | 0.069 |
| Al | 0.010 |
| Cu | 0.009 |

Al and Cu are ~7-14x worse than the other four - mixing either into a blend
with them can turn a 30-minute run into a multi-hour one. `thinwall_graded_4mat_parabola.k`
deliberately uses TI64/IN718/1018/CPTi for this reason. If you want a
high-thermal-contrast comparison on purpose (that's what `thinwall_graded_cu.k`
is for), budget accordingly and use a small `--stop-fraction` first.

## Subfolders

- **`comparison_data/`** - analysis figures and the `plot_*.py`/`render_*.py`
  scripts that produced them, comparing pairs/sets of completed `results/`
  runs (pure vs. graded, TI64/IN718 vs. TI64/Cu, external-mesh 2-mat vs.
  4-mat). Run these from inside `comparison_data/` itself - they use `../`
  paths back to `results/`. See the per-file table below.
- **`external_mesh/`** - imports an externally-generated mesh (`sol_100.vtu`,
  via `*EXTERNAL_MESH`) with its own per-element composition field, instead
  of a composition computed by this codebase.
- **`parameter_sweep/`** - infrastructure for sweeping a single
  `*GAUSS_LASER` field (power or absorptivity) across many values with
  automatic isolation checking. Only relevant if you're setting up a new
  parameter study.
- **`synthetic_toolpath/`** - infrastructure for generating a synthetic
  toolpath (e.g. for a hatch-spacing study) instead of using a real
  dataset's own toolpath.
- **`toolpath_arc_length/`** - the toolpath-arc-length composition research
  project: the `toolpath_arc_length` vs `global_x` coordinate-mode
  comparison, diagnostics, one-off figure/analysis scripts for specific
  past experiments (kept for reproducibility of already-published results,
  not templates), and the two test suites. Run these from inside
  `examples/wall` (not from inside the subfolder), since they import
  `composition_lib.py` from the top level.

### `comparison_data/`

| File | What it does |
|---|---|
| `plot_peak_temperature.py` | Peak temperature at the wall's central node: pure-TI64 vs pure-IN718 vs graded. |
| `plot_peak_temperature_cu.py` | Same comparison, TI64 vs Cu vs graded - the high-thermal-contrast counterpart. |
| `plot_transient_vs_final.py` | Each run's true peak temperature vs. its final-frame temperature, showing how much of the TI64/Cu/graded gap washes out by the end of the run. |
| `render_early_frame_comparison.py` | Side-by-side image of the pure-Cu and graded-Cu runs at their frame of largest thermal contrast. |
| `plot_external_mesh_comparison.py` | Probe-node + domain-wide relaxation curves proving the external mesh's own composition field drives real per-location thermal behavior. |
| `plot_external_mesh_comparison_4mat.py` | Same, extended to the true 4-material external-mesh run vs. the original 2-material collapsed version. |

### `external_mesh/`

| File | What it does |
|---|---|
| `external_mesh_import_check.py` | Original mesh-import sanity check: geometry only, no toolpath, seeded initial temperature, laser off. Kept as a reference example. |
| `run_external_mesh.py` | General runner for the `sol_100_control_*.k` decks in this folder - parameterized version of the import check. |
| `external_mesh_composition.py` / `external_mesh_composition_4mat.py` | Regenerate the graded control decks (`sol_100_control_graded.k` / `_graded_4mat.k`) from the source mesh's own composition data (2 collapsed columns, or all 4 real columns). |

### `parameter_sweep/`

| File | What it does |
|---|---|
| `generate_parameter_sweep.py` | Generic single-`*GAUSS_LASER`-field sweep-deck generator (power or absorptivity), reusable across studies. |
| `run_parameter_sweep.py` | Runs every deck in a sweep folder through `run_wall.py` with identical settings, so only the swept field differs. |
| `analyze_parameter_sweep.py` | Aggregates a completed sweep into a scaling-behavior report + plot. |

### `synthetic_toolpath/`

| File | What it does |
|---|---|
| `generate_raster_toolpath.py` | Reusable synthetic back-and-forth raster-toolpath generator, for controlled hatch-spacing studies. |
| `build_synthetic_deck.py` | Assembles a new deck around a synthetic toolpath while keeping a real dataset's mesh/materials. |
| `validate_synthetic_toolpath.py` | Toolpath/birth-consistency validation for a synthetic-toolpath dataset - run before any thermal simulation on one. |

### `toolpath_arc_length/`

| File | What it does |
|---|---|
| `gradient_material_continuous_TI64_IN718.py` / `_TI64_Cu.py` | Generate `thinwall_graded.k` / `thinwall_graded_cu.k` (continuous 2-material blend). |
| `gradient_material_discrete_bands_TI64_IN718.py` | Generates `thinwall_discrete_bands.k` (70 discrete material bands instead of a continuous blend). |
| `diagnose_composition_mapping.py` | General diagnostic/validation report for the `toolpath_arc_length` composition mapping, for any dataset. |
| `diagnose_part002_mapping.py` | Same, frozen as it was when part002's results were first reported - kept byte-for-byte for reproducibility, not updated for new datasets. |
| `compare_endpoint_validation.py` | Confirms the graded-solver path collapses exactly to the plain-material path at composition=[1,0]/[0,1]. |
| `compare_composition_modes_part002_1pct.py`, `_part002_10pct.py`, `_part003_10pct.py`, `_part004_15pct.py`, `_part005_15pct.py` | One-off matched-run comparisons (`toolpath_arc_length` vs `global_x` coordinate mode) for specific past dataset/depth combinations - kept for reproducibility of published results, not templates. |
| `make_gradient_comparison_figure_part001.py` ... `_part005.py`, `_hatch_pilot_h1.12.py` | One-off presentation figures, one per dataset geometry. |
| `make_presentation_figures.py` | Three presentation figures built from the completed part002 results. |
| `render_full_build.py` | Renders final-frame temperature + composition fields from a completed 100%-depth build. |
| `test_composition_regression.py` | Regression test: proves `composition_lib.py`'s refactored pipeline produces byte-identical output to what shipped in the committed `.k` decks. Run after any change to `composition_lib.py` or `prepare_composition.py`. |
| `test_toolpath_arc_length.py` | Tests for the `toolpath_arc_length` coordinate mode specifically (arc-length table, nearest-segment projection). |

## A note on runtime

Simulation cost scales with mesh size, material diffusivity (see above),
and how much of the build you simulate (`--stop-fraction`). As a rough
guide: a ~14,000-element mesh with well-matched materials takes a few
minutes at 15% depth and under an hour at full depth; the same mesh with
a 70-material discrete-band deck (`thinwall_discrete_bands.k`) is
noticeably slower per step (~1.6x) due to the per-material Python loop in
the solver's property-update path. Start with a small `--stop-fraction`
(0.05-0.15) to sanity-check a new deck before committing to a full run.

## Cleanup history

The old `legacy/` folder (7 hand-copied per-material/per-mode runner
scripts, one per material/mode combination) and `scratch/` (a throwaway
mesh-inspection snippet and a script duplicating `run_wall.py`'s own
pattern) were removed - everything they did is covered by `run_wall.py`.
`scratch/thinwall_graded_linear.k` was kept (it's the real input behind
`results/graded_linear`) and promoted to the top level alongside the other
`thinwall_graded_*.k` variants.
