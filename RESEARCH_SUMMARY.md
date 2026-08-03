# Toolpath-Aware Composition Grading for Multi-Material DED: Research Summary

Status: core research arc complete (as of commit `3880bad` + documentation
fixes on branch `toolpath-experiments`). This document is the single
entry point for the whole body of work — read this before any individual
report file under `examples/`.

## 1. The problem

Directed Energy Deposition (DED) can deposit spatially-varying material
composition (functionally graded materials), but the natural way to
specify that grading — as a function of global spatial position (e.g.
composition varies linearly along the X axis) — ignores how the part is
actually built. A real DED process deposits material by tracing a toolpath
layer by layer, track by track; a composition scheme defined purely in
space has no relationship to that sequence. Two open questions followed:

1. Can composition be defined in a way that respects the actual deposition
   sequence (toolpath arc length) instead of raw spatial position, in a
   way that works on arbitrary, real part geometries and toolpaths — not
   just an idealized test case?
2. If so, does the *choice* of composition-mapping scheme (toolpath-aware
   vs. naive spatial) have real thermal consequences during the build, and
   if it does, what manufacturing parameter actually controls the size of
   that effect?

## 2. The framework developed

A three-stage, composable architecture (`examples/wall/composition_lib.py`):

```
s = coordinate_function(centroid, mode, ...)      # global_x/y/z OR toolpath_arc_length
composition = composition_function(s, mode, ...)  # linear/sinusoidal/step/sigmoid/constant
```

- **`toolpath_arc_length` mode**: exact nearest-*segment* projection (not
  nearest-waypoint — waypoint spacing is ~18x element size on real
  datasets, so nearest-waypoint would produce a visibly stair-stepped
  field). Clamps per segment, keeps the global minimum across the whole
  path; chunked (default 2000 centroids/batch) to bound memory on large
  meshes without approximation.
- Every centroid/composition value is computed once at preprocessing time
  and written directly into the `.k` deck (`*ELEMENT_COMPOSITION`) — the
  solver itself (`update_cp_cond`) is composition-agnostic and was never
  modified; all new logic lives in preprocessing.
- Unified tooling replacing per-material duplicated scripts:
  `prepare_composition.py` (general preprocessor, 5 modes: pure/constant
  ×2 materials + graded), `run_wall.py` (single parameterized simulation
  runner, replacing 6 hand-copied `main_wall_<material>.py` scripts).
- Auto-detection of which material ID is substrate vs. build region by
  birth time (never trusts `*PART` labels, which were found to be
  backwards from physical reality in more than one donor dataset).

## 3. Validation

**Exact, floating-point-level.** Forcing composition to a constant [1,0]
or [0,1] must reproduce the legacy pure-material solver path exactly.
Measured max abs. difference: 2.27e-13 K (Inconel endpoint), 4.55e-13 K
(Titanium endpoint) against a 1e-3 K tolerance — genuine floating-point
noise, not approximate agreement.

**Generalized across three geometrically and process-distinct real donor
datasets** (not synthetic test cases): part002 (hexagonal ring, spiral
toolpath), part005 (block+boss, curved toolpath), part004 (notched block,
dense raster toolpath). On every dataset:
- Composition field is bounded, NaN/Inf-free, and structurally distinct
  from the naive global-X baseline (correlation ≈ 0, e.g. −0.001 to −0.007
  across datasets).
- A toolpath-resolution artifact (nearest-segment "seam" ambiguity when
  spatially-adjacent elements are temporally/arc-length distant) was
  found, characterized quantitatively (not assumed), and shown to be a
  real geometric consequence of toolpath density relative to layer count
  — not a bug — with wildly different prevalence by dataset (part002:
  0.46%, part005: 0.00%, part004: 34.7%, the last one directly explained
  by part004's coarse-per-layer raster).
- In every dataset and at every depth tested, the largest thermal
  differences between composition-mapping schemes track the active melt
  pool, never the seam artifact — confirmed via spatial (not index-based)
  cross-checks, accounting for `domain_mgr`'s birth-time element reordering.

**A significant methodological event**: a real bug was found in
`run_wall.py` (`--toolpath` resolved relative to the working directory,
never joined with `--input-data-dir`) that caused two already-published
results (part004's standalone 15%-depth validation and an entire 10-run
power sweep) to silently use the wrong toolpath — a demo toolpath, not the
real dataset's. This was caught not by a crash but by *noticing an
inconsistent finding* (a follow-on pilot experiment produced results 6×
larger than the flagship sweep at the same nominal conditions) and tracing
it to source. Both results were rerun and corrected; part002 and part005
were checked directly and confirmed unaffected. This is disclosed openly
in the record (`examples/wall/run_wall.py`'s fix commit `26e9706`,
`memory/feedback_verify_dont_assume.md`) rather than quietly fixed —
transparent error correction is treated here as part of the validation
story, not a footnote.

## 4. The scientific question investigated

Given the framework works, which manufacturing parameter actually governs
the *magnitude* of the thermal difference between composition-mapping
schemes? A systematic review preceded any new experiment (see
`memory/project_controlled_parameter_studies.md`):

| Variable | Independently controllable? | Difficulty | Notes |
|---|---|---|---|
| Laser power | Yes | Low | Single field edit, isolation-verified |
| Absorptivity | Yes | Low | Mathematically degenerate with power (`q_in = lp × absorptivity`) — low marginal value |
| Scan speed | Yes | Moderate | Confounds energy density *and* revisit interval simultaneously |
| Hatch spacing | Yes | High | Needs a synthetic toolpath; uniquely isolates track-proximity effects from energy input |
| Scan strategy | Yes | Very high | Multivariable, not a single scalar — too coarse for a first controlled experiment |
| Revisit interval | **No** | — | Emergent property of the above, not a real independent lever |
| Birth order (decoupled from toolpath) | Only at the file-format level | — | Not a real physical manufacturing variable |

**Laser power was chosen** as the highest-value, lowest-risk first
controlled experiment: a real, independently-adjustable manufacturing
parameter, cheap to isolate (single-field deck edits, automatically
verified to change nothing else), and directly motivated by an earlier
cross-dataset observation that datasets with higher power showed larger
composition-mode thermal differences (though that observation was
confounded by geometry and toolpath varying simultaneously — exactly what
a controlled sweep was needed to disentangle).

**Experiment**: part004 (smallest mesh, fastest turnaround, cleanest
toolpath coverage — 100% of elements reached, 0 never-reached), 400W to
1200W in 200W steps, both composition-mapping schemes, 15%-depth,
everything else (geometry, mesh, toolpath, scan speed, composition
profile, materials, boundary conditions) held identical by construction
and automatically isolation-verified (every generated deck diffed
line-by-line against its base; exactly one line differs).

## 5. What we learned

**Laser power has a substantial effect on the absolute magnitude of
composition-induced thermal sensitivity.** Transient max temperature
difference between composition schemes rose from 219K at 400W to 490K at
1000W (peak temperatures 2629K → 4914K over the same range; melting is
present at every power level tested). This reverses an earlier
(bug-affected) result that had suggested only a weak effect.

**Relative sensitivity is comparatively stable.** Normalized by each run's
own peak temperature rise, composition-mode sensitivity holds at roughly
0.08–0.11 across the whole 400–1200W range — the absolute effect appears
to scale roughly in proportion with overall thermal magnitude, rather than
power disproportionately amplifying or damping composition's importance,
over this range.

**Composition-driven differences are always melt-pool-localized**, never
seam-artifact-driven, across every dataset and every power level tested —
the single most consistent finding across the entire project.

**An apparent non-monotonic dip at 1200W (490K → 432K) was investigated
using already-computed data (no new simulation — the solver is
deterministic, so a literal repeat run would add no information) and is
most likely a measurement-window artifact, not a real reversal**: checking
every power level's divergence-vs-time trace showed that 400W, 600W, and
1000W each completed a full rise-and-decline cycle within the fixed
15%-depth window, while 800W and 1200W were *still rising* when the window
closed — meaning their recorded "transient max" is a lower bound, not the
true peak. Under a fixed observation window, two of five sweep points were
truncated mid-event; the "dip" likely reflects that truncation rather than
a genuine physical effect. (Full detail: `power_sweep_report.txt`'s
addendum.)

## 6. Where future work begins

The hatch-spacing study — infrastructure built and pilot-validated
(`h=1.12mm`, all 7 toolpath/birth/mesh checks pass, smoke test confirms
correct track-by-track thermal behavior) but **not run to completion,
deliberately**. It would answer a genuinely different question (does
track-to-track thermal accumulation, independent of energy input, drive
composition sensitivity?) but is not required to complete the current
research arc, which already has a clear problem statement, a validated
general-purpose technical contribution, and one clean controlled finding.
Extending to hatch spacing now would open a second study rather than
closing the first — explicitly deferred per project-scope discussion
(`memory/project_controlled_parameter_studies.md`).

Other natural extensions, in rough priority order: (1) resolve the
1200W/800W truncation by extending the sweep to a longer stop-fraction
(not a repeat run) if the exact shape of the power-sensitivity curve
becomes important later; (2) execute the hatch-spacing sweep
(0.56/1.12/1.68/2.24/3.36mm, infrastructure ready); (3) scan-strategy
comparison; (4) additional alloy pairs beyond IN718/TI64; (5) extend from
thermal-only to mechanical (residual stress) consequences of
composition-mapping choice.

## Reusable infrastructure delivered

All in `examples/wall/`, all independently tested:
`composition_lib.py` (coordinate/composition engine), `prepare_composition.py`,
`run_wall.py`, `diagnose_composition_mapping.py`, `compare_composition_modes_*.py`,
`generate_parameter_sweep.py` / `run_parameter_sweep.py` / `analyze_parameter_sweep.py`
(general single-field sweep infrastructure — reusable for absorptivity or
any future `*GAUSS_LASER`-field study with no changes), `generate_raster_toolpath.py`
/ `build_synthetic_deck.py` / `validate_synthetic_toolpath.py` (synthetic
toolpath + birth-time generation for hatch-spacing/scan-strategy studies),
plus regression tests (`test_composition_regression.py`,
`test_toolpath_arc_length.py`, 21 hand-verified cases).

## Where the detailed evidence lives

- Framework + endpoint validation: `examples/wall/composition_lib.py`,
  `endpoint_validation/` under part002
- Cross-dataset generalization: `part004_generalization_summary.txt` (see
  superseded-notice header), the equivalent part005 report, and the
  original part002 diagnostics
- The critical bug and its fix: `examples/wall/run_wall.py` commit
  `26e9706`, `memory/feedback_verify_dont_assume.md`
- Power sweep, corrected: `power_sweep/power_sweep_report.txt`
  (commit `3880bad`), raw data in `power_sweep/analysis/`
- Hatch-spacing pilot: `hatch_spacing_study/pilot_h1.12/pilot_report.txt`
- Full narrative history: `memory/project_external_dataset_import.md`,
  `memory/project_controlled_parameter_studies.md`
