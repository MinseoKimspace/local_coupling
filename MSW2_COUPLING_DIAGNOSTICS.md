# MS-W2 coupling diagnostics

Run all four new diagnostics together using an existing MS-W2 bank and verified
checkpoint. The current baseline remains `[4,20,100]` with weights `[1,1,1]`.
This command only reads the existing bank/config/checkpoint and writes diagnostic
reports. It does not prepare banks, train models, alter a coupling, or change the
model, loss or Gaussian prior.

From the project directory on the experiment computer, pass the **saved run**
`config.yaml`, not `multiscale_experiments/horse.yaml` or a training template.
Replace the example run-directory names below with the actual run directories.
All example paths are project-relative.

```powershell
python audit_ms_w2_coupling.py "runs/horse/ACTUAL_MSW2_RUN/config.yaml" --check-only
python audit_ms_w2_coupling.py "runs/horse/ACTUAL_MSW2_RUN/config.yaml"
```

Horse and checkerboard can run in one invocation. The dataset is inferred from
the config or its standard `runs/<dataset>/<run>/config.yaml` location; verified
checkpoint metadata must agree.

```powershell
python audit_ms_w2_coupling.py "runs/horse/ACTUAL_MSW2_RUN/config.yaml" "runs/checkerboard/ACTUAL_MSW2_RUN/config.yaml"
```

The default uses the checkpoint's configured device. `--device cuda` or
`--device cpu` changes only the diagnostic inference device. If a bank was moved,
pass its current path with **one** run config:

```powershell
python audit_ms_w2_coupling.py "runs/horse/ACTUAL_MSW2_RUN/config.yaml" --bank "coupling_cache/multiscale_wasserstein/ORIGINAL_BANK.npz"
```

The bank must match the SHA256 recorded in the checkpoint, even if the YAML
omits that hash. A different bank is rejected. Missing banks are never rebuilt.

## Four checks

| Check | Measurements | Main files |
| --- | --- | --- |
| 1. Coupling geometry | Same/cross source-patch velocity jumps in shared distance strata; same-time interpolant neighbors | `geometry_distance_strata.csv`, `geometry_time_conflicts.csv`, `1_coupling_geometry.png` |
| 2. Gaussian routing proxy | Gaussian nearest-bank probability weights, fixed-region masses and query Monte Carlo SE | `routing_regions.csv`, `2_gaussian_routing.png` |
| 3. Fitting and cloud context | Time/region/boundary/roughness FM residuals; fixed-anchor context bias and variance | `fitting_fitting_groups.csv`, `fitting_context_anchors.csv`, `fitting_context_groups.csv`, `3_fitting_context.png` |
| 4. Bank/fresh generation | Endpoint CD/JS/leakage/ROI mass; bank-teacher/rollout distribution gaps | `generation_rollout_quality.csv`, `generation_rollout_paths.csv`, `generation_rollout_region_mass.csv`, `4_bank_fresh_generation.png` |

Each run writes a separate timestamped directory:

```text
analysis_results/<dataset>/<checkpoint>_coupling_diagnostic_TIMESTAMP_ID/
    summary.json
    diagnostics.npz
    bank_points.csv
    ... CSV files and five PNG figures ...
    region_mass_maps.png
```

`summary.json` contains all four summaries, exact input hashes, settings, metric
definitions, environment and original bank provenance. `bank_points.csv` joins
source/target coordinates, source leaf, boundary proxy, velocity roughness,
Gaussian routing weights, overlapping diagnostic groups and observed fitting
errors for the same bank index. Unsampled pair fitting values are blank, not
zero. `diagnostics.npz` contains numeric bank, edge, sample-index, context and
trajectory arrays; load with `allow_pickle=False`.

## Defaults and computation

- Source kNN: 8 neighbors, 8 common distance strata; all 10,000 bank points.
- Gaussian routing: 100,000 fresh standard Gaussian queries, k=1 and k=8.
- Fitting and generation: 32 clouds, the original number of points per cloud;
  inference microbatch 8.
- Context: up to 64 stratified bank anchors, 8 contexts per anchor.
- Times: `0 .25 .5 .75 .9 .95 .99 1`; endpoints are included automatically.
- Endpoint Euler NFEs: 1 and 128; full path uses 128 steps.
- Endpoint/proxy maps: 16x16 on `[-1,1]^2`, plus a square-exterior bin.
- Path TV: a common 32x32 grid on `[-4,4]^2`, plus an exterior bin, to retain
  Gaussian structure outside the final target square.

The expensive part is saved-model inference, especially context tests on the
horse model. kNN trees avoid dense 10,000-by-10,000 cost matrices. No new OT solve
is needed. Progress is printed by diagnostic, fitting time and rollout stage.
For a smaller first run:

```powershell
python audit_ms_w2_coupling.py "runs/horse/ACTUAL_MSW2_RUN/config.yaml" --clouds 8 --anchors 16 --contexts 4 --routing-draws 20000
```

Other useful options: `--nfes 1 8 128`, `--batch-size 4` for lower GPU memory,
`--skip-render` for numeric outputs only, and `--output PATH` for another result
root. `--check-only` verifies input files and hashes without diagnostic output.

## Interpretation

Distance strata adjust distances approximately. Inspect same/cross counts and
unmatched strata; these dependent kNN edges are not independent experimental
replicates. Coincident interpolant edges have separate velocity-jump statistics;
ratio summaries use positive-distance edges. The source boundary proxy means a
point has a neighbor in another source leaf, not an exact kd-boundary distance.

Gaussian k=1 estimates nearest-bank Voronoi probability masses. k=8 allocates a
query equally among its nearest eight bank sources as a sensitivity probe.
Neither is the learned Transformer's routing rule. Region SE uses whole-query
contributions, not eight independent neighbors. Horse ROIs can overlap; their
masses do not form a global probability partition. Every regional mass uses all
points as denominator, including leakage.

Fitting uses fresh cloud compositions from the **seen training pair bank**.
Quantile groups use strict cuts; ties are excluded from extremes. Context anchors
are stratified and descriptive, not population-weighted. Context decomposition
uses the empirical population variance (denominator K), giving
`MSE = squared mean deviation from pair label + context prediction variance`.
This variance is not irreducible uncertainty. No velocity labels are fabricated
for fresh Gaussian inputs.

Bank-source and fresh-source endpoint quality use the same independent fresh
target reference, with separate bank-target and target-only controls. Endpoint
CD/JS/leakage/ROI definitions reuse the project evaluator; arithmetic uses CPU
float64 for both prediction and reference. Fitting/rollout model inference uses
the original model dtype and standard evaluation attention backend.

At intermediate t, compare distributions with the bank teacher interpolant.
An independent same-sized bank-teacher replicate gives a finite-sample reference
for path TV. No final-silhouette leakage is computed at intermediate times.
Off-grid times sample the piecewise-linear Euler polygon at the actual requested
time without changing the uniform solver or adding model evaluations. A paired
coordinate discrepancy is descriptive; FM does not require every sample to
follow its original conditional line. This audit does not repeat the previous
solver-convergence study or establish causal/overfitting claims by itself.
