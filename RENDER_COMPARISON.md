# Saved-checkpoint comparison: horse and checkerboard

`render_comparison.py` reads existing checkpoints through their matching saved
`runs/.../config.yaml`. It does not train, construct a coupling, open a training
bank, compute Jacobians, or modify a checkpoint. No extra dependency is needed.

The default now uses the ORIGINAL `eval.py` / `eval_horse.py` image rendering:
same histogram, Gaussian smoothing, viridis, percentile PowerNorm and bilinear
interpolation; horse keeps its black background. Only the arrangement changes:
six generated t=1 panels side-by-side, one method per column, one NFE per row.
No scatter, colorbar, extra metric captions or target column is added by default.
Checkerboard's intermediate t snapshots are omitted: the comparison is at t=1.
Its original normalization rule is applied to that displayed endpoint; horse
retains the original target/generated-pair normalization. This reproduces the
eval rendering style, not necessarily identical pixels from past random draws.

## Run from the project root (Windows CMD)

`comparison_runs.yaml` is already populated with the 12 run folders from the
provided screenshots. All entries use `./runs/.../config.yaml`, relative to this
manifest. Copy it to the lab machine's project root beside `runs/` and
`render_comparison.py`; no username, drive letter or absolute project path is needed.
Do NOT copy `comparison_runs.example.yaml` over this populated file.

Only for a DIFFERENT set of runs, copy the example to a new manifest name and
replace its `REPLACE_*` paths. Keep each `.pt` file beside
its saved `config.yaml` when moving runs between machines. Do not use the original
experiment template YAML if training changed its seed/settings. Paths in the
manifest are relative to the manifest's own directory.
Use forward slashes or single quotes for Windows YAML paths.

First validate every selected file, method, dataset, architecture and checkpoint
training signature on CPU, without running inference or creating outputs:

```bat
python render_comparison.py comparison_runs.yaml --check-only
```

Then render both datasets with the requested NFEs:

```bat
python render_comparison.py comparison_runs.yaml --nfes 1 2 4 8 16 32 128 --clouds 64 --batch-size 8 --device cuda
```

All those NFE/cloud/batch settings are defaults. `--device auto` (the default)
uses CUDA if available, otherwise CPU. If GPU memory is limited, lower only
`--batch-size` to 1 or 2; this never changes the number of points in a cloud.
Only one checkpoint is held on the GPU at a time. Computation includes
191 Euler steps per source cloud and model across the default seven NFEs.

For a quick one-dataset smoke run before the full comparison:

```bat
python render_comparison.py comparison_runs.yaml --datasets horse --nfes 1 8 --clouds 4 --batch-size 2 --device cuda
```

No `--view` option is needed. `--view eval` explicitly selects the default.
The previous diagnostic views remain opt-in only (`--view density/scatter/both`).
`--bins`, `--sigma`, `--gamma` and `--cloud-index` control those legacy diagnostic
views, not the original evaluator style, whose settings are reused unchanged.

## Select the correct method explicitly

The manifest's list order is the figure's column order. The example contains:

| Column | Required saved `coupling` |
| --- | --- |
| Independent | `independent` |
| EFM (permutation) | `equivariant_ot_permutation` |
| Minibatch OT | `minibatch_ot` |
| NSOT | `nsot` |
| TG (cached) | `target_guided_cached` |
| MS-W2 | `multiscale_wasserstein` |

If your TG run is streaming `target_guided`, change BOTH the manifest's `coupling`
and its label to match. Other TG variants must likewise be named explicitly.
The renderer fails if a declared method disagrees with its checkpoint/config.
In particular, a cross-leaf `multiscale_refined` run cannot silently occupy the
baseline MS-W2 slot. NSOT beta/bank settings remain in the saved comparison JSON.
EFM here means this repository's permutation-only adapter, not a full external
EFM implementation with rigid rotations. NSOT is this repository's 2D variant.

One dataset or a smaller subset of methods is allowed. Remove unwanted entries
from a copied manifest, or use `--datasets horse` / `--datasets checkerboard`.
Nothing auto-discovers a newest/best checkpoint or skips a missing model.
An explicitly supplied relative wildcard path is supported ONLY if exactly one
config file matches. Zero matches fail; multiple matches list candidates and fail.
The supplied horse EFM entry uses `20261006T*_c4770473` to retain the exact run ID
without manually transcribing its timestamp. Update BOTH `render_comparison.py`
and `comparison_runs.yaml` on the lab machine to use this feature. The concrete
resolved path and checkpoint hash are still saved in the comparison report.

Legacy weights-only checkpoints are rejected by default. If you genuinely need
them, `--allow-legacy` labels their plots `UNVERIFIED` and retains that flag in
JSON/CSV. The supplied config cannot be authenticated against legacy weights.
Never treat a legacy comparison as verified training provenance.

## Output

Each invocation creates a unique directory; no previous result is overwritten:

```text
eval_results/model_comparison_<timestamp>_<id>/
  request.json                 # resolved models; present even if a later step fails
  comparison.json              # status=complete only after all datasets finish
  checkerboard/
    comparison_grid.png        # rows=NFE, columns=6 generated models
    comparison_nfe_001.png     # original eval style, one strip per NFE
    ... nfe_002,004,008,016,032,128 ...
    comparison.json            # hashes, configs, metrics, rendering settings
    cd_table.png               # NFE rows x model columns, raw cloud-mean CD
    metrics.csv                # pooled metrics, one row per model/NFE
    clouds.npz                 # unchanged sources, targets, all generated endpoints
  horse/
    ... same files ...
```

Images show just the model labels and NFEs, without diagnostic overlays.
CD/leakage and other metrics remain available in JSON/CSV. Model inference never
receives target points or target patches.

### CD tables without repeating inference

New comparisons automatically save `horse/cd_table.png` and
`checkerboard/cd_table.png`, and print their values in the terminal.
Each table uses NFE rows and model columns, with raw CD values (no x1000 scaling).
CD is the sum of directional mean squared distances, averaged across clouds;
lower is better. Missing results appear as N/A, never zero. The JSON retains
full-precision table values; the image displays six significant digits.

For an already completed comparison, replace `YOUR_RUN` with its actual folder:

```bat
python render_comparison.py eval_results/YOUR_RUN --cd-table-only
```

This reads only the saved `comparison.json` files. No model, training bank, GPU
or repeated inference is needed. It creates a new `cd_tables_<timestamp>_<id>`
folder inside the selected run containing `horse_cd_table.png`,
`checkerboard_cd_table.png` and `cd_tables.json` with values/source hashes.
A single dataset's `comparison.json` or directory also works. NFEs come from
the stored results, not the CLI's inference defaults. Existing files are untouched.

## Fairness and interpretation

- Fresh CPU-generated standard-Gaussian arrays and independent target reference
  arrays are reused **byte-for-byte across all models and NFEs** within a dataset.
  Isolated indexed sampling is unaffected by model initialization or loop order.
  Matching N/dtype also gives the same source draw across the two datasets.
- Every NFE is a separate Euler solve from `t=0` to `t=1`, restarting from the
  original source. These are NOT intermediate snapshots of a 128-step solve.
- The renderer enforces equal point counts, dtype and target grid within a
  dataset. It uses the checkpoint's original N, never resamples a trained model
  at a different N. Architecture, training-budget, training-batch-size and seed
  differences produce warnings; all original configs are retained for auditing.
- Default eval-style plots pool 64 **separate N-point clouds**, not one model input
  with `64*N` points. Horse uses extent 1.1, 256 bins, sigma 1.0; checkerboard
  uses extent 1.2, 192 bins, sigma 1.2. Norms follow the original evaluators,
  not a new global cross-model maximum. Color brightness alone is not a metric.
- The optional LEGACY diagnostic density view fixes axes to `[-1.2,1.2]^2`.
  Bin count, smoothing and one common color norm
  are shared across **all methods and NFEs**, including the per-NFE strips, for
  each dataset. Probability density divides by ALL points, including those
  outside the view. Outside-view percentages are shown, not silently normalized
  away. The shared maximum avoids upper clipping; gamma is 0.5 by default.
- Leakage/CD/ROI metrics use the original, unsmoothed points, including outliers.
  Metric histogram bins (default 64) are separate from display bins (default 192).
  CD is the mean across separate clouds; cell/ROI mass metrics are pooled.
- One checkpoint per method is a visual pilot, not multi-seed statistical evidence.
  Different devices/kernels or microbatch sizes may have small numerical effects.
  Inference time includes transfers and no warmup, so it is NOT a speed benchmark.

All NPZ keys are numeric arrays, usable with `allow_pickle=False`: `source`,
`target`, and `prediction_<method_id>_nfe_<nfe>`, each `[clouds,N,2]`.
