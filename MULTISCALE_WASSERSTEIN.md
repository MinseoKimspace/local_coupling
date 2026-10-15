# Multiscale Wasserstein matching with a 10,000-point bank

## Question and scope

Does matching **distributions at multiple spatial scales** improve the coupling learned by the existing checkerboard and horse flow-matching models?

This is a new coupling construction, not the removed candidate-bank/selector experiment. No displacement/joint/multiscale score chooses among eight candidates. No FGW structural term, new prior, model conditioning, waypoint, Jacobian penalty, or online perturbation is added to the multiscale method.

Default: one multiscale model per dataset, bank size **M=10,000**, 256 points per training cloud, seed 0, 10,000 optimizer updates, original dataset-specific model/training/evaluation settings. M counts individual bank points, **not** 10,000 clouds. The bank is larger than the removed 2,048-point experiment, while matching the repository's 2D NSOT baseline size. This is not the NSOT paper's main 100K-point experiment.

## Run on the experiment machine

For a controlled **MS-W2 / leaf-only `[0,0,1]` / QAT coupling** comparison on both
2D datasets, see [COUPLING_COMPARISON.md](COUPLING_COMPARISON.md), or run:

```powershell
python run_multiscale.py --coupling-study --stage all --device cuda --seeds 0
```

This is separate from the partition-direction study. The original default
MS-W2 command below is unchanged.

For read-only coupling geometry, Gaussian routing, fitting/context and bank/fresh
generation checks on an existing checkpoint, see
[MSW2_COUPLING_DIAGNOSTICS.md](MSW2_COUPLING_DIAGNOSTICS.md).

From the project directory, prepare, train and evaluate both datasets:

```powershell
python run_multiscale.py --stage all --device cuda
```

This trains **two models** and evaluates each at NFE 1, 2, 4, 8, 16, 32, 64, 128 using the existing evaluator. JSON metrics and rendered PNGs are both saved.

Preparation only (the default; no model training):

```powershell
python run_multiscale.py
```

Keep the exact `experiment_manifest=...` path printed by that command. Resume those prepared jobs:

```powershell
python run_multiscale.py --manifest "experiment_manifests/multiscale_TIMESTAMP_ID/manifest.json" --stage all
```

Train or evaluate separately with the same manifest:

```powershell
python run_multiscale.py --manifest "experiment_manifests/multiscale_TIMESTAMP_ID/manifest.json" --stage train
python run_multiscale.py --manifest "experiment_manifests/multiscale_TIMESTAMP_ID/manifest.json" --stage eval
```

The placeholder above must be replaced by the actual printed path. A manifest references explicit prepared YAMLs, bank hashes, returned training directories, checkpoints and evaluator output files; it does not guess which run is newest. Completed jobs/evaluations are reused after hash verification. Changing a prepared YAML/bank/checkpoint is an error; start a new experiment for changed settings. An interrupted optimizer run has no resumable optimizer checkpoint and is retrained; completed runs are reused.

Optional matched-bank comparisons:

```powershell
python run_multiscale.py --stage all --device cuda --with-nsot
```

This trains **six models**: multiscale, exact superset OT with beta=0, and NSOT with beta=0.2 on each dataset. The two NSOT jobs share one exact-OT cache. The runner explicitly verifies that multiscale and NSOT caches have identical original source and target arrays, M and bank seed. Thus beta=0 isolates matching geometry; beta=0.2 additionally changes online source perturbation. The latter is a method comparison, not a coupling-only ablation.

The controls share endpoint banks and the iid-with-replacement index-sampling law, **not identical step-by-step index draws**. The unchanged NSOT sampler still draws Gaussian noise when beta=0, which advances its random generator differently from the multiscale sampler. Pairwise seed comparisons therefore do not constitute common-random-number online training.

Other examples:

```powershell
python run_multiscale.py --datasets horse --stage all --device cuda
python run_multiscale.py --stage all --device cuda --seeds 0 1 2
python run_multiscale.py --manifest "PATH_TO_MANIFEST" --stage eval --nfe 256
```

The last command adds an NFE to the saved evaluation list. Bank seeds can be varied in separate experiments with `--bank-seed 1`; model/training seeds are controlled independently by `--seeds`.

A tiny pipeline check, not an experiment result:

```powershell
python run_multiscale.py --smoke --stage all
python run_multiscale.py --smoke --with-nsot --stage all
```

Smoke uses CPU, M=64, N=16, partition counts [2,4,8], a tiny model, batch size 2, two updates and NFE 1/2. It does not establish generation quality.

## Diagnose original MS-W2: quality and Jacobian on the same clouds

`audit_ms_w2_clouds.py` is a **read-only, original-MS-W2-only** diagnostic. It accepts an existing
`audit_jacobian.py` JSON with `rollout` or `both` scope, replays those exact Gaussian clouds,
and joins their generation quality with their existing rollout Jacobians. It does not train,
change coupling or load a training cache.

Run from the project directory **on the machine containing the checkpoint**:

```powershell
python audit_ms_w2_clouds.py "PATH_TO_ORIGINAL_MS_W2_HORSE_JACOBIAN.json"
python audit_ms_w2_clouds.py "PATH_TO_ORIGINAL_MS_W2_CHECKERBOARD_JACOBIAN.json"
```

Replace the placeholders with the actual Jacobian JSON paths, not NFE evaluation JSONs.
The dataset, checkpoint config, diagnostic seed, cloud count and time grid are read from
the report. If the saved run moved, pass `--config "runs/.../config.yaml"`; do not pass a
pre-training YAML. The config must match the checkpoint and original audit. Defaults are
NFE **1, 8, 128, 256, 512**, plus the original Jacobian rollout NFE if absent.

The checkpoint SHA256, Gaussian bank SHA256, and **every recorded rollout-state SHA256**
must match. CPU/CUDA or software/backend changes can invalidate bitwise replay. In that
case the program stops rather than joining unrelated measurements. Regenerate the rollout
audit in the current environment (no training cache needed), then use its new JSON:

```powershell
python audit_jacobian.py "SAVED_MS_W2_RUN/config.yaml" --dataset horse --scope rollout --clouds 16 --probes 16 --rollout-steps 256 --seed 2026 --times 0 0.125 0.25 0.5 0.75 0.875 0.9375 0.96875 1
```

For checkerboard use its saved run config and `--dataset checkerboard`. Times must include
0 and 1. The new diagnostic deliberately uses the original single-cloud math-attention
Euler path; it prioritizes reproducibility over batched inference throughput.

### What is compared

- Each generated cloud gets CD against one independent, equal-size target cloud held
  fixed across NFEs; leakage; and checkerboard cell-mass TV or all fixed horse ROI metrics.
- Original rollout Jacobian values are **reused, not recomputed at each NFE**. Quality
  quartiles and correlations use that original rollout's endpoint NFE (256 for the current
  reports), so their generated endpoint is exactly the one measured by the Jacobian audit.
- CD, leakage and front-leg **absolute mass error** (or checker cell TV) each define separate
  low/high-error quartiles. Jacobian values never influence selection. IDs resolve tied
  ranks, boundary ties are flagged, and constant/insufficient metrics produce no groups.
- All-cloud scatter plots and Spearman coefficients relate these quality metrics to late
  Jacobian, endpoint Jacobian and trapezoidal normalized squared-Jacobian energy. No p-values
  or causal claims are made. Late mean is the unweighted mean of recorded times >=0.75.
- Two finite Euler doublings, 128->256->512, check paired endpoint MSE and quality changes
  per cloud. Defaults are absolute tolerances 1e-5 and 1e-3 respectively. Passing means
  empirical stability on these grids, **not** proof of convergence or correct density.
  Leakage changes in units of one point/N can fail the quality tolerance even with tiny MSE.
- 64 independent target-only clouds, each compared against another independent target
  cloud, estimate ordinary finite-sampling variation at the same N. Their 10-90% bands are
  descriptive, not confidence intervals, optimal-CD bounds, or a statistical success test.
  Sparse per-cloud histogram JS is deliberately excluded from rankings; pooled JS is saved.

Outputs in a new `analysis_results/<dataset>/*_same_cloud_*` directory:

| File | Use |
| --- | --- |
| `same_cloud.json` | Full provenance, per-cloud results, finite-refinement checks, groups and correlations |
| `per_cloud.csv` | One cloud/NFE per row; quality, reused original-rollout Jacobian features and stability flags |
| `target_controls.csv` | Target-only sampling baseline |
| `clouds.npz` | All Gaussian sources, fixed references, target controls, generated points and original rollout states |
| `quality_by_cloud.png` | NFE curves per cloud and target-only bands |
| `quality_vs_jacobian.png` | All-cloud quality/Jacobian scatter with cloud IDs |
| `gallery_*.png` | Quality-selected examples at multiple NFEs, target-only examples and each cloud's Jacobian curve |
| `jacobian_groups_*.png` | Raw Jacobian curves and medians of quality-only quartiles |
| `front_legs_zoom.png`, `horse_rois.png` | Horse ROI view; zoom is omitted if the mass-error ranking cannot separate groups |

`--skip-render` retains JSON/CSV/NPZ and terminal summaries. `--control-clouds 128` increases
the target-only baseline, **not** the number of Jacobian clouds. To increase that count,
first generate a larger rollout audit. `--reference-nfe` and `--max-reference-nfe` must
allow at least two doublings; requested NFEs must not exceed the latter.

Interpretation: first check whether high-error examples remain poor after refinement and
whether errors exceed typical target-only sampling variation. Then inspect whether their
Jacobian trajectories differ from low-error examples. A correlation can guide a subsequent
controlled experiment, but cannot by itself identify coupling, approximation, density, or
integration as the cause. One checkpoint and 16 clouds are exploratory; the Monte Carlo
Jacobian estimates also contain probe noise. No result from this diagnostic alone justifies
penalizing Jacobians or changing the coupling.

## Side-by-side comparison with other saved models

Use [RENDER_COMPARISON.md](RENDER_COMPARISON.md) and `render_comparison.py` to compare
Independent, EFM (permutation), Minibatch OT, NSOT, TG and baseline MS-W2 at
NFE 1/2/4/8/16/32/128. It reuses saved checkpoints and common Gaussian inputs,
producing density/scatter grids and per-NFE strips for horse and checkerboard.
No retraining or training-bank access is required.

## Matching construction

### Controlled partition experiment: axis vs PCA

This is a **2D experiment in the current repository**, not the proposed 3D port.
The baseline remains unchanged unless the new options are explicitly selected.
The study varies only how patch memberships are constructed:

| Variant | Source partition | Target partition |
| --- | --- | --- |
| `axis_axis` | Original widest-coordinate-range splits | Original widest-coordinate-range splits |
| `axis_pca` | Original splits | Local PCA projection splits |
| `pca_pca` | Local PCA projection splits | Local PCA projection splits |

Run from the project directory on the lab machine:

```powershell
python run_multiscale.py --partition-study --stage all --device cuda
```

This prepares and trains **six models**: three variants on checkerboard and three
on horse. Defaults remain M=10,000, N=256, levels `[4,20,100]`, relative weights
`[1,1,1]`, seed 0, 10,000 updates and uniform-time unweighted FM. It uses the
existing dataset-specific architecture, optimizer, evaluator and native PNG renderer.
NFE values are 1, 2, 4, 8, 16, 32, 64 and 128. No noise, late-time weighting,
overlap/refinement, coordinate rotation, new loss or Sinkhorn backend is added.

Prepare only, without training:

```powershell
python run_multiscale.py --partition-study
```

The printed `experiment_manifest=...` is under
`eval_results/partition_study/multiscale_TIMESTAMP_ID/manifest.json`.
Resume with that **actual** path; do not repeat `--partition-study` on resume:

```powershell
python run_multiscale.py --manifest "eval_results/partition_study/multiscale_TIMESTAMP_ID/manifest.json" --stage all
```

Other supported runs:

```powershell
python run_multiscale.py --partition-study --datasets horse --stage all --device cuda
python run_multiscale.py --partition-study --seeds 0 1 2 --stage all --device cuda
python run_multiscale.py --partition-study --smoke --stage all
```

Smoke is a tiny CPU two-update integration check, **not performance evidence**.
Three model seeds train 18 models; the three banks per dataset are shared across
model seeds. `--bank-seed` controls the endpoint bank independently.
`--partition-study` cannot be combined with `--with-nsot` or individual partition
overrides. For an individual variant rather than the three-way study:

```powershell
python run_multiscale.py --source-partition axis --target-partition pca --stage all --device cuda
```

The corresponding optional YAML fields, inside `multiscale_wasserstein`, are:

```yaml
source_partition: axis
target_partition: pca
pca_eigengap_tol: 0.000001
```

For an individual YAML, use a distinct cache path for each variant. The runner
automatically creates separate content-addressed paths. Historical axis/axis
caches remain loadable; a PCA cache cannot be loaded as an axis baseline or with
a different PCA tolerance. Nothing overwrites an existing incompatible cache.

**PCA definition.** At every recursive split, compute the centered covariance in
float64 and sort by the leading principal-component projection. Retain the original
capacity split formula, original-index tie breaking, and parent-to-child hierarchy.
For example, a five-way split recursively allocates two groups to the left and
three to the right; it is not an independent five-cluster fit. Fix eigenvector
sign by making its largest-absolute component positive. If the leading eigenvalue
is zero or the relative top eigengap is at most `pca_eigengap_tol`, use the original
widest-coordinate-axis split. Split counts, fallback counts and the minimum observed
relative eigengap are recorded separately for source and target. This is a
deterministic degeneracy rule, not a learnability penalty or a claim of complete
rotation invariance. PCA also changes range-based direction selection to a
variance-based direction, not just the allowable angle.

Only temporary sorting projections are centered/projected. **All OT costs and
training endpoints use the unchanged original coordinates.** Every patch remains
equal-sized, every fine patch has an actual ancestor at each level, and final
matching remains exact leaf assignment followed by exact within-leaf assignment.

Before training, the runner checks equality of the actual source and target arrays
across all study variants. Bank construction does not consume the training RNG.
For a fixed dataset/model seed, variants use the same initialization seed,
with-replacement pair-index RNG and uniform-time RNG; target partners differ.
Evaluation uses the same fresh Gaussian protocol. Numerical reproducibility across
different GPU/software environments is not guaranteed by common seeds alone.

Study outputs beside its manifest:

- `summary.json`: complete evaluator metrics, provenance and timings.
- `comparison.csv`: CD, leakage, histogram JS, checkerboard mass error, horse ROI
  metrics, training/inference times, and exact artifact paths.
- `bank_comparison.csv`: endpoint hashes, coupling costs, preparation times and PCA
  split/fallback diagnostics. Repeated model seeds refer to the same bank; use
  `summary.json` for unique-bank compute totals.
- `<dataset>_seed<seed>_nfe_<NFE>_comparison.png`: left-to-right `axis_axis`,
  `axis_pca`, `pca_pca`, assembled from the **original evaluator PNGs**, without
  rerunning inference or replacing the rendering style. Generated once all three
  variants for that dataset/seed/NFE exist.

Original per-model JSON/PNG files remain in `eval_results/<dataset>/`; checkpoints
remain in `runs/<dataset>/`. Completed runs/evaluations are hash-verified and reused.
An interrupted training run is restarted (the existing trainer does not save
optimizer-resume checkpoints). Printed paths are resolved on the executing lab
machine; no developer-machine path is required in the commands or templates.

Judge NFE 1 and high-NFE generation jointly using quality metrics and images.
Lower transport cost, compact patches or lower fitting MSE alone do not establish
an improvement. The existing four-way coupling audit also reads the configured
source partition, so PCA checkpoints are not incorrectly diagnosed with axis leaves.

### Original exact matching steps

1. Draw one standard Gaussian source superset and one target superset with the same isolated sampler and cache seed as the existing 2D NSOT implementation. Keep both arrays unchanged.
2. Independently build **nested balanced spatial partitions** of the two point sets. Defaults contain 4, 20 and 100 patches. Each patch at a level has equal mass; finest patches contain 100 points for M=10,000.
3. For every source/target patch pair at each active level, compute empirical squared Wasserstein distance from all original points in those patches. Equal-mass point assignment is solved exactly with squared Euclidean ground cost. No centroid-only approximation, Sinkhorn regularization or subsampled Wasserstein score replaces this calculation.
4. For each finest-patch pair, average the Wasserstein costs of their ancestor patch pairs across levels using the configured scale weights normalized to sum to one. Solve an exact assignment between finest patches for that fixed additive multiscale cost.
5. Within every selected finest-patch pair, solve exact pointwise OT. Assemble those bijections into one full source-to-target permutation.

Writing `a_l(i)` and `b_l(j)` for the level-l ancestors of finest patches i,j, the fixed patch cost is

```text
C[i,j] = sum_l weight[l] * W2_squared(source_patch[a_l(i)], target_patch[b_l(j)])
```

The finest patch assignment minimizes the sum of these costs. Every source and target point occurs once in the full stored permutation. Training samples pair indices iid with replacement, as in the local NSOT baseline. Inference starts from **fresh standard Gaussian** points; neither target patches nor target points are needed at inference.

### What exact does and does not mean

- Each patch-pair empirical W2 problem and the finest-patch assignment are solved exactly for their specified costs, to floating-point solver precision.
- The assembled point permutation is **restricted by the selected patch blocks**. It is not the global exact OT permutation for the original 10K point sets and is not a global optimum over all multiscale pointwise couplings.
- Coarse levels contribute costs; they are not independent bijections that constrain all child patches to stay inside a single matched coarse parent. The final bijection is between finest patches.
- Balanced partitions guarantee finite empirical marginal preservation. Finite cached points still only approximate population Gaussian and target marginals; fresh-Gaussian inference can still generalize poorly.
- Matching uses original coordinates and scales, not centered/whitened local shapes. Equal scale weights do not imply equal numerical contributions because patch costs have different magnitudes. These weights are a prespecified first test, not a learned or proven optimum.
- No proof or result promises that multiscale matching reduces conditional variance, Jacobian norm, overfitting or final CD. Those are empirical questions.

## Cost and timing

All assignment preprocessing runs on CPU; using an A100/H200 for training does not automatically speed up SciPy preprocessing. Default levels require 4^2+20^2+100^2=10,416 patch-pair solves, followed by the finest-patch assignment and the selected fine point assignments. The largest default patch-pair cost matrix has 2,500^2 float64 entries, approximately 47.7 MiB. The memory guard applies to an individual dense cost matrix, **not** total process peak RAM or total runtime.

This is not guaranteed faster than one global NSOT solve: many exact solves, Python overhead and matrix allocations can make preprocessing substantial. Its duration is measured rather than predicted. Preparation is sequential; do not launch multiple large preparations merely to fill GPU slots. Optional global NSOT M=10,000 preparation allocates a 10,000^2 float64 cost matrix, approximately 762.9 MiB, plus solver/process overhead.

The experiment manifest and `summary.json` retain bank preprocessing seconds, each model's online training seconds and each NFE's inference seconds. Preprocessing is shared once per unique bank; beta/seed copies must not multiply it when reporting total compute. Existing inference timing excludes rendering and quality-metric computation.

## Outputs and interpretation

- `experiment_manifests/multiscale_.../manifest.json`: exact jobs, hashes and endpoint equality checks.
- The same directory's `summary.json`: per-bank preprocessing time, per-model training time, complete evaluator JSONs and image paths.
- `coupling_cache/multiscale_wasserstein/`: validated content-addressed multiscale banks.
- `runs/checkerboard/` and `runs/horse/`: standard checkpoint, saved `config.yaml`, and `training.json`.
- `eval_results/`: standard per-NFE JSON and PNG outputs; horse results retain fixed thin-structure/gap ROI metrics.

For the first comparison, look at NFE=128 CD **and** leakage/density/horse thin-structure metrics, then the full NFE curve and compute. A lower transport or patch score is not itself success. One seed is a pilot; repeat independent bank/model seeds before a general claim. Compare results at matched N=256 and evaluation settings rather than mixing the smoke artifacts with full results.

Dependencies are the repository's existing Python environment (PyTorch, NumPy, SciPy, PyYAML, scikit-image and existing evaluator dependencies). No extra package is introduced by the runner.
