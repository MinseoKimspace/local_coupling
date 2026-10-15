# Multiscale Wasserstein matching with a 10,000-point bank

## Question and scope

Does matching **distributions at multiple spatial scales** improve the coupling learned by the existing checkerboard and horse flow-matching models?

This is a new coupling construction, not the removed candidate-bank/selector experiment. No displacement/joint/multiscale score chooses among eight candidates. No FGW structural term, new prior, model conditioning, waypoint, Jacobian penalty, or online perturbation is added to the multiscale method.

Default: one multiscale model per dataset, bank size **M=10,000**, 256 points per training cloud, seed 0, 10,000 optimizer updates, original dataset-specific model/training/evaluation settings. M counts individual bank points, **not** 10,000 clouds. The bank is larger than the removed 2,048-point experiment, while matching the repository's 2D NSOT baseline size. This is not the NSOT paper's main 100K-point experiment.

## Run on the experiment machine

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

## Optional sparse cross-leaf refinement comparison

The original multiscale method and default commands are unchanged. To compare it with a refined bank from scratch:

```powershell
python run_multiscale.py --with-refinement --stage all --device cuda
```

This trains **four models**: original multiscale and refined multiscale on checkerboard and horse. It changes only the fixed offline pairing bank. Both branches retain the same original endpoint arrays, M, architecture, optimizer, learning rate, steps and training seed. Inference still starts from fresh Gaussian points with no patch labels. Add `--with-nsot` only if the additional NSOT beta=0/0.2 controls are desired (eight total models).

### Reuse existing completed experiments

To keep your already trained baseline model and results, use its original manifest:

```powershell
python run_multiscale.py --manifest "PATH_TO_ORIGINAL_MANIFEST" --add-refinement --stage all
```

This creates a **new** manifest referencing the verified original bank/checkpoint/evaluations and adds refined jobs. The original manifest, YAMLs and baseline artifacts are not overwritten. Completed baseline training/evaluations are reused; only missing jobs/results run. If the original manifest was only prepared but not trained, its baseline still needs training. The cloned experiment uses the original saved device and training settings, so do not add `--device`, `--steps`, or other baseline overrides.

The runner prints the new `cloned_manifest=...` path. To continue that comparison, resume the **new** path without `--add-refinement`:

```powershell
python run_multiscale.py --manifest "PATH_TO_CLONED_MANIFEST" --stage all
```

Preparation only and the zero-refinement identity control:

```powershell
python run_multiscale.py --manifest "PATH_TO_ORIGINAL_MANIFEST" --add-refinement
python run_multiscale.py --with-refinement --refinement-neighbors 0 --stage all --device cuda
python run_multiscale.py --smoke --with-refinement --stage all
```

`--refinement-neighbors` defaults to 8 and is set only when creating/adding refinement. A resumed comparison keeps its saved setting. For a different neighbor count, create another comparison from the original baseline manifest rather than editing a prepared YAML.

### Refinement algorithm and scope

- Start from the exact prepared baseline permutation. Preserve **all point-to-point candidate edges inside its matched finest patch blocks**, not just the selected permutation edges.
- In source space, query each point's k nearest other source points and retain those lying in different source leaves. Add symmetric swapped-target edges using their existing partners. Repeat in target space using the inverse permutation and different target leaves. This filters ordinary kNN; it does not search until k cross-leaf neighbors have been found.
- Solve one full sparse bipartite assignment on the combined candidate graph. Keeping the original edges guarantees that a feasible bijection exists. Target coordinates are neither averaged nor moved. `neighbors=0` returns the exact baseline permutation as a no-refinement control.
- The cost is the normalized weighted sum of original non-leaf ancestor-patch W2 costs plus the finest weight times pointwise squared Euclidean distance. On the original block-optimal assignment, this equals the baseline multiscale objective. Refinement relaxes the leaf-block restriction; it does not silently replace that objective by global Euclidean OT or add a second copy of the finest-scale cost.

This is **offline sparse-support refinement**, not a training-time proximal update, stochastic diffusion, or a reproduction of the complete Gerber--Maggioni algorithm. It does not implement the paper's scale-by-scale capacity propagation or potential-based support search. “Exact” here means optimal assignment **on the retained graph for this specified cost**, not global unconstrained Wasserstein OT. A non-increasing objective does not guarantee better CD, smaller Jacobian, or fewer visual artifacts.

The runner verifies byte-identical original source/target arrays, the original permutation hash and the exact baseline cache file hash. Refined cache identity includes that baseline file hash, refinement parameters, and the refinement/cost implementation hashes. Reusing a baseline manifest does not regenerate its original bank just because source code has since changed.

At M=10,000 and 100 finest leaves, retaining all original blocks uses 1,000,000 edges. With k=8, the conservative pre-deduplication upper bound is 1,320,000 edges, below the default 2,000,000 guard. Sparse does not mean linear-time or zero-overhead: graph storage, exact sparse assignment and recomputed coarse patch costs still cost CPU time/memory. The edge guard is not a total process memory limit.

### Paired report and timing

`summary.json` adds `paired_refinement` entries for each matching dataset/seed/NFE. They include baseline/refined CD, leakage, histogram JS, checkerboard mass error and all available horse ROI metrics, with **refined minus baseline** deltas. Negative is favorable for error metrics, but not automatically for raw ROI mass. Metric/evaluation-setting mismatches suppress quality deltas rather than presenting an unfair comparison.

Baseline preprocessing is counted once per unique original bank. Refined `precompute_seconds` measures **incremental** loading/coarse-cost reconstruction/graph refinement only; stored `base_precompute_seconds` is not counted a second time. `unique_precompute_seconds` sums distinct base and incremental preparation costs across the comparison, not wall-clock time spent during a resumed command. Training and per-NFE inference times remain separate. These are single-run comparisons unless multiple seeds are requested; no statistical improvement is claimed automatically.

### Optional matched Jacobian audit

The standard FM and full-cloud Jacobian diagnostic tools also support the refined method. Jacobian measurement is **not** run automatically by `--stage all`. After training, use the saved run configs referenced by the comparison manifest, not the pre-training experiment YAMLs:

```powershell
python audit_jacobian.py "BASELINE_RUN/config.yaml" --dataset horse --scope both --clouds 16 --probes 16 --rollout-steps 256 --seed 2026
python audit_jacobian.py "REFINED_RUN/config.yaml" --dataset horse --scope both --clouds 16 --probes 16 --rollout-steps 256 --seed 2026
```

Replace both placeholder paths with their actual saved run directories; for checkerboard, use its two run configs and `--dataset checkerboard`. Keep the audit seed, N, cloud count, probes, times and rollout resolution equal across methods. `scope=both` requires each method's own training cache; `scope=rollout` and normal generation evaluation do not. The comparison runner itself still verifies both banks when resuming a manifest.

Inspect mean, median and per-cloud outliers separately for coupling and rollout. A smaller Jacobian alone is not a success criterion: require preserved low-NFE quality and improved high-NFE CD/leakage/thin-region metrics. The paper motivates support expansion, not a theorem linking this refinement to flow quality: [Gerber and Maggioni, 2017](https://arxiv.org/abs/1708.02469). The sparse assignment implementation uses [SciPy's full bipartite matching solver](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csgraph.min_weight_full_bipartite_matching.html).

## Diagnose original MS-W2: quality and Jacobian on the same clouds

`audit_ms_w2_clouds.py` is a **read-only, original-MS-W2-only** diagnostic. It accepts an existing
`audit_jacobian.py` JSON with `rollout` or `both` scope, replays those exact Gaussian clouds,
and joins their generation quality with their existing rollout Jacobians. It does not train,
change coupling, load a training cache, or run the cross-leaf experiment.

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
