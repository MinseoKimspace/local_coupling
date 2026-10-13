# Multiscale Wasserstein matching with a 10,000-point bank

## Question and scope

Does matching **distributions at multiple spatial scales** improve the coupling learned by the existing checkerboard and horse flow-matching models?

This is a new coupling construction, not the removed candidate-bank/selector experiment. No displacement/joint/multiscale score chooses among eight candidates. No FGW structural term, new prior, model conditioning, waypoint, Jacobian penalty, or online perturbation is added to the multiscale method.

Default: one multiscale model per dataset, bank size **M=10,000**, 256 points per training cloud, seed 0, 10,000 optimizer updates, original dataset-specific model/training/evaluation settings. M counts individual bank points, **not** 10,000 clouds. The bank is larger than the removed 2,048-point experiment, while matching the repository's 2D NSOT baseline size. This is not the NSOT paper's main 100K-point experiment.

## Run on the experiment machine

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
