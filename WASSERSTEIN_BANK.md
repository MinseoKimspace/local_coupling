# Wasserstein coupling-bank selection experiment

This is a new 2D experiment independent of TG. It tests whether a score on a
complete coupling bank predicts velocity ambiguity and actual neural learnability.
It does not assert that Wasserstein consistency controls conditional variance or
Jacobian norms. Full learning runs should be performed on the experiment machine.

## Run

Run from the repository root, with the existing requirements installed.
Preparation is the default stage: it creates banks/reports/configs, **not training**.

```powershell
# Tiny CPU check: M=64, N=16, two updates, then fresh-Gaussian evaluation.
python run_wasserstein_bank.py --smoke --stage all --nfe 1 2
```

```powershell
# A/B/C plus random-bank, distance and local-variogram selection controls.
python run_wasserstein_bank.py --stage prepare --selectors random transport displacement joint multiscale variance --seeds 0 1 2 --device cuda
```

The program prints `manifest=.../manifest.json` and the exact continuation command.
Keep this manifest to resume completed jobs without preparing a different bank.

```powershell
python run_wasserstein_bank.py --manifest "experiment_manifests/PRINTED_DIRECTORY/manifest.json" --stage train
python run_wasserstein_bank.py --manifest "experiment_manifests/PRINTED_DIRECTORY/manifest.json" --stage eval --nfe 1 2 4 8 16 32 64 128
```

Or perform preparation, training and evaluation in one invocation:

```powershell
python run_wasserstein_bank.py --stage all --selectors random transport displacement joint multiscale variance --seeds 0 1 2 --device cuda
```

Dataset variants use the original model/training templates for checkerboard/horse:

```powershell
python run_wasserstein_bank.py --dataset checkerboard --stage all --selectors random transport displacement joint multiscale variance --device cuda
python run_wasserstein_bank.py --dataset horse --stage all --selectors random transport displacement joint multiscale variance --device cuda
```

Without `--selectors`, only A/B/C run. Without `--seeds`, training seed 0 runs.
All selectors/seeds in an experiment share one preparation; its time must not be
charged once per model when reporting total experiment cost. To test bank-seed
robustness, change `cache_seed` in a copy of the YAML and prepare another experiment.
With `--manifest`, saved settings are used; only stage and evaluation NFEs can change.

## Bank and selection

- Default: M=2,048 superset points, N=256 points per training cloud, eight candidates.
- Every candidate contains **every target shape with equal probability**. A complete
  bank is selected once for each criterion. Source-wise target dropping is not used.
- All candidates share exactly the same source and target arrays. One common Gaussian
  source superset is used for all shapes, avoiding shape-specific source-bank shifts.
- Training draws one shape for each cloud, then samples paired indices independently
  **with replacement** from its selected superset map. The network receives only
  the current point cloud and time, never the shape ID or candidate ID.
- The default mixture contains filled disk, annulus, and two separated disks. Fresh
  targets use the same analytic laws. These are three shape modes, not three separate
  training models. Single-horse/checkerboard runs do not test ambiguity between shapes.
- Candidate families: random permutation, exact Euclidean assignment, three projected
  sorting assignments, and three assignments with noise in the **cost construction**.
  The latter never modify the stored/training coordinates. No TG partition is required.
- Validation repeats these candidate families on independently drawn supersets. The
  candidate ID selected on the selection split is retained; validation does not reselect.
  This tests new points in the same modes, not generalization to unseen shape modes.
- Superset mappings are fixed; training subsets are freshly sampled. Finite banks only
  approximate population Gaussian/target laws. Inference uses fresh iid Gaussian noise.

The A/B/C scorers compare uniformly sampled subsets of one candidate's empirical
feature distribution with that candidate's complete **scoring pool**, not its entire
dense superset. Pool indices, subset draws and projection directions are shared across
candidates. The pool uses the same number of points from each shape.

- A (`displacement`): mean W2 squared of displacement subsets, subset size 32.
- B (`joint`): same score on `(x_t, alpha*u)`, size 32. Default `joint_times: [0.0]`
  gives the proposed `(x_0, alpha*u)` definition. To test path-aware scoring, prepare a
  separate configuration with e.g. `[0.25, 0.5, 0.75, 0.9]`. `alpha=velocity_weight`
  is fixed across candidates; features are not normalized separately per candidate.
- C (`multiscale`): joint scores at sizes 8/32/128, each divided by the same split's
  random-candidate reference, then equally averaged. No scale may cover the whole pool,
  which would give an identically zero consistency score.
- `transport`: lowest mean squared displacement over all superset points and shapes.
- `random`: one uniformly drawn candidate bank, not necessarily the random-permutation
  candidate. Candidate 0 is always the random-permutation scale reference.
- `variance`: smallest time-integrated **point-position velocity variogram proxy**.

`backend: exact` uses POT's weighted empirical W2 squared (and checks convergence).
`--backend sliced` uses explicitly labelled sliced W2 squared with shared directions;
it is a different approximate criterion. Compare candidate rankings before scaling up.
Dense candidate assignment is protected by `max_cost_mib`; the default experiment
is not a 15K/PSF implementation. Existing TG/NSOT experiments remain separate baselines.
The transport selector uses an exact bank without NSOT's hybrid Gaussian perturbation,
so it must not be labelled a full NSOT reproduction.

## What is measured

The local velocity proxy is a row-normalized kernel average of
`||u_i-u_j||^2 / (2*point_dim)` for nearby `x_t`, excluding self-pairs. It reports three
bandwidths, effective neighbor counts, coverage, global values across shapes, and
within-shape values. Their difference is **not** an exact variance decomposition.
Finite bandwidth also penalizes legitimate mean-field variation. Point-position
conditioning does not include all information available to the point-set Transformer.
Consequently this proxy is not the model's exact irreducible FM loss.

Reports include Spearman correlations and residual rank correlations controlling for
transport cost. With eight candidates these are exploratory, not significance evidence.
Selection and independent-superset reports use separate scoring draws. Actual training
records train loss and fixed held-out coupling validation MSE by time, preserving
training RNG and model mode. This validation risk includes both conditional ambiguity
and approximation error; it cannot isolate them by itself. Candidates may legitimately
tie or A/B/C may select the same bank; no artificial differences are introduced.

Artifacts under the printed experiment directory:

- `bank_scores.json`, `candidate_scores.csv`, `bank_diagnostics.png`;
- one generated training YAML per selector/seed;
- `manifest.json` with cache hash, selected candidate IDs and completed run locations;
- `learning_curves.png` and, after evaluation, `summary.csv`.

Each saved run contains verified checkpoint/config, `training.json`, and
`learning_curve.json`. Standard checkerboard/horse evaluators work on those saved
run configs. Mixture evaluation uses `eval_wasserstein_bank.py`, fresh stratified
reference clouds and CD-based MMD/COV/1-NNA, plus a nearest-reference shape-mass proxy.
MMD means Minimum Matching Distance. 1-NNA should be interpreted relative to 0.5;
larger coverage is better. Mixture plots show individual clouds, not just pooled density.
Inference timing excludes warmup, metrics and rendering. Online training time includes
scheduled validation; preparation time is reported separately.

The existing full-cloud Jacobian audit also accepts the new banks, including mixture2d:

```powershell
python audit_jacobian.py "runs/mixture2d/PRINTED_RUN/config.yaml" --dataset mixture2d --scope both --clouds 16 --probes 8
```

`rollout` uses fresh Gaussian inference; `coupling` measures paths from the selected
training bank. Neither scope's Jacobian is the exact conditional velocity variance.

Select on the selection split, inspect robustness on independent supersets and multiple
bank seeds, and judge generation at both low and sufficiently high NFE. A lower score
or validation MSE alone does not establish better final density or thin structure.
