# Same-size target bank mass-control ablation

This is a separate MS-W2 experiment, not NSOT or cross-leaf refinement.
Original configs, caches, runs and evaluators are not overwritten.

## Run on the lab computer

Run from the repository root. No machine-specific paths are required.

```powershell
python run_bank_mass_control.py --stage all --device cuda
```

This prepares and trains **four models**: IID and stratified-target MS-W2 on
checkerboard and horse. Default M=10,000, N=256, bank seed 0, training seed 0,
10,000 updates, original dataset-specific architecture/optimizer, and Euler
NFE 1, 2, 4, 8, 16, 32, 64, 128. No source noise is injected.

To first inspect the full-size banks **without OT or training**:

```powershell
python run_bank_mass_control.py --stage audit
```

Keep its printed `experiment_manifest` path and resume the same study:

```powershell
python run_bank_mass_control.py --manifest "experiment_manifests/bank_mass_control_TIMESTAMP_ID/manifest.json" --stage all
```

Replace `TIMESTAMP_ID` with the actual printed name. There is no latest-run
guessing. `--stage prepare`, `--stage train`, and `--stage eval` are also
available. `eval` requires an existing trained manifest. Completed artifacts
are checked by SHA256 before reuse. Failed/interrupted optimizer runs start
over; only completed training jobs are reused. For another grid/bank/training
configuration, start a new study rather than modifying a saved YAML.

Optional repeatability check and small pipeline test:

```powershell
python run_bank_mass_control.py --stage all --device cuda --seeds 0 1 2
python run_bank_mass_control.py --stage all --device cuda --bank-seed 1
python run_bank_mass_control.py --smoke --stage all
```

The first command trains 12 models sharing four prepared banks. Training seeds
test optimization variation; `--bank-seed` changes the endpoint banks and must
be varied separately to assess bank variation. Smoke is M=64, N=16, CPU, a tiny
network, two updates and NFE 1/2: it is NOT a quality experiment.

## Controlled change

Both conditions use byte-identical original Gaussian source coordinates. The
sole designed change is offline target sampling. M, MS-W2 matching algorithm,
partition counts [4,20,100], weights, training seed/settings and evaluation
settings are matched. MS-W2's data-dependent target partitions and permutation
are recomputed: their induced changes are part of the treatment, not held fixed.

The target mass grid is **not** the MS-W2 patch hierarchy:

- Checkerboard: a fixed 16x16 spatial grid on [-1,1]^2; each of the eight active
  squares contains 4x4 subregions. There are 128 positive-area regions.
- Horse: 16x16 rectangular blocks of the original mask, with edges rounded to
  native pixel boundaries. Region probability is foreground pixel count in the
  block divided by total foreground pixel count, NOT box area or equal mass per
  anatomical part. This is exact area for the existing continuous pixel sampler.

For probability p_r, allocate floor(M p_r) points, then give remaining points
to the largest fractional remainders. A bank-seeded random tie break avoids
systematic top-left preference. Counts sum to M and each region differs from
M p_r by less than one point (up to floating-point tolerance). Zero-area regions
receive zero points. Each allocated point is drawn uniformly from foreground
area inside its region. The array is shuffled before matching. Source coordinates
are neither rotated nor perturbed. Within-pixel uniform coordinates are the
original target distribution, NOT additive noise on existing bank points.

Quotas apply to the **whole target bank**, not each training/generated cloud.
Pair indices are still sampled IID with replacement. Matching only permutes
the target, so it preserves the constructed finite bank's mass. No barycentric
projection, online reweighting, new regularizer or inference-time correction.
The 16x16 regional bound does not guarantee an exact quota for the union of
multiple bins (e.g. a whole checker square), a horse ROI that cuts across bins,
or smaller-scale point spacing. More/finer bins are not automatically better.
`--bins 8` starts a different, coarser ablation; checker bins must divide into
whole checker cells (be a multiple of `data.grid_size`).

The IID path remains byte-identical to the original bank generator. Stratified
cache specs and filenames include the sampling mode/grid and implementation
dependency; IID and stratified caches cannot be loaded interchangeably.

## Outputs and interpretation

- `experiment_manifests/bank_mass_control_.../`: immutable job YAMLs, manifest,
  and summary. The manifest lists exact bank, checkpoint and evaluation paths.
- `coupling_cache/multiscale_wasserstein/`: separately hashed bank caches.
- `runs/{dataset}/...`: original training artifacts; checkpoint names distinguish
  `multiscale_iid` from `multiscale_stratified`.
- `eval_results/{dataset}/...`: **original evaluator PNGs and JSONs**, with a
  stratified label added to the controlled condition's title.
- `eval_results/bank_mass_control/bank_mass_control_.../`: `bank_audit.json`,
  `summary.json`, and `comparison.csv` (empty except header until evaluation).

`bank_audit.json` reports both banks' endpoint hashes, exact quotas, regional
TV, worst regional count error, leakage, standard histogram JS and dataset
mass/ROI metrics. This can establish that the bank manipulation succeeded,
but cannot establish improved learned generation. It does not perform OT.

Generated results are evaluated against the **unchanged fresh IID target
distribution**, NOT against the stratified bank. Same training seed gives the
same initial model and pair-index/time RNG sequence within each paired dataset
configuration; GPU kernels are not promised bitwise deterministic. Evaluation
uses the same fresh Gaussian source/target RNG protocol for both conditions.

Read NFE 1 and 128 first, then the rest of the curve. Compare histogram JS,
checker cell mass error, horse thin-region/ROI mass errors, leakage and CD
together. `comparison.csv` is paired by dataset/training seed/NFE, with
delta = stratified - IID. Negative is better for errors, not necessarily raw
ROI mass. Timing columns use the standard evaluator's integration-only scope;
bank preparation times are recorded separately, once per unique cache.

Improved bank TV is expected by construction. Improved generator JS/ROI errors
without worse leakage or low-NFE CD would support this intervention. Better
bank TV with unchanged generated density would point toward matching/learning
effects rather than finite target sampling alone. One bank/training seed does
not establish statistical significance or eliminate finite-bank overfitting.
