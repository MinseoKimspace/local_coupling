# Wasserstein bank scores and coupling learnability

## Question, hypothesis, and decision rule

**Question:** Do scores computed before training predict which coupling bank a
network learns well, beyond what squared transport cost already predicts?

**Hypothesis:** On common source/target supersets and the same candidate set,
joint position–velocity Wasserstein scores predict held-out FM risk and final
generation quality. TG is an external comparison method; no TG partition is
required. A low Wasserstein score is not assumed to be low conditional variance.

**Primary scores:** `joint` at t=0 and `path_joint` integrated over the path.
Other scores, alpha sweeps, and bandwidth diagnostics are secondary.

**Primary endpoints, separately for checkerboard and horse:**

- Final independent-bank validation FM MSE, integrated uniformly over t in [0,1]
  with the trapezoidal rule on the saved nonuniform time grid.
- Chamfer distance at NFE 128, using the existing dataset evaluator.

**Decision rule:** A full study needs at least three complete bank seeds and all
planned candidate × training-seed jobs. For a predeclared primary score, the same
at least `ceil(2 * number_of_bank_seeds / 3)` bank seeds must show both lower mean
selector regret than transport selection and positive transport-controlled rank
correlation, for both primary endpoints. This is provisional exploratory support,
not a significance test or universal guarantee. Ties with transport, disagreeing
endpoints, or incomplete coverage do not pass. Secondary scores cannot replace a
primary after seeing results. Pilot and smoke runs cannot support the primary claim.

No generated-quality improvement is asserted by the implementation or its tests.

## Run the checkerboard and horse study

Run from the repository root on the experiment machine with the existing Python
requirements installed. Both datasets run separately. Every candidate is trained;
selectors are retrospective table lookups, without duplicate training per selector.

Preparation only is the default. It creates selection/validation banks and score
reports, saves job configurations, and prints `study_manifest=.../manifest.json`:

```powershell
python run_wasserstein_study.py
```

The full default is **2 datasets × 8 candidates × 3 bank seeds × 3 training seeds
= 144 models**, each with 10,000 optimizer updates. Preparation does not train
them. To execute the full study explicitly:

```powershell
python run_wasserstein_study.py --stage all --device cuda
```

Start with a pilot comparing random assignment, exact OT, and cost-noise amplitude
0.2 on both datasets:

```powershell
python run_wasserstein_study.py --pilot --stage all --device cuda
```

The pilot defaults to one bank seed and one training seed: **six full 10,000-update
models**. It reduces candidate and seed counts, not training length. Add
`--steps 1000` for a deliberately shorter pilot; that changes the original budget.

A tiny CPU pipeline check runs both datasets, three candidates each, two updates,
and NFE 1/2. It is not a quality experiment:

```powershell
python run_wasserstein_study.py --smoke --stage all
```

Keep the printed study manifest and resume without regenerating banks:

```powershell
python run_wasserstein_study.py --manifest "experiment_manifests/PRINTED_STUDY/manifest.json" --stage all
python run_wasserstein_study.py --manifest "experiment_manifests/PRINTED_STUDY/manifest.json" --stage train
python run_wasserstein_study.py --manifest "experiment_manifests/PRINTED_STUDY/manifest.json" --stage eval
python run_wasserstein_study.py --manifest "experiment_manifests/PRINTED_STUDY/manifest.json" --stage report
```

With a study manifest, only `--stage` changes. Saved dataset, seeds, candidate
family, budget, and NFE settings remain fixed. `report` reads existing artifacts
and regenerates summaries without training. Incomplete coverage is reported.

Optional axes are `--datasets checkerboard` or `--datasets horse`,
`--bank-seeds 0 1 2`, and `--seeds 0 1 2`. Bank and training seeds are separate.
The standard NFE sweep is `1 2 4 8 16 32 64 128`; a custom sweep without 128 leaves
the primary generation endpoint incomplete.

## Candidates and comparison conditions

| Default ID | Assignment |
|---|---|
| 0 | Uniform random permutation |
| 1 | Exact squared-Euclidean assignment |
| 2, 3, 4 | Projection sorting at 0, pi/4, pi/2 |
| 5, 6, 7 | Exact assignment with Gaussian cost noise of amplitude 0.05, 0.2, 0.5 |

The optional ordered family uses random, exact, and eight cost-noise amplitudes:
`0.01, 0.025, 0.05, 0.1, 0.2, 0.35, 0.5, 1.0`.

```powershell
python run_wasserstein_study.py --family cost_noise --stage prepare
```

Its full default is **180 models**. Amplitudes share Gaussian draws within each
bank split, so only the magnitude changes. Noise modifies assignment costs while
stored and training coordinates stay unchanged. This is not NSOT's training-time
hybrid source perturbation. No K-selection sweep is included.

Candidates within a bank seed share identical source/target arrays, and each
bijection uses every endpoint once. Validation uses independent supersets with
the same candidate families; the selected ID is never reselected on validation.
Training freshly samples paired indices with replacement from a fixed bank.
Inference uses fresh iid Gaussian points, without a bank or candidate ID.

Checkerboard/horse YAMLs match the existing TG/NSOT comparison's data sampler,
N=256, batch size 64, model, optimizer, 10,000-update budget, and evaluator:

- Checkerboard: width 128, 2 Transformer layers, learning rate 0.001.
- Horse: width 256, 4 Transformer layers, learning rate 0.0003.
- Both: AdamW, weight decay 0.01, dropout 0, float32, standard renders and NFE sweep.

Data exposure is not identical: **this bank uses M=2,048; existing NSOT YAMLs use
M=10,000**. Bank sampling and coupling also differ. The study records baseline
template hashes and configuration matches, but does not retrain TG/NSOT baselines.
Use their existing evaluator outputs for external comparison and report bank size.
Finite empirical banks approximate, rather than exactly equal, population laws.

## Scores and mathematical audit

Wasserstein scores compare uniform subsets with the full **scoring pool**, not
the entire dense superset. Default pool size is 256 per shape and subset sizes
are 8/32/128. Pool indices, subset draws, and projection directions are shared
across candidates.

- `transport`: full-bank mean squared displacement.
- `displacement` (A): mean subset-to-pool W2 squared on u, subset size 32.
- `joint` (B): the same score on `(x_t, alpha*u)`, size 32. Default
  `joint_times: [0.0]` uses source position.
- `path_joint`: trapezoidal time integral of the joint score over
  `variance_times: [0, 0.1, 0.25, 0.5, 0.75, 0.9, 1]`, divided by interval length.
  It approximates uniform-time expectation, not the unweighted grid mean.
- `multiscale` (C): joint scores at 8/32/128, normalized by the same split's
  random-permutation-candidate score at each scale and equally averaged.
- `variance`: integrated point-position local velocity variogram proxy.
- `random`: uniformly selected candidate bank; it need not be candidate 0.

`alpha_values: [0.25, 0.5, 1, 2, 4]` audits t=0 joint rankings with common draws.
Alpha is not chosen using final performance. For exact W2 squared, alpha tending
to zero gives the common source-only score and `B(alpha)/alpha^2` tends to A at
large alpha. B itself does not tend to A. With positive-time positions, the small
alpha scores can differ because x_t differs between candidates.

For unchanged full-bank endpoints, mean displacement is fixed and
`transport = ||mean(u)||^2 + trace(Cov(u))`. Reports include these moments and
score–transport correlations. A scoring pool's mapped target subset varies by
candidate, so its mean displacement need not be fixed. A is not assumed to equal
transport; the collinearity audit measures redundancy.

The exact t=0 fixed-bank Bayes floor is separate. With S uniform shapes, shared
source indices k, and displacement u[s,k], its coordinate-averaged value is:

```text
R0 = mean over s, k, and coordinates of (u[s,k] - mean_s u[:,k])^2.
```

Shape and index draws are independent, so the source cloud gives no shape label.
Identical source coordinates are pooled by the implementation. **For a single
checkerboard or horse distribution with distinct source coordinates, R0 is zero
for every candidate, including a random finite lookup.** It is marked uninformative
for ranking and does not imply easy neural learning. The formula applies exactly
at t=0 on a fixed finite bank, not at small positive time or to a population of
freshly regenerated couplings.

The variogram averages `||u_i-u_j||^2/(2*point_dim)` with row-normalized Gaussian
weights on nearby x_t and excludes self-pairs. It reports bandwidths 0.05/0.1/0.2,
effective neighbors, coverage, and global/within-shape values. Point-position
conditioning omits cloud context. Finite bandwidth includes valid mean-field
variation; at fixed finite sample size, shrinking bandwidth selects nearest
neighbors. A nonzero plateau is not proof of irreducible ambiguity. h and h-squared
asymptotic interpretations require extra dense-sample regularity assumptions and
are not asserted here.

`backend: exact` uses empirical W2 squared with POT and checks convergence.
`--backend sliced` is explicitly labelled sliced W2 squared: a different criterion
with shared directions. Dense assignments have a memory guard. This is a 2D
experiment, not a 15K/PSF implementation.

## Learnability, timing, and output

All candidates are trained before selector results are looked up. Reports provide
Spearman score–performance correlations, transport-controlled rank correlations,
and regret relative to the retrospective endpoint-specific best candidate.
Training seeds are paired, and bank seeds are reported separately. A/B/C selecting
the same candidate is a valid outcome. Eight candidate methods are not eight
independent statistical replicates; no p-values or significance claims are made.

Validation is recorded at logging steps, 5% of updates, and the final update with
isolated RNG and model evaluation mode. Independent-bank FM risk includes transfer
to regenerated correspondences and does not isolate approximation error. Fresh
generation CD is a separate primary endpoint.

The early-validation selector uses only records at or before 5% of updates.
Every candidate must be trained to that point; the report sums updates and online
seconds. This is not a free pretraining score. Early and final validation reuse
the holdout bank, which can favor this selector on the validation endpoint. The
oracle is retrospective and cannot be deployed without knowing the outcomes.

Preparation time, online time including validation, validation time, and online
time excluding validation are reported separately. Validation is observational
but has nonzero overhead. Shared bank preparation is not charged once per model.
Inference timing follows the existing evaluator and excludes warmup, metrics,
and rendering.

Root study artifacts:

- `manifest.json`: fixed axes, protocol references, claim, child paths.
- `candidate_outcomes.csv`: per-bank/candidate/training-seed outcomes, timings,
  and standard NFE quality measurements.
- `selector_regret.csv`: score/transport/random/early-validation/oracle outcomes
  and regret; random is included when its selected candidate was trained.
- `predictivity.json`: correlations, completeness, primary/secondary roles, and
  provisional decision status per dataset.

Each child bank has `bank_scores.json` containing path/alpha/floor audits,
`candidate_scores.csv`, `bank_diagnostics.png`, generated YAMLs, its manifest,
`learning_curves.png`, and an evaluation `summary.csv`. Each model run retains its
checkpoint, `config.yaml`, `training.json`, `learning_curve.json`, and normal
checkerboard/horse evaluation JSONs and figures. Horse uses the existing ROI and
background-gap metrics.

## Standalone runner and existing results

`run_wasserstein_bank.py` remains supported and now defaults to checkerboard.
Explicit candidates on one bank can be run with:

```powershell
python run_wasserstein_bank.py --dataset checkerboard --candidates 0 1 6 --stage all --device cuda
python run_wasserstein_bank.py --dataset horse --candidates 0 1 6 --stage all --device cuda
```

The legacy selector-only workflow is available, but cannot measure predictivity
over untrained candidates:

```powershell
python run_wasserstein_bank.py --dataset horse --selectors random transport displacement joint path_joint multiscale variance --stage all --device cuda
```

Use child manifests with the standalone runner and the root study manifest with
`run_wasserstein_study.py`; their formats differ. Standalone resume allows stage
and NFE changes, while the study freezes NFE. Optional mixture2d runs remain in
`wasserstein_experiments/mixture2d.yaml` and are outside the default study.

The cache is now **v2**, including new audits and shared cost-noise draws. V1 cache
paths fail validation and need new banks; automatic paths contain the updated
specification hash. Existing results are not deleted. Do not relabel old scores
as v2 or edit prepared job configurations to change an active study.

The existing full-cloud Jacobian audit accepts saved Wasserstein run configs:

```powershell
python audit_jacobian.py "runs/horse/PRINTED_RUN/config.yaml" --dataset horse --scope both --clouds 16 --probes 8
```

`rollout` starts from fresh Gaussian noise and `coupling` uses the trained bank.
Jacobian size is a diagnostic, not an exact conditional variance or a substitute
for the predeclared generation endpoints.
