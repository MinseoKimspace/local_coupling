# Balanced-anchor prior vs GMM-EM prior, with the same NSOT

This is a source-prior ablation for the fixed checkerboard and horse 2D training
distributions. It is not a comparison with the original Gaussian-prior NSOT,
not a coupling-only gain, and not evidence about unseen-shape generation.

## What changes

Both priors are `K=8`, equal-weight mixtures with shared fixed isotropic standard
deviation `sigma=0.1`. Only the training-derived component centers change:

- `balanced_anchor`: the existing one-pass balanced FPS/exact-OT partition and
  its patch means. Existing configs and their default behavior are unchanged.
- `gmm_em`: maximum-likelihood EM center fitting for an equal-weight,
  fixed-isotropic-covariance mixture. Five initializations, at most 100 iterations
  each, tolerance `1e-6`; select the best training-reference likelihood.

The second variant is deliberately a **constrained GMM**, not a full-covariance
GMM with learned mixture weights or learned variances. Those parameters remain
fixed to separate center-fitting benefits from prior capacity and noise scale.
Inference samples the same saved prior used by training; no evaluation target is
used to fit centers. Gaussian NSOT source priors remain available separately.

## Matched settings

The four experiments use the original dataset-specific isotropic NSOT
configs: same training/reference/cache seeds, 4096 training-reference points,
10000-point exact-OT supersets, component-centered hybrid `beta=0.2`, pointwise iid
uniform component sampling, 256 points per cloud, batch size 64, 10000 updates,
unconditioned backbone, optimizer and learning rate. Both fit procedures receive
the same reference sample and use scalar, isotropic component-centered noise.
Fresh GMM pair caches are required; source labels/noise and target draws follow
the same seeds and sampling code, while the fitted centers and OT permutation
are allowed to differ. A finite pair bank still approximates the population GMM.

## Execute on the existing experiment machine

Use its existing environment and run from the repository root. No full training
needs to run on the development laptop. After transferring the changed code and
new configs, the complete four-run workflow is:

```powershell
python run_prior_comparison.py
```

It prepares or validates all four caches, trains all four models, then evaluates
each at NFE `1,2,4,8,16,32,64,128`. It captures the **returned** training run paths
and saves them in a unique `comparison_results/prior_comparison_.../manifest.json`.
No run prefix is guessed and no positional YAML path can become an empty string.
Existing caches are validated and reused, never silently overwritten. If a cache
fails validation, investigate the exact reported mismatch; do not remove unrelated
caches or use a Gaussian cache for the new GMM prior.

Preparation and training can also be separated:

```powershell
python run_prior_comparison.py --stage prepare
python run_prior_comparison.py --stage train
```

`--stage train` prepares or validates the caches and runs the four trainings without
evaluation. To evaluate its saved runs later, pass the exact manifest path printed
by that training invocation:

```powershell
python run_prior_comparison.py --stage eval --manifest "comparison_results/prior_comparison_REPLACE_WITH_PRINTED_DIRECTORY/manifest.json"
```

The manifest placeholder above must be replaced with the printed path. Evaluation
uses saved `runs/.../config.yaml` files with resolved centers and matching
checkpoints, not the unresolved experiment templates. Reevaluation creates a new
manifest and does not modify the original. After an interrupted workflow, it can
evaluate the completed training jobs already captured in that manifest.

For a small development smoke test only, `--steps 1 --nfes 1` lowers the update and
evaluation counts, **but still prepares the configured 10000-point banks**. Do not
use those checkpoints as the full experiment's results. There is no built-in
full-run resume; rerunning `all` or `train` creates fresh training runs.

### Individual preparation and training commands

If executing one experiment at a time is preferable:

```powershell
python prepare_nsot.py checkerboard_experiments/nsot_anchor_prior_k8_n256_seed0.yaml --dataset checkerboard
python prepare_nsot.py checkerboard_experiments/nsot_gmm_prior_k8_n256_seed0.yaml --dataset checkerboard
python prepare_nsot.py horse_experiments/horse_nsot_anchor_prior_k8_n256_seed0.yaml --dataset horse
python prepare_nsot.py horse_experiments/horse_nsot_gmm_prior_k8_n256_seed0.yaml --dataset horse

python train.py checkerboard_experiments/nsot_anchor_prior_k8_n256_seed0.yaml
python train.py checkerboard_experiments/nsot_gmm_prior_k8_n256_seed0.yaml
python train_horse.py horse_experiments/horse_nsot_anchor_prior_k8_n256_seed0.yaml
python train_horse.py horse_experiments/horse_nsot_gmm_prior_k8_n256_seed0.yaml
```

Each trainer prints the actual saved `runs/.../config.yaml` in `eval_command`.
Pass that path followed by the NFE to `eval.py` or `eval_horse.py`. The runner is
recommended because it records these paths automatically.

## What to report

Compare generated CD, histogram JS and leakage; checkerboard cell mass error;
and horse foreground thin-structure mass/ROI fidelity plus background-gap leak.
The `source_*` metrics in every evaluation JSON are a no-flow prior baseline
computed against the **same sampled target** as the generated metrics. Report
both source and generated scores, including whether thin-structure mass improves
or worsens rather than reporting only the best CD.

Separate timing scopes:

- Prior fitting plus NSOT bank generation: original cache
  `precompute_seconds`; cache metadata states its scope. The manifest separately
  records this invocation's prepare/validation wall time and whether a cache was
  reused. A reused cache's validation time is not a new fitting measurement.
- Training: `training.json` / evaluation JSON `training_seconds` measures the
  online update loop, excluding prior fitting, bank generation and loading.
- Inference: existing evaluation JSON `inference_seconds` is **integration only**,
  after warmup; it excludes source sampling, loading, metrics and rendering. Do not
  label it as end-to-end sampling latency. Both priors use the same source sampler,
  but an end-to-end timing claim would need a separate matched measurement.

Default training seed 0 and evaluation seed 1 provide an initial matched check,
not statistical significance. Multiple training seeds are needed before a robust
superiority claim. If constrained GMM-EM matches balanced anchors, that weakens an
anchor-center-specific claim; if balanced anchors preserve density/thin regions
better, verify that effect across seeds before attributing it to the center rule.
