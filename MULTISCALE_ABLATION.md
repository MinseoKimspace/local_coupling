# MS-W2 patch-count and scale-weight ablation

Run the five requested conditions on checkerboard and horse. Each condition uses
M=10,000 bank points, N=256 points per cloud, model seed 0, bank seed 0 and 10,000
training updates by default. The existing model, loss, dataset settings and
evaluator are reused.

| Variant | Patch counts | Relative scale weights | Points per finest patch |
| --- | --- | --- | --- |
| `baseline` | `[4,20,100]` | `[1,1,1]` | 100 |
| `fine_only` | `[4,20,100]` | `[0,0,1]` | 100 |
| `fine_emphasis` | `[4,20,100]` | `[1,1,2]` | 100 |
| `larger_leaf` | `[4,20,40]` | `[1,1,1]` | 250 |
| `smaller_leaf` | `[4,20,200]` | `[1,1,1]` | 50 |

Counts are numbers of patches at successive partition levels. Weights are
normalized to sum to one. `fine_only` retains the same three-level partition as
the baseline and removes only the first two cost contributions. The current
matcher still computes costs for zero-weight levels, so this condition does not
automatically save preprocessing time. Changing the finest count changes the
partition and its block-restricted matching; finer patches are not assumed to
improve generation.

## Commands on the lab computer

Run from the project directory. The default command writes the ten job configs
and their manifest without preparing banks or training:

```powershell
python run_multiscale_ablation.py
```

Keep the printed `experiment_manifest=...` path. Prepare, train and evaluate the
saved jobs with:

```powershell
python run_multiscale_ablation.py --manifest "experiment_manifests/multiscale_ablation_TIMESTAMP_ID/manifest.json" --stage all
```

Replace the placeholder with the actual printed path. To choose CUDA when
creating the jobs and then run them immediately:

```powershell
python run_multiscale_ablation.py --stage all --device cuda
```

Preparation and later stages can also run separately:

```powershell
python run_multiscale_ablation.py --stage plan --device cuda
python run_multiscale_ablation.py --manifest "PATH_TO_MANIFEST" --stage prepare
python run_multiscale_ablation.py --manifest "PATH_TO_MANIFEST" --stage train
python run_multiscale_ablation.py --manifest "PATH_TO_MANIFEST" --stage eval
```

A resumed manifest keeps its saved settings and verifies the recorded files.
Completed jobs and evaluations are reused; an interrupted training job is
retrained because there is no optimizer-state resume. Add evaluations using
`--manifest "PATH_TO_MANIFEST" --stage eval --nfe 256`.

For a new experiment, `--datasets horse` selects one dataset, `--seeds` controls
model seeds, `--bank-seed` controls the endpoint bank, and `--steps` overrides
training updates. These options must be set when creating the manifest.

`--smoke --stage all` is a tiny CPU pipeline check with M=80, N=8 and two training
updates. Its settings and outputs are for verification, not generation-quality
comparisons.

## What is held fixed and saved

For each dataset, all five variants must have byte-identical original source
and target arrays. The runner checks their hashes before training. Different
model seeds share the same bank for a given variant. Inference uses the existing
fresh-Gaussian evaluation procedure and matched evaluation settings.

Default evaluation NFEs are 1, 2, 4, 8, 16, 32, 64 and 128. The existing evaluator
produces its usual PNGs and JSONs under `eval_results/<dataset>/`; image titles
identify the patch counts and weights. Checkpoints and training reports are
under `runs/<dataset>/`, with variant-specific names. Configs and the manifest
are under `experiment_manifests/multiscale_ablation_TIMESTAMP_ID/`.

The aggregate report is saved under
`eval_results/multiscale_ablation/multiscale_ablation_TIMESTAMP_ID/`:

- `summary.json`: provenance, endpoint equality checks, timing and complete
  evaluation results.
- `comparison.csv`: metrics for each dataset, variant, model seed and NFE, plus
  variant-minus-baseline deltas for comparable evaluations. Missing metrics or
  mismatched evaluation settings suppress the corresponding deltas.

Preprocessing time counts each unique bank once across model seeds. Training
time and per-NFE inference time are reported separately; existing inference
timing excludes rendering and quality-metric computation.

Compare NFE1 quality alongside NFE128 CD, leakage, density metrics and horse ROI
mass errors. A smaller matching objective alone does not establish better
generation, and one seed is a pilot comparison.
