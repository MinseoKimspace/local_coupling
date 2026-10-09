# Fixed-anchor Gaussian-mixture prior experiments in 2D

The original standard-Gaussian Hard Bank TG and paper-based NSOT configurations remain unchanged. The experimental configurations below compare TG and an NSOT extension using a fixed anchor Gaussian-mixture (GMM) prior, with K=8, N=256, seed=0, sigma=0.10 and 10,000 training updates. The existing per-dataset backbone, optimizer and training batch size are preserved. These are fixed-shape 2D experiments, not a quality guarantee or a multi-shape 3D anchor generator.

## Shared fixed source prior

`anchor_flow.mode: anchor_prior` changes the source distribution. Once, a separate 4096-point reference sample from the original **training** distribution is partitioned using FPS and balanced exact OT. Actual patch means become K fixed mixture centers. There is no iterative refinement, whitening or evaluation-target fitting.

Each source point draws a component independently:

```text
k_i ~ Uniform({0,...,K-1}), epsilon_i ~ Normal(0,I_2)
X_i = fixed_center[k_i] + sigma * epsilon_i
sigma = 0.10
```

The declared source law is the product over N points of this equally weighted GMM. Component counts are random, not forced to N/K. `sigma` is measured in original 2D coordinates, not a normalized relative radius.

Inference samples the **same saved fixed mixture** and integrates the trained network. The resolved centers are stored in the saved run configuration and checkpoint; generation needs neither the coupling cache nor an evaluation target. Evaluate only `runs/.../config.yaml`, not an unresolved training-template YAML.

The prior already contains shape information learned from the training distribution. These experiments therefore are **not standard-Gaussian-prior coupling-only comparisons**. `source_*` evaluation scores and the audit's `initial_source_quality` measure the no-flow prior against the same sampled evaluation targets, so the model's added value can be checked separately.

## TG with the fixed GMM prior

TG retains its 4096-pair cloud bank, exact balanced target partition, exact source-to-patch-centroid assignment and fresh uniform within-patch random bijection on every visit. Prior component labels do not force assignment to a particular target patch. The conditional path remains linear, with velocity `Y_paired - X`.

The saved cloud bank is a finite empirical approximation to the declared GMM source law. Its target clouds, coarse assignments and source coordinates are reused; fine point pairings and training times are sampled afresh.

## NSOT extension with component-centered hybrid noise

This is an extension of this project's **paper-based 2D exact-superset reproduction**, not author code and not the paper's main 100K 3D setting. The new cache contains M=10,000 GMM source points, their component labels, fixed centers and original target points, coupled by the existing exact squared-Euclidean assignment solver. Training draws paired point indices with replacement to compose each cloud, as in the Gaussian-prior baseline.

The original standard-Gaussian NSOT hybrid is:

```text
X_hybrid = sqrt(1-beta) * Z + sqrt(beta) * epsilon
Z, epsilon ~ Normal(0,I_2)
```

Applying that zero-centered formula to GMM points would shrink their component centers and would not preserve the requested mixture. The extension instead uses the cached component label k:

```text
Z = c_k + sigma * eta
X_hybrid = c_k + sqrt(1-beta) * (Z-c_k) + sigma * sqrt(beta) * epsilon
beta = 0.20
eta, epsilon ~ Normal(0,I_2), independently
```

For a population GMM draw, conditional on k, the new residual has covariance `sigma^2 I_2` and mean zero; thus this operation preserves the GMM source law. It does not preserve pointwise OT correspondence: it intentionally perturbs the cached source around its own component. The conditional path to its paired target is linear and the FM label uses the actual hybrid source, `Y_paired - X_hybrid`.

Two distinctions matter:

- With a finite fixed cache, the hybrid source is an empirical-weight Gaussian mixture around cached residuals, only an approximation to the declared GMM. Component frequencies also fluctuate in a finite cache; do not claim exact finite-bank marginal equality.
- At beta=1 the within-component residual becomes fresh noise, but the component label remains associated with its cached paired target. Unlike the original zero-centered Gaussian hybrid, beta=1 does **not** imply globally independent source-target coupling.

The GMM source centers for TG and NSOT are fitted with the same per-dataset reference size, seed and rule. Check the saved centers rather than assuming equality after changing settings. Each method uses separate cache paths; neither overwrites the Gaussian-prior baseline cache.

## Prepare and train

Run from the project directory in the existing Python environment. Preparation verifies an existing cache rather than overwriting it. Change the cache path and checkpoint name when changing K, sigma, reference sample settings, superset size or cache seed.

TG anchor prior, if not already trained:

```powershell
python prepare_tg.py checkerboard_experiments/target_guided_cached_anchor_prior_k8_n256_seed0.yaml --dataset checkerboard
python train.py checkerboard_experiments/target_guided_cached_anchor_prior_k8_n256_seed0.yaml

python prepare_tg.py horse_experiments/horse_target_guided_cached_anchor_prior_k8_n256_seed0.yaml --dataset horse
python train_horse.py horse_experiments/horse_target_guided_cached_anchor_prior_k8_n256_seed0.yaml
```

NSOT anchor-prior extension:

```powershell
python prepare_nsot.py checkerboard_experiments/nsot_anchor_prior_k8_n256_seed0.yaml --dataset checkerboard
python train.py checkerboard_experiments/nsot_anchor_prior_k8_n256_seed0.yaml

python prepare_nsot.py horse_experiments/horse_nsot_anchor_prior_k8_n256_seed0.yaml --dataset horse
python train_horse.py horse_experiments/horse_nsot_anchor_prior_k8_n256_seed0.yaml
```

## Evaluate the newest completed run for each method and dataset

The following PowerShell block chooses only exact expected run-name prefixes followed by a timestamp and run ID. It requires the saved YAML, checkpoint and training metadata, prints each resolved path, and uses argument arrays so an absent path cannot silently become the NFE argument. To evaluate an older run, explicitly substitute its saved `runs/.../config.yaml` instead.

```powershell
$ErrorActionPreference = 'Stop'
$anchorJobs = @(
    [pscustomobject]@{ Dataset='checkerboard'; Prefix='checkerboard_target_guided_cached_k8_n256_seed0_anchor_prior'; Script='eval.py'; Checkpoint='target_guided_cached_anchor_prior_k8_n256_seed0.pt' },
    [pscustomobject]@{ Dataset='checkerboard'; Prefix='checkerboard_nsot_k8_n256_seed0_anchor_prior'; Script='eval.py'; Checkpoint='nsot_anchor_prior_k8_n256_seed0.pt' },
    [pscustomobject]@{ Dataset='horse'; Prefix='horse_target_guided_cached_k8_n256_seed0_anchor_prior'; Script='eval_horse.py'; Checkpoint='horse_target_guided_cached_anchor_prior_k8_n256_seed0.pt' },
    [pscustomobject]@{ Dataset='horse'; Prefix='horse_nsot_k8_n256_seed0_anchor_prior'; Script='eval_horse.py'; Checkpoint='horse_nsot_anchor_prior_k8_n256_seed0.pt' }
)
$anchorConfigs = @()
foreach ($anchorJob in $anchorJobs) {
    $anchorRunPattern = '^' + [regex]::Escape($anchorJob.Prefix) + '_[0-9]{8}T[0-9]{12}Z_[0-9a-f]{8}$'
    $anchorRun = Get-ChildItem -LiteralPath ('runs/' + $anchorJob.Dataset) -Directory |
        Where-Object {
            $_.Name -match $anchorRunPattern -and
            (Test-Path -LiteralPath (Join-Path $_.FullName 'config.yaml') -PathType Leaf) -and
            (Test-Path -LiteralPath (Join-Path $_.FullName 'training.json') -PathType Leaf) -and
            (Test-Path -LiteralPath (Join-Path $_.FullName $anchorJob.Checkpoint) -PathType Leaf)
        } |
        Sort-Object Name -Descending | Select-Object -First 1
    if ($null -eq $anchorRun) { throw ('Completed run not found: ' + $anchorJob.Prefix) }
    $anchorCfgPath = Join-Path $anchorRun.FullName 'config.yaml'
    if ([string]::IsNullOrWhiteSpace($anchorCfgPath) -or -not (Test-Path -LiteralPath $anchorCfgPath -PathType Leaf)) {
        throw ('Missing saved config: ' + $anchorCfgPath)
    }
    Write-Host ('Evaluating: ' + $anchorCfgPath)
    $anchorConfigs += [pscustomobject]@{ Dataset=$anchorJob.Dataset; Config=$anchorCfgPath }
    foreach ($anchorNfe in @(1,2,4,8,16,32,64,128)) {
        $anchorEvalArgs = @($anchorJob.Script, $anchorCfgPath, [string]$anchorNfe)
        & python @anchorEvalArgs
        if ($LASTEXITCODE -ne 0) { throw ('Evaluation failed: ' + $anchorCfgPath + ' / NFE=' + $anchorNfe) }
    }
}
```

Both ordinary evaluators also record `source_*` metrics before any model call. Source sampling consumes random numbers differently across priors, so equal ordinary evaluation seeds alone do not guarantee identical target draws between changed-prior and standard-Gaussian runs. Use the independent diagnostic streams below for strict common-target comparisons.

## Audit no-flow prior quality, generation and finite-reference stability

Run this block in the same PowerShell session after the preceding selection block. It uses fresh evaluation clouds, not cached training targets, and skips optional coupling residuals so the audit does not require the training cache.

```powershell
if ($null -eq $anchorConfigs -or $anchorConfigs.Count -ne 4) {
    throw 'Run the four-run selection/evaluation block first.'
}
foreach ($anchorEntry in $anchorConfigs) {
    $anchorAuditArgs = @('audit_generation.py', $anchorEntry.Config, '--dataset', $anchorEntry.Dataset,
        '--clouds', '32', '--skip-fm', '--seed', '2026',
        '--nfes', '1', '2', '4', '8', '16', '32', '64', '128',
        '--reference-nfe', '128', '--max-reference-nfe', '512')
    & python @anchorAuditArgs
    if ($LASTEXITCODE -ne 0) { throw ('Generation audit failed: ' + $anchorEntry.Config) }
}
```

The audit records `initial_source_quality`, source/target hashes, NFE-dependent quality and 128/256/512 finite-reference checks. Target hashes should match across methods for the same dataset, diagnostic seed, device and dtype. All NFE levels within one run share the same source cloud bank. Compare saved centers and source hashes as well as target hashes when checking equal-prior comparisons.

Both methods use a linear conditional path. To include optional raw FM regression residuals, keep the matching training cache available and replace `--skip-fm` with `--batches 2`. These residuals are not a direct estimate of mean-field approximation error.

Report CD, leakage, density/ROI scores and the no-flow prior together. Low leakage can reflect missing thin structures rather than successful generation; low CD alone does not establish correct occupancy. The endpoint MSE is against the finest finite Euler rollout of the same model, not an exact target correspondence or a direct measure of mean-field approximation error.

## Interpretation and cost limits

- Both variants use a linear conditional path and the same saved-GMM inference prior. They differ in cloud-bank coarse/fine coupling versus exact-superset point coupling with component-centered hybrid noise. This is a same-prior method comparison, not a claim to reproduce original Gaussian-prior NSOT unchanged.
- Fixed-shape anchor fitting uses only training data. Saved centers can be reused for ordinary generation without cache access; extending this to multiple unknown shapes needs a separate anchor-generating mechanism.
- `audit_mean_field.py` does not supply a population conditional-mean-field oracle for the finite cached TG or NSOT training laws. Raw FM regression residuals include conditional variance and are not pure approximation error.
- Standard evaluation `inference_seconds` measures integration only: source sampling, warmup, model loading, metrics and rendering are excluded. Do not call it end-to-end latency. Diagnostic rollout timing has a different scope, including batching/transfers and path bookkeeping.
- Report fixed-center fitting, cache preparation, cache setup and online training separately. Do not assume that prior fitting is included in another timing field without checking the stored scope.
- These experiments do not guarantee improved quality over Hard Bank, Gaussian-prior NSOT, minibatch OT or permutation-only EFM. NFE=128 alone is not a proof of convergence; inspect the successive finite-reference checks.
