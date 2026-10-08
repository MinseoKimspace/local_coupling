# Anchor flow experiments for two dimensional generation

The original Hard Bank TG configurations remain unchanged. These four configurations use K=8, N=256, seed=0, bank=4096, the existing per-dataset backbone/optimizer and 10,000 training updates. Both retain exact balanced coarse assignment and fresh uniform within-patch random pairing. They are experiments, not a claim of improved quality or a 3D method.

## Anchor Gaussian prior

`anchor_flow.mode: anchor_prior` changes the source distribution. Once, a separate 4096-point reference sample from the original **training** distribution is split with the existing FPS + balanced exact-OT rule. Actual patch centroids become K fixed mixture centers. There is no iterative refinement, whitening, or evaluation-target fitting.

Each source point independently draws a uniform component and Gaussian noise:

```text
k_i ~ Uniform({0,...,K-1}), epsilon_i ~ N(0,I_2)
X_i = fixed_center[k_i] + sigma * epsilon_i
sigma = 0.10
```

The joint source law is the product of this mixture over N points. Component counts are random, **not** forced to N/K. The existing coarse OT then assigns these source points to each realized target cloud's patches with the usual fixed capacities. A prior component label is not a hard target-patch assignment.

Training uses the original straight path and velocity `Y_paired - X`. Inference draws from the **same fixed mixture**. Exact centers are embedded in the saved run configuration and checkpoint, so ordinary generation does not load the training cache or inspect evaluation targets. A training-template YAML has unresolved centers and must not be used for evaluation.

This prior already contains fixed-shape training information, so this is **not a Gaussian-prior coupling-only comparison**. Evaluation records `source_*` metrics before any model step against the same evaluation target. Compare these no-flow prior scores with generated scores to identify the model's added value.

## Standard Gaussian source with an anchor waypoint

`anchor_flow.mode: anchor_waypoint` preserves the original standard-Gaussian source. For each sampled bank pair, source-indexed target-patch centroids are recovered from the stored coarse assignment. Fresh waypoint noise is sampled once per visit:

```text
W_i = realized_patch_center[source_assignment[i]] + sigma * epsilon_i
D = W - (X + Y_paired)/2
x(t) = (1-t)*X + t*Y_paired + 16*t^2*(1-t)^2*D
u(t) = Y_paired - X + 32*t*(1-t)*(1-2*t)*D
```

Thus `x(0)=X`, `x(0.5)=W`, `x(1)=Y_paired`, and the FM target is the **exact derivative** `u(t)`. The bump derivative is zero at both endpoints, preserving `u(0)=u(1)=Y_paired-X` rather than introducing arbitrary endpoint acceleration. The original target endpoint is not deformed. This smooth quartic path can still bend or overshoot; it is not guaranteed to improve low-NFE quality. Inference starts from standard Gaussian and integrates the trained model only, without target patches or an explicit inference waypoint.

Its separate cache uses the same baseline X/Y/coarse-label generation rule, with the same seed; waypoint randomness is fresh during training, not stored as a fixed fine matching. This is a **path change**, not a coupling-only comparison. Baseline cache directories are not overwritten.

## Prepare and train

Run from the project directory in the existing Python environment. Preparation and training use the same YAML; existing caches are verified and never overwritten.

```powershell
python prepare_tg.py checkerboard_experiments/target_guided_cached_anchor_prior_k8_n256_seed0.yaml --dataset checkerboard
python prepare_tg.py checkerboard_experiments/target_guided_cached_anchor_waypoint_k8_n256_seed0.yaml --dataset checkerboard
python prepare_tg.py horse_experiments/horse_target_guided_cached_anchor_prior_k8_n256_seed0.yaml --dataset horse
python prepare_tg.py horse_experiments/horse_target_guided_cached_anchor_waypoint_k8_n256_seed0.yaml --dataset horse

python train.py checkerboard_experiments/target_guided_cached_anchor_prior_k8_n256_seed0.yaml
python train.py checkerboard_experiments/target_guided_cached_anchor_waypoint_k8_n256_seed0.yaml
python train_horse.py horse_experiments/horse_target_guided_cached_anchor_prior_k8_n256_seed0.yaml
python train_horse.py horse_experiments/horse_target_guided_cached_anchor_waypoint_k8_n256_seed0.yaml
```

`anchor_flow.sigma` is in the original 2D coordinate units, not a normalized relative radius. Prior `reference_points` and `seed` control only the fixed-center fit. Use a new `tg_cache.path` and checkpoint name when changing these settings.

## Evaluate the four completed runs at all NFEs

This PowerShell block explicitly selects the **newest completed run per variant** and prints its exact path before evaluation. If you want a particular older run, set its saved `runs/.../config.yaml` directly instead; do not use training-template YAMLs. Argument arrays and explicit checks prevent a missing path from being replaced by the NFE argument.

```powershell
$ErrorActionPreference = 'Stop'
$anchorJobs = @(
    [pscustomobject]@{ Dataset='checkerboard'; Prefix='checkerboard_target_guided_cached_k8_n256_seed0_anchor_prior'; Script='eval.py' },
    [pscustomobject]@{ Dataset='checkerboard'; Prefix='checkerboard_target_guided_cached_k8_n256_seed0_anchor_waypoint'; Script='eval.py' },
    [pscustomobject]@{ Dataset='horse'; Prefix='horse_target_guided_cached_k8_n256_seed0_anchor_prior'; Script='eval_horse.py' },
    [pscustomobject]@{ Dataset='horse'; Prefix='horse_target_guided_cached_k8_n256_seed0_anchor_waypoint'; Script='eval_horse.py' }
)
$anchorConfigs = @()
foreach ($anchorJob in $anchorJobs) {
    $anchorPattern = $anchorJob.Prefix + '_*'
    $anchorRun = Get-ChildItem -LiteralPath ('runs/' + $anchorJob.Dataset) -Directory |
        Where-Object { $_.Name -like $anchorPattern -and (Test-Path -LiteralPath (Join-Path $_.FullName 'config.yaml') -PathType Leaf) } |
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

For strict common-target comparisons use the isolated source/target diagnostic streams below. The ordinary evaluators use a shared global RNG: drawing mixture-component labels consumes extra random numbers, so equal evaluation seeds alone do not imply identical sampled target clouds between changed-prior and Gaussian-prior runs.

```powershell
foreach ($anchorEntry in $anchorConfigs) {
    $anchorAuditArgs = @('audit_generation.py', $anchorEntry.Config, '--dataset', $anchorEntry.Dataset,
        '--clouds', '32', '--skip-fm', '--seed', '2026')
    & python @anchorAuditArgs
    if ($LASTEXITCODE -ne 0) { throw ('Generation audit failed: ' + $anchorEntry.Config) }
}
```

The generation audit records `initial_source_quality`, fixed target hashes, source hashes, NFE quality and finite-reference refinement checks. All NFE levels within a run share the same source cloud bank. Target hashes should match across methods at the same diagnostic seed/device/dtype, while source hashes deliberately differ for the changed prior. Report leakage, histogram/ROI scores and no-flow quality alongside CD; low CD alone does not establish good shape occupancy.

## Diagnostic and timing limits

- `eval.py`, `eval_horse.py` and `audit_generation.py --skip-fm` support both variants. NFE remains the number of Euler model calls.
- `diagnose.py` and the optional FM-residual branch in `audit_generation.py` assume a straight conditional path. They explicitly reject waypoint mode; use `--skip-fm` for its generation/integration audit. The prior variant retains a straight path and can use those residuals, provided its training cache is available.
- `audit_mean_field.py` does not claim an analytic population conditional field for a finite TG bank; its cached-coupling exclusion remains. The waypoint's velocity does equal `Y_paired-X` at t=0, but this does not make the existing fresh-cloud estimator valid for the finite cached law. At intermediate times the waypoint derivative differs from `Y_paired-X`.
- Standard evaluation `inference_seconds` is integration-only and excludes source sampling, warmup, loading, metrics and rendering. A prior sampler is therefore outside this timer; do not call it end-to-end latency. The generation audit's rollout timer also includes batching/transfers and diagnostic path bookkeeping, so its values are a different timing scope.
- Report fixed-center fitting/cache preparation, cache setup and online training separately. Both methods use target information only during training; neither proves improvement over Hard Bank, NSOT, minibatch OT or permutation-only EFM.
- Keep the current experiment separate from the baseline when comparing final quality, low-NFE quality and training cost. No claim of numerical convergence should be made solely from NFE=128.
