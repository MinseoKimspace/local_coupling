# Cached hard TG: exact patch-internal pairing

This is a coupling-only ablation against the existing cached hard TG configs.
The Gaussian source, sampled target clouds, balanced target patches, exact
source-to-patch allocation, model, optimizer, batch size and 10,000 updates are
unchanged. Only the fine pairing within each patch changes:

- Existing random: draw a fresh target permutation inside each patch when a
  cached cloud is used.
- New exact: precompute a minimum-squared-distance bijection inside each patch
  and reuse the stored pairing during training. No online OT solve is added.

Exact fine pairing is **not global OT**, a new prior, a new loss, distillation,
or a guarantee of sharper generation or better results than NSOT. This tests
whether unrestricted patch-internal randomness is a practical learning
bottleneck. In bank mode, exact fine correspondences are fixed across revisits;
random fine correspondences are resampled. In stream mode each cloud is used
once in both conditions, making it the cleaner first comparison for the current
stream baseline.

The new YAMLs have separate `_fine_exact` cache paths and checkpoint names.
Existing random YAMLs, caches, runs and checkpoints are not replaced. Cache
preparation validates/reuses matching caches; it does not overwrite a different
or incomplete cache. Do not edit a saved training config to bypass a cache hash
or checkpoint-signature mismatch.

Alternatively, the original random YAML can be left untouched and passed to
`prepare_tg.py ... --fine-pairing exact`. This derives a separate `_fine_exact`
cache and prints the training command for its generated `config.yaml`.
Combining `--fine-pairing exact --sampling stream` derives
`_fine_exact_stream`. Use either that CLI route or the new YAMLs below, not both
as separate experiments with identical settings.

## Bank: 4,096 cached clouds, sampled with replacement

Run from the project root in the existing environment, using PowerShell:

```powershell
python prepare_tg.py checkerboard_experiments/target_guided_cached_fine_exact_k8_n256_seed0.yaml --dataset checkerboard
if ($LASTEXITCODE -ne 0) { throw "Checkerboard cache preparation failed" }
python prepare_tg.py horse_experiments/horse_target_guided_cached_fine_exact_k8_n256_seed0.yaml --dataset horse
if ($LASTEXITCODE -ne 0) { throw "Horse cache preparation failed" }
python train.py checkerboard_experiments/target_guided_cached_fine_exact_k8_n256_seed0.yaml
if ($LASTEXITCODE -ne 0) { throw "Checkerboard training failed" }
python train_horse.py horse_experiments/horse_target_guided_cached_fine_exact_k8_n256_seed0.yaml
if ($LASTEXITCODE -ne 0) { throw "Horse training failed" }
```

Random bank baselines remain
`checkerboard_experiments/target_guided_cached_k8_n256_seed0.yaml` and
`horse_experiments/horse_target_guided_cached_k8_n256_seed0.yaml`.
Use the same sampling mode on both sides of the comparison.

## Stream: 640,000 cached clouds, each used once

`--sampling stream` writes a separate effective config in a `_stream` cache
without changing the input YAML. Train from that generated config, not from
the original bank YAML:

```powershell
python prepare_tg.py checkerboard_experiments/target_guided_cached_fine_exact_k8_n256_seed0.yaml --dataset checkerboard --sampling stream
if ($LASTEXITCODE -ne 0) { throw "Checkerboard stream preparation failed" }
python prepare_tg.py horse_experiments/horse_target_guided_cached_fine_exact_k8_n256_seed0.yaml --dataset horse --sampling stream
if ($LASTEXITCODE -ne 0) { throw "Horse stream preparation failed" }
python train.py coupling_cache/checkerboard_tg_hard_k8_n256_seed0_fine_exact_stream/config.yaml
if ($LASTEXITCODE -ne 0) { throw "Checkerboard stream training failed" }
python train_horse.py coupling_cache/horse_tg_hard_k8_n256_seed0_fine_exact_stream/config.yaml
if ($LASTEXITCODE -ne 0) { throw "Horse stream training failed" }
```

For a fresh random stream baseline, run `prepare_tg.py` with the corresponding
original random YAML and `--sampling stream`, then use its printed
`train_command`. Existing validated random caches can be reused.

At N=256, K=8, each balanced patch has 32 points. A full stream requires
5,120,000 small patch OT solves per dataset in addition to the original coarse
preparation. Do not assume that preprocessing is negligible. Measure a small
pilot before committing to the full stream, and report preprocessing, cache
setup and training time separately. A smaller pilot must use a new cache path
and matching training budget; do not change an already prepared cache in place.
The stored int32 fine permutations add 625 MiB for that full stream, bringing
each exact-fine cache to about 4.29 GiB (versus about 3.68 GiB for random fine).

With the same cache seed, preparation batch size, cloud count and environment,
the five base arrays (source, target, both patch labels and capacities) match
the random control byte-for-byte. Fine solving consumes no random numbers.
Check their `array_sha256` values in each `metadata.json` when comparing against
an older cache from another environment. The new exact cache uses format v2;
existing random-fine format-v1 caches and checkpoint fingerprints remain valid.

## Evaluate the saved run config

Training prints `checkpoint=...` and `eval_command=...`, and saves
`runs/<dataset>/<unique-run>/config.yaml`. Use that saved config, **not** the
input experiment YAML or the cache preparation YAML. The evaluator does not
solve OT and still starts from fresh standard Gaussian noise.

The following asks for the two actual saved config paths, checks them before
running, and sweeps the same NFEs for both datasets:

```powershell
$checkerRun = (Read-Host "Checkerboard saved runs/.../config.yaml").Trim().Trim('"')
$horseRun = (Read-Host "Horse saved runs/.../config.yaml").Trim().Trim('"')
foreach ($runConfig in @($checkerRun, $horseRun)) {
    if ([string]::IsNullOrWhiteSpace($runConfig) -or -not (Test-Path -LiteralPath $runConfig -PathType Leaf)) {
        throw "Saved config not found: $runConfig"
    }
}
foreach ($nfe in @(1, 2, 4, 8, 16, 32, 64, 128)) {
    python eval.py "$checkerRun" $nfe
    if ($LASTEXITCODE -ne 0) { throw "Checkerboard evaluation failed at NFE=$nfe" }
    python eval_horse.py "$horseRun" $nfe
    if ($LASTEXITCODE -ne 0) { throw "Horse evaluation failed at NFE=$nfe" }
}
```

Repeat for random and exact runs with the same evaluation settings. Compare
CD, leakage, histogram JS, checkerboard cell mass, and horse thin/gap ROIs, as
well as rendered density. The saved JSONs include training time, inference
time and coupling metadata. Include precomputation in total-cost comparisons;
one seed is a pilot, not evidence of consistent superiority.
