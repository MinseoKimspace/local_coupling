# TG K comparison: checkerboard / horse, seed 0

Only K and the checkpoint filename change in the four new YAMLs. Each dataset
keeps its existing K=8 model, data, optimizer and 10,000-step training budget.
Existing K=8/16 files and results are not changed. Compare the saved configs and
code provenance of older runs before pooling their generation results.

## Train only the missing K=4/32 conditions

From the project root, with the existing environment activated:

```powershell
python train.py checkerboard_experiments/target_guided_k4_n256_seed0.yaml
python train.py checkerboard_experiments/target_guided_k32_n256_seed0.yaml
python train_horse.py horse_experiments/horse_target_guided_k4_n256_seed0.yaml
python train_horse.py horse_experiments/horse_target_guided_k32_n256_seed0.yaml
```

Each run saves its own config and checkpoint under `runs/<dataset>/`.
Evaluate the **saved run config**, using the existing `eval.py` or
`eval_horse.py` and the same NFE sweep used for K=8/16.

## Training-free partition scores

```powershell
python audit_k.py checkerboard_experiments/target_guided.yaml --dataset checkerboard --ks 4 8 16 32 --clouds 32 --seed 0
python audit_k.py horse_experiments/horse_target_guided_k8_n256_seed0.yaml --dataset horse --ks 4 8 16 32 --clouds 32 --seed 0
```

No weights are needed; existing K=8/16 weights are not loaded or overwritten.
The same fresh target cloud is reused for every K. The audit calls the actual
`balanced_target_partition` function used by TG training: deterministic FPS,
exact capacity-constrained assignment to anchors, then patch means. There is no
separate k-means optimization or change to training coupling.

The audit uses the config device/dtype. Optional `--device cpu` avoids GPU use;
sampling and floating-point assignment can differ between CPU and CUDA, so use
one device consistently when comparing scores. `--seed` is the diagnostic
sampling seed, distinct from the training seed.

Output: a unique `analysis_results/<dataset>/k_audit_.../` directory containing:

- `k_audit.json`: per-cloud scores/capacities, target and label hashes, config,
  environment, code hashes, mean/sample-SD summaries, and CH/DB-selected K.
- `scores.png`: CH, DB and normalized WCSS against K.

## Definitions and limits

- CH: `(between_scatter / WCSS) * (N-K)/(K-1)`; higher is better.
- DB: each patch's worst `(mean_radius_k + mean_radius_l)/centroid_distance`,
  averaged across patches; lower is better. Radius is mean Euclidean distance,
  not squared distance or RMS.
- Normalized WCSS: `WCSS / total_scatter`, where total scatter is
  `sum ||y_j - mean(Y)||^2`. This is dimensionless and **not WCSS/N**.
  Raw WCSS and WCSS/N are saved separately.

Scores use float64 arithmetic on the original sampled coordinates and TG labels.
CH/DB selection compares mean scores, requires validity on every cloud, and
breaks exact ties by smaller K. Undefined/infinite scores are JSON null: CH when
WCSS=0; DB when centroids coincide; normalized WCSS when total scatter=0.
The candidate sweep requires `2 <= K < N`. Degeneracy handling is explicit,
not scikit-learn's finite sentinel convention.

Error bars describe variation across target clouds, **not training seeds**.
Selected K is a partition-score candidate, not a generation-quality optimum.
No YAML is automatically edited and no model/loss/inference setting is changed.
No new dependencies are required.
