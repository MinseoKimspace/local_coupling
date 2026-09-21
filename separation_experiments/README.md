# TG 구조 분리 시각 실험

Checkerboard와 horse 각각 `baseline`, `t050`, `t075`를 seed 0/1/2로 비교합니다(총 18개 YAML). 모든 설정은 K=8, N=256, batch=64, 10,000 step이며, 데이터별 기존 모델과 학습 설정을 유지합니다. 기존 실험 YAML은 수정하지 않습니다.

| 조건 | coupling | 분리 목표 시각 τ | 목표 margin / target gap | penalty weight |
|---|---|---:|---:|---:|
| baseline | 기존 TG | 해당 없음 | 해당 없음 | 해당 없음 |
| t050 | TG + separation penalty | 0.5 | 0.25 | 10 |
| t075 | TG + separation penalty | 0.75 | 0.25 | 10 |

## 무엇을 바꾸나

Gaussian source 좌표·target 좌표·FPS target partition·균등 capacity·patch 내부 uniform random pairing을 유지하고 **source-to-patch assignment만** 보정합니다. 모델, FM loss, 선형 보간, 학습 시간 샘플링, 추론도 그대로입니다.

방향 `n`은 두 target FPS anchor를 잇는 단위 벡터입니다. **Source를 target anchor에 직접 할당하는 방식이 아닙니다.** Source 기본 비용은 기존처럼 target patch **centroid**까지의 제곱거리입니다.

각 target patch 쌍 `(k,l)`에 대해 아래 projected gap이 `min_target_gap=1e-6`보다 큰 경우만 사용합니다. 모든 unordered pair를 검사합니다.

```text
gY = min(y in P_l) dot(n,y) - max(y in P_k) dot(n,y)
m  = margin_fraction * gY
r  = (m - tau*gY) / (1-tau)
```

기존 TG의 source 할당에서 `b=(max(S_k·n)+min(S_l·n))/2`를 한 번 계산해 고정합니다. Source 비용에 다음 squared hinge를 더하고 동일 capacity의 exact assignment를 풉니다.

```text
patch k penalty = weight * max(dot(n,x) - (b-r/2), 0)^2
patch l penalty = weight * max((b+r/2) - dot(n,x), 0)^2
```

여러 pair의 penalty는 합산합니다. 기준 TG가 이미 모든 제약을 만족하는 cloud는 추가 solve를 생략합니다. 내부 random pairing은 보정 후 새로 뽑습니다. `tau`는 이 비용의 고정 설정이지, 실제 FM 학습 시간 `t`가 아닙니다.

실제 source gap을 `gX`라 하면 모든 해당 patch 내부 endpoint 조합에 대해 투영 간격은 `(1-t)*gX+t*gY` 이상입니다. 따라서 `gX >= r`이면 `t >= tau`에서 margin `m`을 만족합니다. 그러나 **soft penalty이므로 만족 자체는 보장하지 않습니다.** 여러 pair의 조건이 충돌할 수도 있습니다. `training.json`의 `coupling_diagnostics`와 `diagnose.py`의 pair별 기록에서 기존 TG 대비 실제 gap·분리 시각·위반량을 확인해야 합니다. 학습 중 진단은 step 1과 `log_every` 시점에만 기록됩니다.

진단의 `separation_time`은 위 gap 하한이 0에 도달하는 경계 시각입니다(`gX>=0`이면 0). `gX<=0`인 경우 엄밀한 양의 분리는 이 시각 **이후**에 해당합니다. `margin_time`은 gap 하한이 양의 목표 margin `m` 이상으로 유지되기 시작하는 시각이며, 실제 제어 기준은 이것이 `tau` 이하인지입니다. `margin_satisfied`는 float 계산 결과를 tolerance 없이 직접 비교하므로 경계에서 미세한 수치 차이가 생길 수 있습니다. 함께 기록되는 `margin_deficit`의 크기도 확인하세요.

`mean_penalty`는 weight를 곱하기 전의 점당 평균 penalty `R`입니다. 최적화한 평균 목적값은 `mean_base_cost + weight * mean_penalty`입니다. `baseline_violating_pairs`, `resolved_pairs`, `newly_violating_pairs`는 보정으로 기존 위반을 해결했는지, 다른 pair에 새 위반을 만들었는지를 구분합니다. 위반 건수 역시 float 비교 기준입니다.

이 gap은 고정된 유한 point cloud의 조건부 선형 경로에 대한 값입니다. 양의 gap이 실제 해부학적 빈틈이라는 뜻도, 모델의 생성 궤적이나 최종 고주파 품질이 개선된다는 보장도 아닙니다. `tau`를 낮춰도 달성된 분리 시각이나 생성 품질이 반드시 좋아지지는 않습니다.

제공한 `tau=0.5/0.75`, margin 비율 0.25, weight 10은 보정 효과를 비교할 **초기 실험값**이지, 성능으로 보정된 최적값이 아닙니다. 이 설정의 `r`은 각각 `-0.5*gY`, `-2*gY`이므로 source 단계에서 일부 겹침을 허용합니다. 우선 실제 할당 변화율과 달성한 margin을 보고 실험이 유효하게 작동하는지 확인하세요.

## 비용

기존 TG는 cloud당 target/source exact solve 2회입니다. 이 실험은 기준 source solve 후, 위반이 있을 때 보정 solve 1회가 추가되어 최대 3회입니다. Solver 비용 외에 모든 pair의 projection/penalty 비용은 `O(B N K² d)`이며 pair를 순차 처리합니다. Tensor 작업 메모리는 `O(B N K + B N d + B K² d)`이고, 입력 tensor와 exact solver 자체 메모리는 별도입니다.

## 실행 (프로젝트 루트의 PowerShell)

우선 한 조건만 학습하려면:

```powershell
python train.py separation_experiments/checkerboard/tg_t050_k8_n256_seed0.yaml
python train_horse.py separation_experiments/horse/tg_t050_k8_n256_seed0.yaml
```

18개 전체 학습:

```powershell
$separationStartedAt = [DateTime]::UtcNow
foreach ($dataset in @('checkerboard', 'horse')) {
    $script = if ($dataset -eq 'horse') { 'train_horse.py' } else { 'train.py' }
    Get-ChildItem -LiteralPath "separation_experiments/$dataset" -Filter '*.yaml' |
        Sort-Object Name | ForEach-Object {
            python $script $_.FullName
            if ($LASTEXITCODE -ne 0) { throw "Training failed: $($_.FullName)" }
        }
}
```

학습 후 **같은 PowerShell 세션에서** 다음을 실행하면 위 시각 이후 저장된 이 실험의 run만 평가합니다. 입력 YAML을 다시 평가하지 않고 각 run의 frozen `config.yaml`을 사용합니다. 오래된 재학습 run은 포함하지 않습니다.

```powershell
if ($null -eq $separationStartedAt) { throw 'Run the training block in this session first, or use a specific saved run config.' }
$separationSourceRoot = (Resolve-Path -LiteralPath 'separation_experiments').Path + [IO.Path]::DirectorySeparatorChar
$separationRuns = @(Get-ChildItem -LiteralPath 'runs' -Recurse -Filter 'training.json' | ForEach-Object {
    $record = Get-Content -LiteralPath $_.FullName -Raw | ConvertFrom-Json
    if ($record.source_config -and $record.source_config.StartsWith($separationSourceRoot, [StringComparison]::OrdinalIgnoreCase) -and
        [DateTimeOffset]::Parse($record.trained_at).UtcDateTime -ge $separationStartedAt) {
        [PSCustomObject]@{ Dataset = $record.dataset; Config = (Join-Path $_.DirectoryName 'config.yaml') }
    }
})
if ($separationRuns.Count -eq 0) { throw 'No completed separation runs found.' }
foreach ($run in $separationRuns) {
    $script = if ($run.Dataset -eq 'horse') { 'eval_horse.py' } else { 'eval.py' }
    foreach ($nfe in @(1, 2, 4, 8, 16, 32, 64, 128)) {
        python $script $run.Config $nfe
        if ($LASTEXITCODE -ne 0) { throw "Evaluation failed: $($run.Config), NFE=$nfe" }
    }
}
```

학습 출력에 표시된 실제 경로로 개별 진단도 가능합니다:

```powershell
python diagnose.py runs/horse/실제_run_폴더/config.yaml --dataset horse
python diagnose.py runs/checkerboard/실제_run_폴더/config.yaml --dataset checkerboard
```

평가 JSON은 기존처럼 `eval_results/<dataset>/<checkpoint별 폴더>/`에 저장됩니다. `summarize_results.py`는 서로 다른 separation 설정을 별개의 seed group으로 유지하되 같은 비교 그래프에 표시합니다. 동일 설정·seed를 여러 번 학습한 결과가 섞이면 임의로 선택하지 않고 해당 group을 제외하므로, 3-seed 요약에는 이번 실험의 평가 결과만 모은 root를 사용하세요.
