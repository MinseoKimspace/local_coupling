# TG coarse 경계 완화 실험

패치 내부는 기존의 **매 방문마다 새로운 uniform random bijection**을 유지한다. Gaussian source, target 점 집합, balanced target patch, 모델, FM 손실, 시간 sampling, 학습 횟수와 추론은 바꾸지 않는다. 변경하는 것은 사전계산한 coarse source-to-patch 배정뿐이다. Fine exact OT, frozen-random, teacher, prior 변경 또는 Jacobian 정규화를 추가하는 실험이 아니다.

비교할 조건은 기존 TG baseline과 다음 두 가지다.

| 조건 | `tg_cache.coarse_mode` | coarse 교환 선택 |
| --- | --- | --- |
| 기존 TG | 생략 또는 `baseline` | 기존 exact coarse 배정 하나 |
| 경계 완화 | `boundary_guided` | 평균 이동 지시의 source-neighbor 급변을 낮추는 교환 |
| 무작위 대조군 | `random_swap_control` | 같은 제약 아래 교환 수와 centroid 비용을 맞춘 무작위 교환 |

## 무엇을 사전계산하는가

각 cloud의 기존 TG source labels를 `a_0`로 놓고, `R=4`개의 hard coarse table을 만든다. 첫 table은 기존 배정을 그대로 유지한다. 나머지 table은 source 이웃 사이의 patch labels를 교환해서 만든다. 각 table은 원래 patch capacity를 유지하며, 한 source는 같은 table에서 최대 한 번만 교환된다. `max_swaps: 16`은 변경 table 각각의 **상한**이지 반드시 실행할 교환 수가 아니다.

학습 중에는 table 하나를 균등하게 선택하고, 선택된 patch 안에서 target을 새로 무작위 일대일 대응시킨다. 온라인 OT나 온라인 경계 최적화는 없다. Bank에서는 같은 cloud를 다시 방문해도 table 선택과 fine permutation을 다시 뽑는다. Stream에서는 각 cloud를 한 번만 사용한다.

최적화하는 점수는 신경망 Jacobian이 아니라, 고정 cloud의 초기 평균 이동 지시에 대한 proxy다. 실제 target patch centroid를 `c_k`라 하면

\[
u_i=\frac1R\sum_r c_{a_r(i)}-x_i,
\qquad
S=\frac1{|E|}\sum_{(i,j)\in E}
\left[\max\left(0,
\frac{\|u_i-u_j\|}{\|x_i-x_j\|+\epsilon}-\tau\right)\right]^2.
\]

`E`는 source의 symmetric kNN graph를 중복 없이 무방향 edge로 만든 것이다. 기본값은 `source_neighbors: 8`, `tau: 2`, `distance_epsilon: 0.001`이다. Guided는 실제 ensemble 점수 `S`가 낮아지는 교환만 받아들인다. Control은 이 점수로 후보를 선택하지 않는다. Control 점수도 우연히 낮아질 수 있다.

두 variant는 동일한 cloud와 초기 coarse 배정에서 함께 계산한다. Guided 교환과 비용이 맞는 control 후보가 존재할 때만 한 쌍을 받아들인다. 따라서 accepted 교환 수는 같고, 각 교환의 centroid 비용 변화와 누적 table 비용은 baseline 비용 대비 `cost_match_tolerance: 0.005` 이내로 맞춘다. 이 허용 오차는 **source-target fine OT 비용**을 맞춘다는 의미가 아니다.

Control은 guided의 accepted 교환 수에 조건부로, 제약과 비용 매칭을 만족하는 후보 중 균등하게 선택한다. 완전히 무제약인 random swap이 아니며 모든 교환 거리·목적지 모호성 통계까지 맞춘 대조군도 아니다. 따라서 이 비교만으로 후반 Jacobian 변화의 인과 메커니즘이 입증되는 것은 아니다.

## 교환 제약과 한계

- 각 table의 centroid 비용 `mean_i ||x_i-c[a_r(i)]||²`는 baseline 대비 `max_cost_increase: 0.02`까지만 허용한다.
- 목적지 patch는 centroid 기준 mutual kNN이어야 한다(`target_neighbors: 2`). 동시에 centroid 간 거리가 두 patch RMS radius 합의 `max_target_distance_factor: 2`배 이하여야 한다.
- 모든 source 각각에 대해 `mean_r ||c[a_r(i)]-c[a_0(i)]||² <= destination_budget² * target_variance`를 적용한다. `target_variance = mean_j ||y_j-target_mean||²`다. 제공 YAML은 체커보드 `destination_budget: 0.5`, 말 `0.35`를 사용한다. 평균 drift만 제한하는 것이 아니라 원래 목적지로부터의 ensemble 제곱 이동량을 제한한다.

두 budget은 생성 결과가 아니라 source/target만 사용하는 작은 geometry pilot으로 정했다. 체커보드 8 clouds에서 budget `0.35`는 6/8 clouds가 no-op이었고, `0.5`는 0/8이었다. `0.5`의 평균 proxy 점수는 baseline `0.87943`, guided `0.55141`, control `0.72939`였으며 비용 매칭 제약은 유지됐다. 말은 `0.35`에서 8 clouds 모두 교환이 발생했다. 이는 실험이 실제로 coupling을 바꾸는지 확인한 결과일 뿐, 생성 품질이나 NSOT 대비 우위를 측정한 결과가 아니다. 각 데이터셋 내 guided/control은 같은 budget을 사용한다.

이 제약은 먼 목적지를 과도하게 섞는 것을 제한하지만, target topology나 얇은 구조 보존을 증명하지는 않는다. 특히 centroid 이웃 관계는 빈 공간 양쪽을 연결할 수 있다. 제약을 만족하는 paired 교환이 없으면 해당 table은 baseline으로 남는다. 코드는 제약을 자동 완화하지 않는다. **실제 accepted 교환 수와 점수 변화부터 확인**해야 하며, 교환이 거의 없으면 품질 차이가 없는 것도 예상 가능한 결과다.

또한 점수 감소가 후반 Jacobian 감소나 품질 개선을 보장하지 않는다. 앵커 목적지를 섞으면서 학습의 모호성이 늘어날 수도 있으므로, CD·밀도·빈 공간 누출·얇은 구조와 Jacobian을 별도로 평가한다.

## Bank 준비 및 학습

기존 실험 장비의 프로젝트 루트에서 실행한다. 네 YAML은 기존 baseline과 같은 `K=8, N=256, seed=0`, bank 4,096 clouds, batch 64, 10,000 updates를 사용한다. 캐시와 checkpoint 이름은 기존 baseline과 분리되어 있다. 기존 캐시를 덮어쓰지 않는다.

```powershell
python prepare_tg.py checkerboard_experiments/target_guided_cached_boundary_guided_k8_n256_seed0.yaml --dataset checkerboard
if ($LASTEXITCODE -ne 0) { throw "체커보드 guided 캐시 준비 실패" }
python train.py checkerboard_experiments/target_guided_cached_boundary_guided_k8_n256_seed0.yaml
if ($LASTEXITCODE -ne 0) { throw "체커보드 guided 학습 실패" }

python prepare_tg.py checkerboard_experiments/target_guided_cached_random_swap_control_k8_n256_seed0.yaml --dataset checkerboard
if ($LASTEXITCODE -ne 0) { throw "체커보드 control 캐시 준비 실패" }
python train.py checkerboard_experiments/target_guided_cached_random_swap_control_k8_n256_seed0.yaml
if ($LASTEXITCODE -ne 0) { throw "체커보드 control 학습 실패" }

python prepare_tg.py horse_experiments/horse_target_guided_cached_boundary_guided_k8_n256_seed0.yaml --dataset horse
if ($LASTEXITCODE -ne 0) { throw "말 guided 캐시 준비 실패" }
python train_horse.py horse_experiments/horse_target_guided_cached_boundary_guided_k8_n256_seed0.yaml
if ($LASTEXITCODE -ne 0) { throw "말 guided 학습 실패" }

python prepare_tg.py horse_experiments/horse_target_guided_cached_random_swap_control_k8_n256_seed0.yaml --dataset horse
if ($LASTEXITCODE -ne 0) { throw "말 control 캐시 준비 실패" }
python train_horse.py horse_experiments/horse_target_guided_cached_random_swap_control_k8_n256_seed0.yaml
if ($LASTEXITCODE -ne 0) { throw "말 control 학습 실패" }
```

기존 baseline을 다시 학습할 때 사용하는 설정은 `checkerboard_experiments/target_guided_cached_k8_n256_seed0.yaml`, `horse_experiments/horse_target_guided_cached_k8_n256_seed0.yaml`이다. 기존 결과를 비교에 사용해도 되지만 bank/stream, seed, 모델·학습 조건을 맞춰야 한다.

## Stream으로 실행하려면

준비 명령에 `--sampling stream`을 붙이면 원본 YAML을 바꾸지 않고 별도의 `_stream` cache/config를 만든다. 예:

```powershell
python prepare_tg.py checkerboard_experiments/target_guided_cached_boundary_guided_k8_n256_seed0.yaml --dataset checkerboard --sampling stream
```

준비가 끝나면 출력되는 `train_command`를 그대로 사용한다. 원래 bank YAML로 학습하면 안 된다. 다른 세 설정도 같은 방식이다. Stream 비교에서는 baseline도 stream이어야 한다.

기본 stream은 640,000 clouds다. 기존 점 배열 외에 `R`개의 coarse tables를 저장하며, `R=4, N=256`이면 variant cache당 약 6.12 GiB다(기존 baseline 약 3.68 GiB). 경계 계산 비용도 cloud 수에 비례해서 늘어난다. Source kNN 구성은 cloud 안에서 `O(N²)` 거리 배열을 사용하므로 이 구현은 현재 `N=256` 실험용이며, 매우 큰 cloud에 그대로 적용하면 안 된다. 처음부터 네 stream을 준비하기 전에 작은 cloud 수로 처리량을 확인하는 것을 권한다. Guided/control 캐시는 각각 두 variant의 paired 알고리즘을 계산하므로 **control 계산까지 포함한 실제 전처리 시간과 저장 공간**을 보고해야 한다. 온라인 OT가 없다고 전체 실행 비용이 무시 가능한 것은 아니다.

## 평가

학습 끝에 출력된 `runs/.../config.yaml`을 사용한다. 실험 YAML이 아니라 **각 checkpoint 옆에 저장된 config**여야 한다. 평가에는 cache가 필요하지 않다.

아래 PowerShell은 네 saved config 경로를 입력받아 먼저 모두 확인하고, 그 다음 NFE 1·2·4·8·16·32·64·128을 평가한다. 경로 앞뒤에 따옴표를 붙여 입력해도 된다.

```powershell
$ErrorActionPreference = "Stop"
$boundaryEvalJobs = @(
    @{ Script = "eval.py"; Dataset = "checkerboard"; Label = "체커보드 guided" },
    @{ Script = "eval.py"; Dataset = "checkerboard"; Label = "체커보드 random control" },
    @{ Script = "eval_horse.py"; Dataset = "horse"; Label = "말 guided" },
    @{ Script = "eval_horse.py"; Dataset = "horse"; Label = "말 random control" }
)

foreach ($boundaryJob in $boundaryEvalJobs) {
    $boundaryConfigInput = (Read-Host "$($boundaryJob.Label)의 saved runs/.../config.yaml 경로").Trim().Trim('"')
    if (-not (Test-Path -LiteralPath $boundaryConfigInput -PathType Leaf)) {
        throw "저장된 config를 찾을 수 없음: $boundaryConfigInput"
    }
    $boundaryJob.ConfigPath = (Resolve-Path -LiteralPath $boundaryConfigInput).Path
}

foreach ($boundaryJob in $boundaryEvalJobs) {
    $boundaryEvalScript = [string]$boundaryJob.Script
    $boundaryRunConfig = [string]$boundaryJob.ConfigPath
    foreach ($boundaryNfe in @(1, 2, 4, 8, 16, 32, 64, 128)) {
        python "$boundaryEvalScript" "$boundaryRunConfig" $boundaryNfe
        if ($LASTEXITCODE -ne 0) {
            throw "평가 실패: $boundaryRunConfig / NFE=$boundaryNfe"
        }
    }
}
```

JSON/PNG는 `eval_results/checkerboard/`, `eval_results/horse/` 아래에 저장된다. Baseline도 같은 NFE·evaluation seed·batch로 평가한다. 말은 동일한 `horse_rois.json`을 유지한다. CD 외에 checkerboard leakage/cell mass/JS, 말 leakage/JS와 foreground·gap ROI를 함께 본다.

후반 Jacobian은 별도 측정한다. 다음 예시의 경로를 각 run의 saved config로 바꾼다. `--scope rollout`은 cache 없이 생성 경로에서 측정한다.

```powershell
python audit_jacobian.py "runs/horse/실제_실행폴더/config.yaml" --dataset horse --scope rollout --clouds 16 --probes 16 --rollout-steps 128 --times 0 0.25 0.5 0.75 1
```

캐시 `metadata.json`에서는 먼저 `boundary_summary.noop_clouds`와 `boundary_summary.metrics.accepted_swaps.mean`을 확인한다. `baseline_score`, `guided_score`, `control_score`, `max_table_cost_difference_fraction`, `guided_max_destination_rms_fraction` 등도 같은 `metrics` 아래에 mean/min/max로 요약된다. 점수는 guided와 control 두 ensemble을 함께 계산한 결과다. 저장된 학습 variant는 `coarse_mode`로 구분한다.

`precompute_seconds`는 paired 계산과 IO를 포함한 전체 전처리 시간이고, `coarse_guidance_seconds`는 그 안의 paired 경계 계산 시간이다. 두 시간을 더하면 중복 계산이다. 학습 `training.json`의 학습 시간, 평가 JSON의 `inference_seconds`, cache 저장 공간을 함께 보관한다. 전체 학습은 이 구현 작업에서 실행하지 않는다. 먼저 seed 0에서 가설을 확인하고, 유망하면 여러 학습 seed로 재검증한다.
