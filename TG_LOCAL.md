# TG 국소 대응 실험

기존 TG의 source와 target을 그대로 사용하면서 대응 선택만 바꾸는 실험이다. 비교에는 원본 TG baseline, `path_affine`, `path_affine_subpatch`의 세 조건을 반드시 포함한다.

| 조건 | 대응 평가 | 매번 새로 뽑는 fine random의 범위 |
| --- | --- | --- |
| 기존 TG baseline | 기존 coarse OT 배정 | 원래 target patch |
| `path_affine` | 실제 fine 대응의 여러 시간대 국소 affine 예측 잔차 | 원래 target patch |
| `path_affine_subpatch` | 위와 동일 | target patch를 두 개로 나눈 subpatch |

두 variant는 coarse 후보를 사전 계산할 때 실제 random fine permutation을 샘플링한다. 대응마다 `u = y[permutation] - x`, `z(t) = (1-t)x + t*y[permutation]`를 구하고 `t = 0, 0.5, 0.75, 0.9`에서 가까운 점들의 속도를 평가한다. 평가할 점 `i`를 제외한 이웃들로 `u_j ≈ b_i + A_i(z_j-z_i)`를 맞추고, 남겨둔 중심 점의 `u_i`를 예측한 잔차를 계산한다. 각 점을 번갈아 남기는 leave-center-out 평가다. 국소 affine 식은 공통 이동뿐 아니라 수축·팽창·회전·전단도 설명할 수 있다. 대응별 잔차를 계산한 다음 평균하므로 반대 방향 속도를 먼저 평균내서 점수가 사라지게 하지 않는다.

먼저 target patch 관계와 target 점 사이 거리 gate를 만족하는 고정된 target 연결 후보 집합을 만든다. 각 interpolant 시간에서는 **그 후보 집합 안에서** 가까운 이웃을 고른다. 전체 점에서 interpolant kNN을 먼저 고른 다음 gate로 거르는 방식이 아니다. 국소 반경으로 좌표를 정규화한 ridge fit을 사용한다. 이웃이 3개보다 적으면 그 점의 잔차는 점수에서 제외한다. 이 gate에서는 permutation을 바꾸어도 이웃 후보 수가 보존되므로 평가 대상 비율이 유지된다. 코드도 기준 대비 평가 비율이 줄어드는 후보를 거부한다. `baseline_scored_fraction_by_time`, `scored_fraction_by_time`, `min_scored_fraction`을 함께 기록해 점수와 평가 범위를 확인한다. Gate가 실제로 충돌하는 서로 다른 branch를 제외할 가능성도 있다.

이 점수는 신경망의 Jacobian, Lipschitz 상수 또는 최종 CD의 직접 측정값이 아니다. 위치·시간만 사용하는 국소 점수와 전체 cloud를 입력받는 모델의 학습 난도가 일치한다는 보장도 없다. 선택용 random permutation에 과적합했는지 보기 위해 별도 random draw로 계산한 heldout 점수도 저장한다. Heldout 점수는 실제 생성 품질의 검증 데이터가 아니라 새로운 대응에 대한 점수 일반화 진단이다.

후보 비교의 random 변동을 줄이기 위해 기존 점수용 bijection에서 group이 바뀌지 않은 대응을 유지하고, group 이동으로 풀린 target 인덱스만 들어온 source에 균등하게 다시 배정한다. 고정된 table 사이의 변환에서 입력 bijection이 uniform이면 출력도 group 내부 uniform 법칙을 보존한다. 선택된 후보의 임시 대응은 다음 탐색 단계에서 이어서 사용한다. 점수를 보고 후보를 선택한 탐색 표본의 조건부 분포까지 uniform이라고 보장하는 것은 아니다. Heldout은 별도의 독립 초기 permutation과 seed를 사용해 기준 table에서 최종 table로 옮기며, heldout 점수로 후보를 선택하지 않는다. 점수용 fine permutation은 cache에 저장하지 않고 학습에서도 사용하지 않는다. 학습은 선택된 table의 group 안에서 fresh uniform random bijection을 새로 뽑는다.

각 cloud에서 기준 coarse 배정을 포함한 `num_tables: 3`개의 hard table을 저장한다. 다른 table은 허용된 target 이웃 patch 사이의 source label 교환으로 만든다. 기본 설정은 table당 최대 4회 교환, 라운드당 후보 4개, 점수 계산에 fine permutation 2개다. centroid 비용과 예상 random fine 비용의 증가는 각각 해당 조건의 기준 배정 대비 2% 이내로 제한한다. 이웃 patch gate와 목적지 변화 budget도 적용한다. target kNN gate가 빈 공간을 가로지르는 연결을 완전히 막아주지는 않으며, 얇은 구조나 topology 보존은 별도로 확인해야 한다.

제공 YAML의 `destination_budget`은 체커보드 `0.5`, 말 `0.35`다. 작은 geometry 검증에서 체커보드 `0.35`는 인접 후보와 비용 조건을 통과해도 목적지 변화 제한 때문에 교환이 전부 막혔다. 그래서 체커보드만 `0.5`로 시작한다. 이 선택은 실제 생성 품질이나 heldout 개선으로 검증된 값이 아니며, 기본값 `0.35`에서의 no-op 결과를 성능 비교로 해석하면 안 된다. 새 budget에서도 accepted 수와 heldout 잔차를 먼저 확인한다.

`path_affine_subpatch`는 각 target patch를 주성분 축의 순위로 두 그룹으로 나누고, source도 child centroid 방향의 순위와 같은 capacity로 배정한다. target 좌표를 평균이나 centroid로 대체하지 않는다. 학습 시 table을 뽑은 다음 선택된 group 안에서 fresh random bijection을 뽑는다. subpatch는 random의 범위를 좁히므로 `path_affine`와의 차이는 추가적인 coupling 변경이다. **이 조건의 table 0과 `baseline_score`도 subpatch 분할을 적용한 기준이며, 원본 TG와 동일한 coupling이 아니다.** 세 조건의 실제 생성 결과를 비교해야 이 효과를 구분할 수 있다.

모든 target 인덱스는 한 번씩 사용된다. source·target 원본 점 집합, Gaussian prior, 모델, FM loss, 시간 sampling, 학습 step 수와 추론은 기존 설정을 사용한다. 유한 bank의 순서 없는 empirical cloud 주변분포를 보존한다는 뜻이며 실제 모집단과 완전히 같다는 뜻은 아니다. 생성 과정에는 cache나 국소 affine 식이 필요 없다.

## 먼저 작은 CPU geometry 검증

기존 실험 장비의 프로젝트 루트에서 실행한다.

```powershell
python run_tg_local.py --dataset all --stage preflight --clouds 8
```

실제 `N=256, K=8` 설정으로 각 dataset의 세 조건을 8 clouds씩 준비한다. CPU에서 geometry만 계산하며 학습은 실행하지 않는다. 원래 실험 YAML과 cache 경로를 바꾸지 않고, 새 manifest 폴더 안의 별도 cache를 사용한다. `--clouds`는 이 검증에만 적용되며 전체 실험의 bank 크기를 줄이지 않는다. 이 검증의 manifest는 학습·평가용으로 사용할 수 없다.

출력된 cache `metadata.json`의 `local_summary`에서 다음을 확인한다.

- `accepted_swaps`, `no_feasible_change`: 실제로 대응 변경이 일어났는가.
- `baseline_score`, `guided_score`: 선택용 실제 fine 대응 점수가 낮아졌는가.
- `heldout_baseline_score`, `heldout_guided_score`: 새로운 random fine 대응에서도 감소하는가.
- `baseline_score_by_time`, `heldout_baseline_score_by_time`: 전체 평균이 시간대별 차이를 가리지 않는가.
- `baseline_scored_fraction_by_time`, `scored_fraction_by_time`, `min_scored_fraction`: 비교하는 잔차의 평가 대상 범위가 유지되는가.
- centroid 및 expected fine 비용: 제약 안에 있는가.
- `precompute_seconds`, `local_guidance_seconds`: cloud당 추가 계산 비용이 어느 정도인가.

점수가 내려가지 않거나 대부분 no-op이면 학습 전에 원인을 확인한다. 기본값은 시작점이며 생성 품질에 대해 검증된 최적 설정은 아니다.

실행기는 전체 metadata를 출력하는 대신 점수·heldout·교환 수·평가 비율·준비 시간을 짧게 출력한다. 전체 진단은 출력된 `metadata_path`의 JSON에 저장된다. Heldout 점수가 개선되지 않으면 경고를 표시하지만 전체 학습 실행을 자동으로 중단하지는 않는다.

실제 `N=256, K=8`, seed 0의 8-cloud CPU preflight에서는 다음 결과를 확인했다. 두 variant 모두 선택용 점수는 낮아졌지만 **독립 heldout 점수는 개선되지 않았다.** 이는 작은 단일 seed geometry 검증이며 생성 모델의 품질 평가가 아니다.

| Dataset / 조건 | 8 clouds 준비 시간 | Coarse no-op clouds | Heldout 기준 → 개선안 |
| --- | ---: | ---: | ---: |
| 체커보드 / `path_affine` | 6.40초 | 5/8 | 0.440680896 → 0.440894045 |
| 체커보드 / `path_affine_subpatch` | 6.60초 | 5/8 | 0.289245389 → 0.289336162 |
| 말 / `path_affine` | 6.43초 | 6/8 | 0.691607303 → 0.691665616 |
| 말 / `path_affine_subpatch` | 8.58초 | 0/8 | 0.439178491 → 0.439472054 |

같은 처리율을 4,096 clouds에 단순 외삽하면 variant당 CPU 준비 시간이 약 55–73분이다. 이는 실제 전체 준비 시간을 측정한 값이 아니며 장비·IO·실행 조건에 따라 달라질 수 있다. 현재 heldout 결과는 개선 효과의 근거가 되지 않으므로 이 실험은 검증할 가설로 취급한다.

## 전체 비교 실행

```powershell
python run_tg_local.py --dataset all --dry-run
python run_tg_local.py --dataset all
```

두 번째 명령은 **평가 전용이 아니다.** 말·체커보드 각각 기존 baseline과 두 variant, 총 6개 조건의 cache 준비 → 학습 → NFE `1 2 4 8 16 32 64 128` 평가를 실행한다. 기본값은 각 조건 bank 4,096 clouds, batch 64, 10,000 updates다. 말과 체커보드의 기존 모델 크기와 optimizer 설정을 각각 유지한다. 기존 cache가 같은 설정으로 존재하면 검증 후 재사용하고, 새 학습은 별도 run 폴더에 저장한다.

하나의 dataset만 실행할 수도 있다.

```powershell
python run_tg_local.py --dataset horse
python run_tg_local.py --dataset checkerboard
```

전체 학습은 사용자의 기존 실험 장비에서 실행한다. 작은 CPU 검증은 학습 성능 개선의 증거가 아니다.

## 단계별 실행과 재개

```powershell
python run_tg_local.py --dataset all --stage prepare
```

출력되는 `manifest=.../manifest.json` 경로를 보관한다. 실행기는 입력 YAML snapshot, 실제 준비된 config, 학습 함수가 반환한 정확한 saved run config, 평가 결과 경로를 이 manifest에 기록한다. 폴더 이름을 직접 타이핑하거나 임의의 latest run을 검색하지 않는다. 각 단계가 끝날 때 기록하므로 중간 실패 후 같은 manifest로 완료된 조건을 재사용할 수 있다. 실행 중인 같은 manifest를 다른 프로세스로 동시에 실행하지 않는다.

```powershell
python run_tg_local.py --stage train --manifest "experiment_manifests/실제_실행_폴더/manifest.json"
python run_tg_local.py --stage eval --manifest "experiment_manifests/실제_실행_폴더/manifest.json"
```

`--stage eval`은 이미 기록된 모델만 평가하며 학습하지 않는다. 기본적으로 이미 기록된 NFE 평가를 재사용한다. 새 평가 파일을 다시 생성하려면 다음처럼 실행한다.

```powershell
python run_tg_local.py --stage eval --manifest "experiment_manifests/실제_실행_폴더/manifest.json" --rerun-eval
```

입력 snapshot이나 saved run config가 기록 후 변경됐으면 실행기를 중단한다. 실험 설정을 바꾸려면 새 manifest를 만든다. manifest 경로는 현재 PC의 절대 경로를 포함하므로 다른 PC에 옮긴 뒤 자동 재사용하는 기능은 제공하지 않는다.

## Stream 비교

```powershell
python run_tg_local.py --dataset all --sampling stream --dry-run
python run_tg_local.py --dataset all --sampling stream
```

세 조건 모두 같은 stream 조건으로 실행한다. 기존 bank YAML을 수정하지 않고 별도 stream cache/config를 사용한다. batch 64 × 10,000 updates이므로 **조건마다 640,000 clouds**를 준비한다. 국소 affine 평가 비용도 이에 비례하므로 반드시 작은 preflight의 cloud당 시간을 먼저 확인한다. 전체 stream을 처음부터 준비하면 오랜 시간과 큰 저장 공간이 필요할 수 있다.

새 cache는 format 4의 `pairing_tables`와 `fine_target_labels`를 사용한다. table 개수·N·subpatch 수에 따라 cache 공간이 늘어난다. cache의 실제 바이트 수와 준비 시간을 확인하고, 논문/보고서에는 사전 계산 시간과 학습 시간을 따로 기록한다. baseline도 동일 sampling 조건이어야 한다.

## 결과 판단

평가 JSON·PNG는 `eval_results/checkerboard/`와 `eval_results/horse/` 아래에 저장되고 manifest에 정확한 경로가 기록된다. CD와 함께 checkerboard leakage/cell mass/JS, 말 leakage/JS 및 동일 foreground·gap ROI를 비교한다. `training.json`의 학습 시간, cache metadata의 준비 시간, 평가 JSON의 integration-only 추론 시간을 함께 보고한다. 추론 시간은 평가 batch 전체 기준이며 데이터 로딩·지표·렌더링 시간은 포함하지 않는다.

현재 비교에는 원본 TG baseline이 필수다. randomized swap control을 포함하지 않으므로 variant 간 차이만으로 잔차 감소의 인과 효과를 단정할 수 없다. seed 0에서 방향을 확인한 뒤 여러 학습 seed로 재현해야 한다. NSOT를 넘는지는 동일 조건의 NSOT 결과를 추가해야 판단할 수 있다.

후반 rollout Jacobian은 saved run config로 별도 측정한다. 생성 경로 측정은 cache 없이 가능하다.

```powershell
python audit_jacobian.py "runs/horse/실제_run/config.yaml" --dataset horse --scope rollout --clouds 16 --probes 16 --rollout-steps 128 --times 0 0.25 0.5 0.75 0.875 0.9375 1
```

잔차 점수가 감소해도 CD·JS·구조 지표 또는 후반 Jacobian이 좋아진다는 보장은 없다. 필요한 수축을 유지하는지를 최종 결과로 확인한다.
