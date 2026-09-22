# TG source assignment randomization + target patch audit

두 실험은 분리한다. 첫 번째만 재학습한다. 두 번째는 체크포인트 없이
현재 exact TG의 target partition을 검사한다. 모델, FM loss, 시간 샘플링,
source/target 좌표, 선형 보간, Euler 추론, patch 내부 uniform random pairing은 바꾸지 않는다.

## 1. 제한된 source 할당 무작위화

Target FPS와 balanced partition을 그대로 유지한다. Target patch centroid를 c_k,
기존 exact source 할당을 A_TG라 하면 D(A) = sum_i ||x_i - c_A(i)||^2 이다.

기존 할당에서 시작해 source 점 두 개를 무작위로 뽑고 소속 patch를 교환한다.
각 cloud에서 D(A) <= (1 + relative_budget) D(A_TG)인 교환만 허용한다.
Patch별 점 개수와 양쪽 점 집합은 정확히 유지된다. 보정이 끝난 뒤 patch 내부는
새로운 uniform random permutation으로 pairing한다. 할당을 학습 시간 t별로 바꾸지 않는다.

| 설정 | source 할당 | 비용 상한 |
| --- | --- | --- |
| baseline | 기존 exact TG | D(A_TG) |
| budget01 | exact TG + random swap walk | 1.01 D(A_TG) |
| budget05 | exact TG + random swap walk | 1.05 D(A_TG) |

각각 seed 0/1/2, K=8, N=256, batch=64, 10,000 step. 각 cloud에서
4N번 교환을 제안한다. 같은 patch 점을 뽑거나 상한을 넘기면 그대로 머문다.
이 1%, 5%, 4 sweeps는 검증할 실험 설정이지 최적값이 아니다.

- 비용은 **cloud 전체 centroid 제곱거리 합**이다. 개별 점 이동거리,
  고주파, low-NFE 성능을 보장하지 않는다. float64 수치 허용오차를 JSON에 기록한다.
- 유한한 random walk이며 feasible assignment의 정확한 uniform/Gibbs sampler가 아니다.
  높은 확률의 assignment나 최대 엔트로피 해를 구한다고 주장하지 않는다.
- relative_budget=0 또는 proposal_sweeps=0은 tie가 있어도 기존 TG와 같은 pairing을
  재현하는 통제 설정이다. 별도 YAML은 늘리지 않고 테스트에서 검증한다.
- 별도 CPU 난수 스트림(seed+3)을 사용한다. source/target/time 난수와 기존 local pairing
  스트림(seed+1)을 추가로 소비하지 않는다. 동일 장치·실행 조건에서 비교한다.
- 기존 target/source OT solve는 그대로 두 번이다. 추가 OT solve는 없다.
  source 비용표 O(BNK), swap 작업 O(BN * sweeps), proposal 메모리 O(BN)가 추가된다.
  N^2 비용표는 만들지 않지만 Python/NumPy loop 및 CPU 전송 시간은 든다.
  따라서 **거의 같은 비용은 transport cost를 뜻하며 실행 시간이 같다는 뜻은 아니다.**
- source centroid 할당 후 local pairing은 그대로다. 기존 source-only Sinkhorn의
  soft plan -> hard greedy rounding 실험과 달리 여기서는 실제 source 할당이 확률적으로 변한다.

### 학습: 프로젝트 루트 PowerShell

```powershell
Get-ChildItem randomized_experiments/checkerboard/*.yaml | Sort-Object Name | ForEach-Object {
    python train.py $_.FullName
    if ($LASTEXITCODE -ne 0) { throw "Checkerboard training failed: $($_.Name)" }
}
Get-ChildItem randomized_experiments/horse/*.yaml | Sort-Object Name | ForEach-Object {
    python train_horse.py $_.FullName
    if ($LASTEXITCODE -ne 0) { throw "Horse training failed: $($_.Name)" }
}
```

두 폴더의 총 18개 YAML만 실행한다. 기존 실험 폴더는 실행하지 않는다.
기존 TG 대조군도 새로 훈련한다. 기존 체크포인트/결과는 덮어쓰지 않는다.

`runs/<dataset>/<unique_run>/training.json`의 `coupling_diagnostics.records`에
step 1 및 log_every마다 아래 값을 기록한다. 모든 step의 통계는 아니다.

- `mean_changed_fraction`: 기존 exact 할당과 최종 patch 소속이 다른 점 비율
- `mean_accepted_swaps`: 허용된 교환 수 (되돌아간 교환도 포함)
- `mean/max_relative_cost_increase`: 실제 비용 증가율
- `per_cloud`: 각 cloud의 비용, 상한, 교환 수, 변경 비율

변경 비율이 거의 0이면 실질적인 무작위화 실험이 아니다. 효과 해석 전에 반드시 확인한다.
이 추가 작업/기록 비용도 `training_seconds`에 포함된다.

평가·trajectory 진단은 기존 명령 그대로 **새 run의 동결된 config.yaml**을 사용한다.
예: `python eval_horse.py runs/horse/<새_run>/config.yaml 4`.
NFE는 1/2/4/8/16/32/64/128을 모두 비교한다. `diagnose.py`도 새 coupling을 지원한다.
`summarize_results.py`는 budget별로 seed를 별도 집계하고 같은 비교 그래프에 표시한다.
같은 seed를 재훈련한 체크포인트들이 섞이면 집계를 거부하므로, 비교할 run을 사전에 정해
그 평가 JSON만 별도 결과 폴더에 모은 뒤 요약한다.

## 2. Target patch가 서로 다른 구조를 묶는지 진단

```powershell
python audit_target_patches.py randomized_experiments/checkerboard/tg_baseline_k8_n256_seed0.yaml --dataset checkerboard
python audit_target_patches.py randomized_experiments/horse/tg_baseline_k8_n256_seed0.yaml --dataset horse
```

재학습 없이 dataset당 한 번씩 실행한다. 모든 budget의 target partition이 같으므로
budget마다 반복할 필요는 없다. 기본값: 새 target cloud 16개, 진단 seed=2026,
kNN의 k=8, 반경 배율 0.75/1/1.5, patch당 최대 256개 target-target chord.
YAML의 N, K, device, dtype를 읽고 **항상 exact TG target partition**을 검사한다.
3-seed 학습 검증과는 별개이며 추가 sampling seed는 `--seed`로 바꾼다.

각 cloud의 동일한 점과 FPS anchor로 다음을 비교한다.

1. 실제 TG balanced partition.
2. Nearest-anchor 분할: 균등 capacity를 제거했을 때의 **진단 기준**.
   이 방식으로 훈련하거나 좋은 생성 품질을 가정하는 실험이 아니다.

`analysis_results/<dataset>/target_patch_audit_<unique>/`에 저장한다.

- `patch_audit.json`: 설정/난수 seed/환경/코드 해시, cloud·patch별 수치, 요약, 첫 cloud의 점·label
- `patches.png`: 두 분할과 support 배경, FPS anchor, 빈 공간을 가로지르는 chord 예시
- `connectivity.png`: 그래프 반경에 따른 patch 연결성 (cloud 평균 ± 표본 SD)

판독할 항목:

- `chord_exit_fraction`: 같은 patch의 두 target 점을 잇는 직선이 실제 배경으로 나가는 비율.
  **FM trajectory가 아니다.** 작은 patch는 모든 쌍, 큰 patch는 uniform distinct-endpoint 쌍을
  복원추출한다. 선분 내부를 유한 간격으로 검사하므로 더 작은 틈은 놓칠 수 있다.
- `fragmented_patch_fraction` / `mean_largest_component_fraction`: 반경 제한 및 support 확인 후
  patch 유도 그래프의 연결성. 전체 그래프의 `global_components`, `isolated_point_fraction`도
  반드시 같이 본다. 희소 sampling 때문에 끊긴 것을 patch 오류로 단정하지 않는다.
- 체커보드: `cells_touched`, `point_weighted_cell_mixing_fraction`, `true_cell_counts`.
  Target의 칸별 표본 수가 균등하지 않으므로 equal capacity가 칸 혼합을 강제할 수 있다.
- 말: 해부학적인 부위 정답 label은 없다. 오목하지만 연결된 형상도 exiting chord가 있으므로
  chord/graph 진단만으로 '잘못된 분할'이나 최종 디테일 손실의 인과관계를 증명하지 않는다.

연결성은 sparse SciPy 그래프로 계산하며 밀집 N×N 거리행렬을 만들지 않는다.
그래프 helper는 3D에도 쓸 수 있지만, 여기의 foreground/선분 검사는 **2D support 전용**이다.
Target partition 자체나 3D support 판정을 이번에 변경/구현하지 않는다.

권장 판정 순서: 여러 반경에서 balanced만 구조 혼합이 심한지 확인 → 이미지와 칸별/patch별
수치로 위치 확인 → 그때 target partition 변경 후보를 설계한다. 생성 성능 개선 여부는
source 무작위화의 3-seed low/high-NFE 결과로 별도 판단한다.
