# TG 연구 질문지 답변 — 구현·근거·주장 범위

작성 기준일: 2026-09-29. 구현·문서 정리 반영일: 2026-09-30. 대상: 첨부된 「TG 연구: CVPR 초안 작성을 위한 질문지」의 1–50번.

## 먼저 고정할 범위

1. **이 문서의 본 방법은 기존 TG다.** Target FPS → 고정 capacity의 target 할당 → patch centroid 계산 → source 할당 → 패치 내부 uniform random bijection을 뜻한다.
2. **WCSS·CH·DB는 `audit_k.py`에서 계산하는 학습 없는 partition 진단이며, t=0 조건부 분해 추정기는 미구현이다.** 진단 지표를 성능 향상의 입증된 원인이나 논문의 확정 기여로 넣지 않는다. 질문 39·40의 수학적 정의는 문서 끝의 **별도 수학적 진단 부록 P**로 분리한다.
3. 2D checkerboard·horse 실험과 PSF 백본을 연결한 3D 예비 실험을 구분한다. 2D 결과를 3D 성능의 근거로 대체하지 않는다.
4. 코드·설정 파일, 저장된 평가 기록, 사용자에게서 보고된 관찰, 신규 실험 계획을 구분한다. 구현 테스트 통과는 생성 품질 검증이 아니다.
5. 저장소 HEAD는 `51bc0c38c9fa52117bcaaf9a8edc3787597a1c38`이지만, 현재 작업 트리에는 미커밋 변경이 있다. 따라서 이 commit만으로 아래 구현을 모두 재현했다고 주장하면 안 된다. 논문용 실험 전 commit과 코드 hash를 확정해야 한다.

### 상태 표기의 의미

| 상태 | 의미 |
|---|---|
| 확인됨 — 구현/설정 | 현재 파일 또는 테스트로 확인했다. 완료된 장기 학습을 뜻하지 않는다. |
| 확인됨 — 기록 | 접근 가능한 평가 JSON·요약에 해당 내용이 기록돼 있다. 원 체크포인트까지 다시 검증했는지는 별도로 적는다. |
| 현재 가설 | 가능한 설명이나 설계 동기이지, 성능 원인의 증명이 아니다. |
| 미확인 | 필요한 원본·로그·프로토콜을 아직 확인하지 못했다. |
| 추가 실험 필요 | 해당 주장을 위해 통제 비교나 반복 실험이 더 필요하다. |

현재 저장소 안에는 `runs/`, `eval_results/`, `analysis_results/`가 없다. 다만 이전에 전달된 Downloads의 일부 평가·요약 파일은 접근 가능하다. **“이 저장소에 없다”와 “실험한 적이 없다”는 다른 말**이다. 접근 가능한 과거 결과는 아래 증거 목록 E에 따로 기록한다.

## A. 연구의 기여와 주장 범위

### 1. ★ 성능 결과를 제외한 방법적 기여

**상태: 확인됨 — 구현.** TG는 독립적으로 선택된 noise–target cloud 쌍 안에서, target이 정한 균형 패치로 source 점을 centroid 기반으로 할당하고 패치 내부에는 uniform random bijection을 남기는 **학습 시점의 계층적 점 대응 방식**이다.

변경 대상은 coupling이다. Source/target 점의 좌표, 모델, 선형 FM 경로, velocity MSE, 시간 샘플링 및 inference 알고리즘은 비교 조건 내에서 유지한다. 핵심은 세 단계의 조합이지, FPS·centroid·OT solver 각각의 발명이 아니다.

**근거:** [coupling.py](coupling.py)의 `balanced_target_partition`, `coupling_permutation`, `pair_within_regions`; [train.py](train.py)의 `coupled_flow_matching_loss`.

**범위·조치:** 방법 정의는 바로 작성 가능하다. 최초성·우월성은 질문 47–48의 선행연구와 실험에 근거해 별도로 주장해야 한다.

### 2. ★ 확보된 결과가 뒷받침하는 주장

**상태: 확인됨 — 일부 과거 기록 / 현재 구현과의 대응은 추가 확인 필요.** 접근 가능한 과거 horse seed-summary에는 Independent와 TG의 다중 seed 비교가 있다. 제한된 source 무작위화 평가 JSON 및 두 데이터셋의 patch audit도 별도 존재한다. 구체적 범위는 증거 목록 E에 기록한다.

안전한 주장은 해당 데이터셋·N·학습 설정·NFE·지표 범위에 한정한다. “모든 데이터셋”, “3D에서도 개선”, “고주파를 보존하는 원리를 증명”으로 확대하지 않는다. 사용자에게서 이전에 보고된 exact 우위·independent의 sharpness 관찰도, 원 평가 파일과 연결되기 전에는 정성적 관찰로 표기한다.

**근거:** 증거 목록 E; [experiment.py](experiment.py)의 checkpoint hash·설정 검증 및 [summarize_results.py](summarize_results.py)의 seed 집계.

**범위·조치:** 과거 기록의 checkpoint hash, 당시 coupling 구현, 평가 코드 버전을 연결하는 run manifest가 필요하다. 현재 코드로 새로 얻은 결과와 합쳐 평균내지 않는다.

### 3. ★ 헤드라인 NFE 구간과 high-NFE 주장

**상태: 일부 기록 확인 / 논문용 최종 주장 추가 실험 필요.** 과거 평가 기록과 현재 진단 도구는 NFE 1, 2, 4, 8, 16, 32, 64, 128을 포함한다. 일반 2D evaluator의 기본값은 NFE=100의 단일 평가이고, 3D evaluator의 기본 sweep은 위 8개 NFE다. Low-NFE와 high-NFE를 모두 보여주되, 유리한 몇 점만 사후 선택하지 않는다. 1–8을 low-NFE, 32–128을 high-NFE로 묶는 것은 가능한 보고 방식이지 검증된 경계가 아니다.

“동등하다”는 표현에는 사전 허용 차이와 반복 변동의 검토가 필요하다. 평균이 비슷하거나 오차 막대가 겹치는 것만으로 동등성이 입증되지는 않는다. 128 step도 자동으로 적분 수렴을 뜻하지 않는다.

**근거:** [diagnose.py](diagnose.py)의 `NFES`, reference/doubled-reference 비교; 증거 목록 E.

**조치:** 같은 checkpoint를 모든 NFE에서 평가하고, high-NFE에서는 128→256→필요시 512 수렴을 확인한다. 지표별 개선과 손실을 함께 보고한다.

### 4. 효율성에 대한 정확한 주장

**상태: 확인됨 — inference 절차 불변 / 나머지는 실험 주장.** 같은 모델·sampler·NFE라면 TG coupling을 inference에서 계산하지 않는다는 것은 코드로 확인된다. 그러나 같은 품질에 필요한 NFE 감소, 실제 latency 감소, 전체 학습·생성 비용 절감은 각각 별도의 측정 대상이다.

**근거:** [sample.py](sample.py)의 `integrate_velocity`; [eval.py](eval.py), [eval_horse.py](eval_horse.py), [eval_3d.py](eval_3d.py).

**범위·조치:** “추론 절차에 추가 coupling 연산이 없다”는 표현은 가능하다. “학습까지 더 빠르다”는 주장은 현재 근거로 확정하지 않는다. 특히 과거 TG의 큰 training overhead를 숨기지 않는다.

## B. 문제 설정과 데이터

### 5. ★ 생성 샘플의 단위와 coupling 수준

**상태: 확인됨 — 구현.** 모델의 샘플 단위는 전체 집합 $X\in\mathbb R^{N\times d}$이다. 2D는 $d=2$, PSF 경로는 $d=3$이다. 각 target cloud에 독립 Gaussian noise cloud를 생성한 후, 같은 batch index의 두 cloud 내부에서 대응을 정한다. Cloud 사이의 minibatch/outer OT는 구현되어 있지 않다.

**근거:** [train.py](train.py)의 `coupled_flow_matching_loss`; [coupling.py](coupling.py)의 batch별 assignment.

**범위·조치:** 논문에서는 **intra-cloud/inner coupling**이라고 명시한다. 일반 OT-CFM의 dataset-sample coupling을 그대로 재현했다고 쓰지 않는다.

### 6. ★ 데이터셋·카테고리·split·조건부 생성

**상태: 확인됨 — 로더·설정 / 실제 3D split별 shape 수는 미확인.**

- Checkerboard: `sample_checkerboard`가 4×4 격자의 8개 활성 cell에서 점을 직접 샘플링한다. 고정된 train/test shape 목록이 없다.
- Horse: `skimage.data.horse()`의 반전 foreground mask에서 매번 새 점 집합을 샘플링한다. 하나의 silhouette 분포를 학습하며, 여러 말 형상 생성 데이터셋이 아니다.
- 3D: PSF의 `ShapeNetCore.v2.PC15k` 로더를 재사용한다. 현재 YAML의 category는 `chair`; bridge는 한 category 문자열을 받는다. 모델에 class label은 입력하지 않는다. `num_classes=3`은 XYZ velocity 출력 채널 수다.

**근거:** [data.py](data.py), [train_horse.py](train_horse.py), [psf_adapter.py](psf_adapter.py)의 `shapenet_dataset`, [PSF dataset loader](third_party/PSF/datasets/shapenet_data_pc.py), [3D TG 설정](psf_experiments/target_guided.yaml).

**범위·조치:** 정확한 데이터 다운로드 원본·압축 파일 checksum, train/val/test shape 수·ID 목록·겹침 여부를 랩실 데이터에서 수집해야 한다. 파일명만으로 특정 논문의 동일 preprocessing/split임을 확정하지 않는다.

### 7. ★ N과 subsampling

**상태: 확인됨 — 구현/기본 설정.** 2D 기본 비교는 N=256이다. 일부 horse 설정에 N=512/1024가 있지만, 설정 파일 존재만으로 완료된 결과를 뜻하지 않는다. 3D 기본 설정은 N=2048이다.

2D는 매 batch 새 점을 생성한다. Horse는 foreground pixel을 복원추출하고 pixel 내부 jitter를 더한다. PSF 로더는 각 shape의 15k 점 중 첫 10k를 `train_points`로 두고, 호출마다 `np.random.choice`의 기본 `replace=True`로 N개를 뽑는다. 따라서 같은 좌표 index가 중복될 수 있다. 평가 reference도 held-out **shape**의 첫 10k 점에서 이 방식으로 뽑는다. TG partition cache는 없다.

**근거:** [train_horse.py](train_horse.py)의 `sample_horse`; [PSF dataset loader](third_party/PSF/datasets/shapenet_data_pc.py)의 `__getitem__`; [eval_3d.py](eval_3d.py)의 reference 구성.

**범위·조치:** “비복원 2048점 샘플링” 또는 “마지막 5k에서 평가한다”라고 쓰면 현재 bridge와 다르다. Augmentation은 현재 연결 경로에 별도 회전·스케일 변형이 없다.

### 8. ★ 정규화와 평가 좌표계

**상태: 확인됨 — 구현.** Checkerboard는 [-1,1]² 좌표다. Horse는 mask의 긴 변 길이를 기준으로 종횡비를 유지해 좌표를 배치한다. 두 경우 cloud마다 재중심화하지 않는다.

PSF bridge는 `normalize_per_shape=False`, `normalize_std_per_axis=False`다. 선택한 category의 **train split에 로드된 모든 shape의 각 15k 점들**을 사용해 전역 3차원 평균과 하나의 scalar 표준편차를 계산한다. 이 scalar는 upstream의 좌표 전체 flatten 표준편차이지, 별도로 정의한 radial RMS나 축별 표준편차가 아니다. 평가에서는 checkpoint에 저장된 training 통계를 reference에 적용하며 생성물과 함께 이 정규화 좌표계에서 측정한다. Shape별 역정규화는 하지 않는다.

**근거:** [psf_adapter.py](psf_adapter.py), [train_3d.py](train_3d.py)의 `normalization`, [eval_3d.py](eval_3d.py).

**범위·조치:** 단위 구면 정규화나 per-shape normalization이라고 기술하지 않는다. 외부 benchmark와 비교 전 동일 좌표계인지 확인해야 한다.

### 9. ★ Source 분포

**상태: 확인됨 — 구현.** 각 cloud의 각 점·좌표는 표준 정규분포에서 독립적으로 생성한다. Noise cloud centering, clipping, 데이터 의존 scale, correlation은 없다. TG는 이 좌표를 바꾸지 않고 target의 대응 순서만 바꾼다.

**근거:** [train.py](train.py)의 `torch.randn`; [experiment.py](experiment.py)의 `sample_for_evaluation`; [eval_3d.py](eval_3d.py)의 noise 생성.

**범위·조치:** “Source는 iid standard Gaussian”이라고 작성 가능하다. 실제 과거 run에 다른 noise scaling이 없었는지는 해당 코드 버전으로 확인한다.

### 10. ★ FM 경로와 velocity target

**상태: 확인됨 — 구현.**

$$
X_t=(1-t)X_0+tY_\pi,\qquad U=Y_\pi-X_0.
$$

$Y_\pi[i]=Y[\pi(i)]$이며 source index는 유지한다. 별도 $\sigma_{\min}$ 항, 경로에 더하는 noise, endpoint parameterization은 없다. PSF의 `t*999`는 백본에 전달하는 time embedding 입력 변환이고, FM 경로 자체의 time schedule 변경은 아니다.

**근거:** [train.py](train.py)의 `linear_path`, `flow_matching_loss`; [psf_adapter.py](psf_adapter.py)의 `PSFVelocity.forward`.

**범위·조치:** Methods에 그대로 명시할 수 있다. 원 PSF의 후속 reflow/distillation 단계와 혼합해 설명하지 않는다.

### 11. ★ Loss와 시간 샘플링

**상태: 확인됨 — 구현.** `F.mse_loss` 기본 reduction으로 batch·point·coordinate 전체 평균을 계산한다. 시간은 cloud마다 하나의 $t\sim U[0,1)$, tensor shape `[B,1,1]`이며 같은 cloud의 모든 점에 공유한다. `eps=0`이 기본값이다. 보조 loss/head, self-conditioning, 시간별 loss 가중치는 없다.

**근거:** [train.py](train.py)의 `sample_time`, `flow_matching_loss`.

**범위·조치:** loss의 “평균”은 좌표까지 평균낸 값이다. 부록의 trace covariance처럼 좌표를 합산한 통계와 수치를 직접 동일시하지 않는다.

## C. Coupling 알고리즘 — 기존 TG

### 12. ★ 전체 의사코드

**상태: 확인됨 — 구현.** 입력은 같은 shape의 `X0,Y: [B,N,d]`와 K, 출력은 `pi: [B,N]` target index다.

```text
각 cloud 쌍 (X0, Y)에 대해:
  1. a[1:K] = FPS(Y); 시작점은 Y의 평균에서 가장 먼 점
  2. n[k] = floor(N/K) + 1[k < N mod K]
  3. target label L = exact balanced assignment(||y_j-a_k||², n)
  4. c[k] = mean{y_j : L[j]=k}
  5. source label A = exact balanced assignment(||x_i-c_k||², n)
  6. 각 k에서 source index S_k, target index P_k를 수집
  7. pi[S_k] = P_k[uniform randperm(n[k])]
return pi
학습 함수에서 Y_pi = gather(Y, pi); X0는 재배열하지 않음
```

**근거:** [coupling.py](coupling.py)의 FPS, partition, centroid, assignment, pairing 함수; [train.py](train.py)의 gather.

**범위·조치:** `coupling_permutation`은 `torch.no_grad()`이고 solver 입력도 detach된다. Coupling을 미분해서 모델과 함께 최적화하지 않는다.

### 13. ★ Target anchor 선택

**상태: 확인됨 — 구현 / 순서 민감도에 대한 포괄적 검증은 미확인.** 현재 TG 기본 K=8. FPS는 해당 iteration에 샘플링한 target cloud에서 수행하고 superset에서 미리 계산하지 않는다. 시작점은 set mean에서 가장 먼 점으로 결정적이다. 이후 현재 anchor까지의 최소 제곱거리가 가장 큰 점을 고른다.

`1≤K≤N`을 검사한다. 선택한 index의 거리를 -1로 바꿔 중복 좌표가 있어도 같은 index를 반복 선택하지 않는다. 동률은 `argmax`의 index 선택과 solver tie-breaking에 의존한다.

**근거:** [coupling.py](coupling.py)의 `farthest_point_sample`; `tests/test_project.py`의 duplicate-point 검사.

**범위·조치:** “random FPS start”는 틀리다. 동률이 있는 입력에 대한 완전한 순서 불변성, GPU/CPU bitwise 일치를 주장하지 않는다.

### 14. ★ Target partition과 capacity

**상태: 확인됨 — 구현.** 비용은 $C^Y_{jk}=\|y_j-a_k\|^2$. N×K 수송 문제를 풀어 모든 점을 하나의 anchor에 할당하고, capacity는 위 의사코드의 정수 $n_k$로 고정한다. N이 K로 나뉘면 동일 크기이고, 아니면 앞쪽 anchor부터 나머지 한 점씩을 받는다. N≥K이므로 기본 TG의 모든 패치는 비어 있지 않다.

**근거:** [coupling.py](coupling.py)의 `balanced_target_partition`, `_exact_assignment_numpy`.

**범위·조치:** `region_centroids`에서 counts를 다시 계산한다고 가변 capacity가 되는 것이 아니다. 기존 TG는 균등/최대한 균등 capacity이고, nearest 변형과 구분해야 한다. Solver 경고·비정상 plan은 예외를 내며 자동 approximate fallback은 없다.

### 15. ★ Source assignment의 기준과 비용

**상태: 확인됨 — 구현.** 기준점은 FPS anchor가 아니라 **target 할당 후 계산한 patch centroid**다. 비용은 squared Euclidean distance뿐이며 Mahalanobis, 표면 거리, 주파수, geometry penalty, 모델 예측 오차 항은 없다.

**근거:** [coupling.py](coupling.py)의 `region_centroids`, `assign_regions`, `coupling_permutation`.

**범위·조치:** centroid 보조정리는 질문 37처럼 고정 target partition과 capacity 조건에서 기술한다. 학습하기 쉬운 평균장을 직접 최소화하는 목적함수라고 쓰지 않는다.

### 16. ★ OT marginal·solver·종료 조건

**상태: 확인됨 — 구현/로컬 라이브러리 / 각 과거 run 버전은 별도 확인.** 실제 호출은 `ot.emd(np.ones(N), capacities, cost, log=True)`이다. 행 질량은 1, 열 질량은 $n_k$, 총 질량은 N이다. 모두 N으로 나눈 확률 marginal과 optimizer는 같지만, 반환 plan의 값은 현재 **정수 할당 스케일**로 해석해야 한다.

현재 exact solver는 POT network simplex다. 비용은 tensor dtype에서 `torch.cdist(...).square()`로 계산한 후 CPU float64 contiguous array로 변환한다. 후처리 float64가 원래 거리 계산의 float32 반올림을 없애지는 않는다. `numItermax`는 명시하지 않는다. 현재 로컬 POT 0.9.7.post1 함수 signature의 기본값은 100000, `numThreads=1`이다. requirements는 POT 0.9.7을 지정한다.

**근거:** [coupling.py](coupling.py)의 `_exact_assignment_numpy`, [requirements.txt](requirements.txt); 로컬 `inspect.signature(ot.emd)` 확인.

**범위·조치:** 논문에는 run의 `environment.POT`를 우선 기록한다. 비용 최소화는 유한 정밀도 solver가 정상 종료한 범위이며, 코드가 독립적으로 primal-dual optimality certificate까지 검사하는 것은 아니다.

### 17. ★ Plan에서 assignment·bijection으로 변환

**상태: 확인됨 — 구현/테스트.** Plan을 반올림한 `hard`와 원 plan이 `atol=1e-7, rtol=0`으로 가까운지, 행합이 1인지, 열합이 정수 capacity인지 확인한다. Solver warning이나 조건 위반이 있으면 예외를 낸다. 통과 후 `argmax`로 patch label을 읽는다. 이는 임의 fractional plan을 argmax해서 강제로 만드는 방식이 아니다.

Source/target patch별 수가 같은지 pairing 때 다시 검사하고, 각 patch 안에서 permutation을 사용하므로 최종 대응은 bijection이다. 최종 정렬 검사로 모든 index가 한 번씩 나오는지 확인하는 테스트·benchmark가 있지만, 매 training call 끝에 별도 전체-sort assertion을 수행하지는 않는다.

**근거:** [coupling.py](coupling.py), [tests/test_project.py](tests/test_project.py), [tests/test_tg_cost.py](tests/test_tg_cost.py), [benchmark_coupling.py](benchmark_coupling.py).

**범위·조치:** Runtime 예외는 존재하지만 모든 training 실패를 구조화한 JSON failure log가 있는 것은 아니다. 실패 수·조건을 논문에 보고하려면 로그 보강이 필요하다.

### 18. ★ 패치 내부 random pairing

**상태: 확인됨 — 구현.** `torch.where`로 얻은 source index 순서는 유지하고 target patch index에 `torch.randperm$n_k$`를 적용한다. 고정 patch에서 uniform random bijection을 새로 뽑는다. 각 coupling 호출마다 갱신되고 cache하지 않는다.

**근거:** [coupling.py](coupling.py)의 `pair_within_regions`; [train.py](train.py)의 별도 `torch.Generator(...).manual_seed(seed+1)`.

**범위·조치:** “patch 안의 모든 pair가 동시에 독립”이라고 쓰지 않는다. 한 bijection 안의 대응들은 비복원 제약 때문에 서로 의존한다. 단일 pair의 주변분포와 전체 permutation의 결합분포를 구분해야 한다.

### 19. ★ Iteration마다 변하는 무작위 요소

**상태: 확인됨 — 구현.** 2D에서는 새 target 점, 새 Gaussian source, cloud별 t, 내부 pairing이 변한다. 3D에서는 shape 순서, 복원 point subsampling, source, t, 내부 pairing과 training dropout이 추가된다. FPS 시작은 random이 아니며, 명시적인 무작위 tie-break·마지막 joint permutation·회전 augmentation은 없다.

**근거:** [train.py](train.py), [train_horse.py](train_horse.py), [train_3d.py](train_3d.py)의 RNG·sampler 저장, [PSF loader](third_party/PSF/datasets/shapenet_data_pc.py).

**범위·조치:** 같은 seed·코드·환경에서 재현을 지향하지만 전 장치 bitwise determinism을 보장하는 옵션은 없다. 3D resume은 optimizer/RNG/sampler 및 코드 hash를 검사한다. 2D runner는 완전한 optimizer-state resume 도구가 아니다.

## D. 분포 보존과 알고리즘의 성질

### 20. ★ 분포 보존의 정확한 의미

**상태: 확인됨 — 유한 집합 보존.** Source 좌표와 순서를 그대로 두며 target index만 bijection으로 바꾼다. 따라서 각 cloud의 empirical point measure와 **unordered point set**은 정확히 보존된다. Source의 원 Gaussian tensor law도 좌표를 건드리지 않으므로 유지된다.

그러나 source에 의존해 target을 재배열했다는 사실만으로 **전체 ordered target tensor law**까지 원래와 같다고 일반적으로 결론낼 수 없다. 그러려면 교환가능성·알고리즘의 equivariance 등 추가 조건을 증명해야 한다. 코드에는 마지막 joint random permutation이 없다.

**근거:** [coupling.py](coupling.py)의 반환 index; [train.py](train.py)의 target-only gather.

**범위·조치:** 현재 안전한 기술은 “preserves both input point sets exactly”다. 점별 marginal 보존과 ordered-cloud joint law 보존을 혼동하지 않는다.

### 21. ★ Target 정보가 모델 입력으로 전달되는가?

**상태: 확인됨 — 구현.** 모델 입력은 보간된 전체 $X_t$와 t다. Anchor, centroid, patch ID, 정답 shape label은 별도 feature로 전달되지 않는다. Target은 training pair와 velocity label을 구성하는 데 사용된다. 추론 입력은 noise와 적분 과정의 시간·현재 상태다.

**근거:** [train.py](train.py), [model.py](model.py), [train_horse.py](train_horse.py), [psf_adapter.py](psf_adapter.py), [sample.py](sample.py).

**범위·조치:** 같은 dataset 내 비교에서 “training coupling만 변경하고 backbone 및 inference를 유지한다”는 설명이 가능하다. 서로 다른 2D/3D 백본을 같은 모델이라고 쓰지는 않는다.

### 22. 모델과 coupling의 permutation 성질

**상태: 확인됨 — 2D 구조와 일부 테스트 / 3D 및 동률 경우 포괄 검증은 미확인.** 2D Transformer에는 point-index embedding이 없고 shared projection·self-attention을 사용한다. Checkerboard 모델의 evaluation-mode permutation equivariance 수치 테스트가 있다. Horse도 같은 대칭 구조를 사용하지만 전 설정·dtype에 대한 독립 검증을 완료했다고 확대하지 않는다.

Coupling은 tie가 없는 경우의 기하학적 선택과 동률 시 index 의존성을 구분해야 한다. 입력 순서를 바꾸고 같은 RNG state를 사용해도 동일한 실현 permutation을 요구할 수는 없다. Uniform random bijection의 **분포적 대칭성**과 하나의 샘플의 일치가 다르다.

**근거:** [model.py](model.py), [train_horse.py](train_horse.py), [tests/test_project.py](tests/test_project.py), FPS와 solver 구현.

**범위·조치:** PVCNN의 FPS·voxel/custom kernel까지 포함한 순서/회전 성질은 따로 검증한다. 2D 테스트를 PVCNN의 증명으로 사용하지 않는다.

### 23. 극한·작은 문제 sanity check

**상태: 확인됨 — 수학/구현 테스트.** K=1이면 cloud 전체 안의 uniform random bijection이다. Independent의 고정 index pairing과 개별 대응은 다르지만, 독립·교환가능한 샘플 설정에서는 independent coupling과 관련되는 극한이다.

K=N이면 각 patch는 한 점이다. Exact target/source solver와 제곱거리 아래에서는 full pointwise OT와 같은 최적 비용을 얻는다. 동률 때문에 최적 permutation index까지 유일하게 같을 필요는 없다.

**근거:** [tests/test_project.py](tests/test_project.py)의 `test_kn_equals_global`, `test_exact_rectangular_cost_equals_scipy_slots`, remainder/duplicate 검사. N×K exact 비용을 capacity만큼 열을 복제한 SciPy assignment와 대조한다.

**범위·조치:** 테스트는 알고리즘 일관성이지 생성 성능 비교가 아니다. K=N을 대규모 학습의 실용 설정으로 추천한다는 뜻도 아니다.

## E. 모델과 학습 설정

### 24. ★ Backbone 구성

**상태: 확인됨 — 현재 2D 구성 및 PSF 연결 / 실제 run과의 대조는 별도.**

| 항목 | Checkerboard 기본 | Horse 기본 | 3D PSF 예비 설정 |
|---|---|---|---|
| 모델 | `PointSetTransformer` | `HorsePointSetTransformer` | PSF의 `PVCNN2Base`를 감싼 `PSFVelocity` |
| hidden/head/layer | 128 / 4 / 2 | 256 / 8 / 4 | upstream SA·FP block config 재사용 |
| FFN | 256 | 1024 | PVCNN·PointNet 계열 구성 |
| trainable parameters | 265,730 | 3,291,906 | 27,649,987 — 아래 구조 계수 조건 참조 |
| time | 좌표에 scalar t를 concat | sinusoidal time embedding + MLP | embed_dim=64, backbone time=t×999 |
| dropout | 0 | 0 | 0.1 |
| normalization | pre-LayerNorm Transformer | pre-LayerNorm Transformer | upstream GroupNorm 등 구성 |
| point-index embedding | 없음 | 없음 | 별도 class/index 입력을 bridge가 추가하지 않음 |

**근거:** [model.py](model.py), [train_horse.py](train_horse.py), [psf_adapter.py](psf_adapter.py)의 `create_backbone`; [PSF 모델](third_party/PSF/model/pvcnn_generation.py).

PSF pin은 `c74b39e1200513039cfb8d776505fb75da599e68`이고 Windows patch를 적용한다. 실제 upstream builder가 만드는 SA의 PVConv 개수는 **2·1·1·0**, FP는 **3·3·2·2**다. 선언 tuple의 block 수와 실제 생성 개수가 다르므로 2·3·3·0으로 기술하지 않는다.

| PSF stage | SA centers / radius / neighbors / MLP | PVConv channels / voxel resolution |
|---|---|---|
| SA1 | 1024 / 0.1 / 32 / (32,64) | 32 / 32 |
| SA2 | 256 / 0.2 / 32 / (64,128) | 64 / 16 |
| SA3 | 64 / 0.4 / 32 / (128,256) | 128 / 8 |
| SA4 | 16 / 0.8 / 32 / (256,256,512) | 없음 |
| FP1 | MLP (256,256) | 256 / 8 |
| FP2 | MLP (256,256) | 256 / 8 |
| FP3 | MLP (256,128) | 128 / 16 |
| FP4 | MLP (128,128,64) | 64 / 32 |

Voxel feature는 같은 voxel의 점 feature를 평균 집계하고, 3D convolution 뒤 trilinear interpolation으로 점에 되돌려 pointwise branch와 더한다. SA는 FPS/ball query와 max aggregation, FP는 이웃 보간 및 skip feature를 사용한다. 시간 embedding은 sinusoidal 64차원 및 64→64→64 MLP이고, output head는 64→128→3이다. 여기의 attention module 인자 8은 GroupNorm group 수이지 eight-head attention을 뜻하지 않는다.

2D parameter 수는 실제 CPU 모델 생성으로 계산했다. 3D 수는 CUDA import를 막은 constructor-only 구조 계수다. Parameter를 갖는 upstream layer는 그대로 사용했지만 functional backend를 placeholder로 대체했으므로 **GPU forward 검증으로 해석하면 안 된다**. 논문 run에서는 실제 backend를 로드한 모델의 parameter count/module dump를 함께 저장한다.

**범위·조치:** 최종 parameter 수·module dump를 각 paper run의 메타데이터에 저장한다. PSF 백본 사용은 원 논문의 전 학습 알고리즘을 재현했다는 뜻이 아니다.

### 25. ★ 전체 학습 설정

**상태: 확인됨 — YAML/runner 기본값.**

| 항목 | Checkerboard | Horse N=256 | 3D PSF pilot |
|---|---:|---:|---:|
| N / 기본 K | 256 / 8 | 256 / 8 | 2048 / 8 |
| optimizer | AdamW | AdamW | Adam, betas=(0.5,0.999) |
| LR | 0.001 | 0.0003 | 0.0002 |
| weight decay | 0.01 | 0.01 | 0 |
| microbatch | 64 | 64 | 16 |
| accumulation | 없음 | 없음 | 16 |
| effective batch | 64 | 64 | 256 |
| 기본 optimizer updates | 10,000 | 10,000 | 600,000 |
| LR schedule / EMA | 없음 / 없음 | 없음 / 없음 | 없음 / 없음 |
| precision | float32 | float32 | float32, AMP 없음 |

2D AdamW의 betas는 PyTorch 기본값을 사용한다. Gradient clipping은 없다. 평가 weight는 raw weight다. 3D의 600k는 **현재 YAML 기본값**이지 완료한 학습량이 아니다. CLI `--steps`와 저장된 checkpoint step이 실제 run의 기준이다.

**근거:** [2D TG 설정](checkerboard_experiments/target_guided.yaml), [horse TG 설정](horse_experiments/horse_target_guided_k8_n256_seed0.yaml), [3D TG 설정](psf_experiments/target_guided.yaml), 각 runner.

**범위·조치:** GPU를 관행적으로 “A6000”이라고 쓰지 말고 학습 checkpoint/`training.json`의 `environment.gpu`를 사용한다. 2D 평가 JSON에는 `training_environment.gpu`로 복사되지만, 3D 평가 JSON의 `evaluation_environment`는 평가 환경이다. RTX A6000과 RTX 6000 Ada Generation은 구분한다. 실제 GPU 개수·CPU·총 시간은 run에서 수집한다.

### 26. 학습 seed와 평가 seed

**상태: 일부 기록 확인 / 최종 실험 집합 미확정.** Horse 설정 파일에는 seed 0·1·2가 있으며, 과거 seed-summary도 세 학습 seed를 묶는다. 반면 제한된 무작위화의 현재 접근 가능한 JSON은 단일 seed 비교 범위다. YAML 수와 완료한 독립 학습 수를 동일시하지 않는다.

2D 기본 evaluation seed는 1이다. `summarize_results.py`의 mean±std는 **학습 seed 간 sample SD(ddof=1)**다. 한 checkpoint의 noise 반복과 독립 training 반복을 합쳐 seed 수를 늘리지 않는다.

**근거:** [experiment.py](experiment.py)의 `evaluation_settings`; [summarize_results.py](summarize_results.py); 증거 목록 E.

**범위·조치:** 최종 run마다 training seed와 evaluation seed를 둘 다 고정·기록한다. SD와 confidence interval은 구분하고, noise/reference 반복은 별도의 평가 변동으로 보고한다.

### 27. ★ Checkpoint·하이퍼파라미터 선택

**상태: 확인됨 — 저장 방식 / 사전 선택 프로토콜은 미확인.** 2D runner는 설정된 마지막 step의 모델을 저장한다. 3D는 periodic/latest 및 milestone checkpoint를 저장한다. 자동 validation-best 선택은 없다. Evaluator는 사용자가 지정한 checkpoint를 평가한다.

**근거:** [experiment.py](experiment.py)의 `save_training`; [train_3d.py](train_3d.py)의 저장; [summarize_results.py](summarize_results.py)의 동일 seed 다중 checkpoint 제외 정책.

**범위·조치:** “validation으로 선택했다”는 근거가 없다면 쓰지 않는다. 공통 학습 예산·선택 step·K 선택 규칙을 정하고, 같은 checkpoint를 NFE 전체에 사용한다. 과거 test 관찰로 설계를 바꾼 탐색 과정도 숨기지 않는다.

## F. 비교군과 평가 프로토콜

### 28. ★ 비교군의 정확한 정의

**상태: 확인됨 — 현재 method registry.**

| 이름 | Target partition / capacity | Source assignment | 내부 pairing | 주의 |
|---|---|---|---|---|
| Independent | 없음 | 없음 | 원 index pairing | 매번 randperm하는 구현이 아님 |
| TG | FPS anchor에 exact balanced / 거의 균등 | centroid로 exact | uniform random | 본 방법 |
| TG exact optimized | TG와 동일 | batched transfer 후 cloud별 exact | uniform random | 수학적 목적 동일, 전송 구현 최적화 |
| source greedy | TG와 동일 | 거리 기반 capacity-preserving greedy | random | exact의 동의어가 아님 |
| source Sinkhorn | TG와 동일 | Sinkhorn-log 후 capacity rounding | random | soft plan을 그대로 학습에 쓰는 것이 아님 |
| TG Sinkhorn | target도 Sinkhorn+rounding | source도 Sinkhorn+rounding | random | source만 바꾸는 ablation 아님 |
| `geometry_aware_ot` | 같은 FPS의 nearest partition / 실제 counts | centroid로 exact | random | 균등 capacity 유지하지 않음 |
| Global OT | patch 없음 | intra-cloud N×N exact | pointwise assignment | outer minibatch OT 아님 |
| Regional | source·target을 각각 분할 | patch centroid 간 matching | random | target-only TG와 다름 |
| Strict | checkerboard의 oracle cell | exact | random | FPS partition 아님 |
| Strict local | 같은 oracle cell | exact | local exact | Strict와 비교하면 내부 pairing만 바뀜 |
| Strict balanced | 같은 oracle cell | greedy | local exact | source solver도 다름 |

현재 `global_hungarian`, `geometry_aware_hungarian`은 각각 `global_ot`, `geometry_aware_ot`의 alias다. 현재 solver는 이름과 달리 POT다. 과거 run도 같았다고 가정하면 안 된다.

**근거:** [coupling.py](coupling.py)의 `METHODS`, `ALIASES`, 분기.

**범위·조치:** **일반 FPS-TG와 정확히 같은 partition/source 할당에서 local pairing만 OT로 바꾸는 공개 method는 현재 없다.** `target_guided_strict_local`을 그 실험으로 잘못 표기하지 않는다. 이전 soft-boundary/제한된 무작위화는 현재 활성 방법에 없으므로 별도 역사적 실험으로 분리한다.

### 29. ★ Coupling 외 조건의 동일성

**상태: 확인됨 — 대응 baseline YAML / 과거 모든 run의 동일성은 별도 확인.** 같은 데이터셋의 기본 Independent/TG 설정은 같은 모델·학습 예산·source/loss/sampler를 사용한다. Checkerboard와 horse의 모델 크기·LR은 서로 다르므로 두 데이터셋 사이를 “같은 모델”로 비교하지 않는다.

Pairing은 별도 RNG를 사용하고 deterministic partition은 global RNG를 소비하지 않으므로 대응 seed에서 source/target/t draws를 맞추기 유리하다. 하지만 모든 GPU 실행의 bitwise 동일성을 보장하는 것은 아니다.

**근거:** 기본 YAML들, [train.py](train.py), [train_horse.py](train_horse.py); 증거 목록 E.

**범위·조치:** 실제 평가 JSON의 config·model·training·environment를 비교한다. 공통 hyperparameter 비교와 방법별 tuning 결과를 분리한다.

### 30. ★ Sampler와 NFE

**상태: 확인됨 — 구현.** Uniform grid의 forward Euler이며 S steps에서 $t_s=s/S$, s=0,…,S−1이다. 한 step당 네트워크 한 번이므로 sample당 NFE=S다. 마지막 별도 endpoint head 호출은 없다.

2D 평가의 warm-up 10회와 그림용 snapshot 재적분은 보고되는 **샘플링 NFE/측정 sampling latency 바깥**이다. 스크립트의 총 forward 호출 수가 S라는 뜻은 아니다. 3D도 warm-up과 여러 generation batch를 구분해야 한다.

**근거:** [sample.py](sample.py), [experiment.py](experiment.py)의 `sample_for_evaluation`, [eval.py](eval.py)의 `sample_snapshots`, [eval_3d.py](eval_3d.py).

**범위·조치:** 논문 caption에서 per-sample NFE와 timing 구간을 명시한다. Heun 1 step을 1 NFE로 세는 식의 혼용은 하지 않는다.

### 31. ★ 지표 정의와 집계 방향

**상태: 확인됨 — 구현.** 공통 cloud distance는

$$
CD(A,B)=\frac1{|A|}\sum_{a\in A}\min_{b\in B}\|a-b\|^2
+\frac1{|B|}\sum_{b\in B}\min_{a\in A}\|b-a\|^2.
$$

두 방향의 **합**이며 1/2를 곱하지 않는다.

- 2D `chamfer`: 생성 cloud와 같은 batch index의 fresh reference cloud 사이 CD를 평균. 3D dataset-level MMD와 다르다.
- Leakage: 전체 pooled 점 중 foreground 밖 비율.
- Checkerboard cell mass error: 유효 cell 내부 점만 조건화한 cell 질량과 uniform cell 질량 사이 TV. 유효 점이 없으면 undefined/null.
- Histogram JS: 64×64 [-1,1]² bin + 영역 밖 bin, natural-log divergence. Reference는 mask/cell 면적을 정확히 bin에 적분한 질량이다. SciPy JS distance를 제곱한다.
- 3D MMD-CD: reference별 generated까지의 최소 CD를 reference에 대해 평균. 여기서 MMD는 **Minimum Matching Distance**, kernel MMD가 아니다.
- 3D COV-CD: generated 각각이 고른 nearest reference의 서로 다른 ID 수 / reference 수.
- 3D 1-NNA-CD: self-match를 제외한 두 집합의 nearest-neighbor 분류 정확도. 균형 표본에서 0.5 부근을 기대하며 “무조건 낮을수록 좋다”가 아니다.

**근거:** [metrics.py](metrics.py), [eval_3d.py](eval_3d.py)의 `cd_metrics`.

**범위·조치:** EMD, 3D occupancy JSD는 현재 3D evaluator에 없다. 2D histogram JS를 3D JSD benchmark와 같은 지표라고 표기하지 않는다.

### 32. ★ 평가 표본 수와 통제

**상태: 확인됨 — 기본 설정 / 실제 run은 JSON 우선.** 현재 2D baseline YAML은 평가 64 clouds × N=256, 즉 16,384 points를 사용한다. YAML에서 override하지 않으면 helper 기본 batch는 training batch의 4배이므로 파일별로 확인해야 한다. CD용 reference는 새로 샘플링하고, silhouette 지표는 알려진 target mask/cell을 사용한다.

3D pilot 기본값은 generated=reference=32 shapes, generation batch=4, seed=1, split=val이다. Reference shape는 중복 없이 고르지만 각 shape의 점은 복원추출한다. NFE 사이에는 같은 noise/reference를 재사용한다. 이 32-shape pilot은 논문 표준 프로토콜이라고 선언하지 않는다.

**근거:** [experiment.py](experiment.py), [eval.py](eval.py), [eval_horse.py](eval_horse.py), [eval_3d.py](eval_3d.py).

**범위·조치:** 과거 기록은 평가 batch=256 등 현재 YAML과 다를 수 있다. 기록의 `total_points`/`evaluation_batch_size`를 우선한다. Coupling용 `ot.emd` 사용과 EMD 평가 수행은 별개다.

### 33. 외부 baseline 수치 비교

**상태: 추가 확인·실험 필요.** 재학습, 공개 checkpoint 재평가, 논문 수치 인용을 구분해야 한다. 현재 3D runner는 PSF 백본을 재사용한 first-stage FM 대조 실험이며 NSOT·PSF의 논문 전체 결과를 재현했다고 볼 수 없다.

**근거:** [psf_adapter.py](psf_adapter.py), [train_3d.py](train_3d.py)의 stage metadata; 관련 연구 목록 R.

**범위·조치:** Split/N/normalization/표본 수/metric 구현이 일치하는지 확인하기 전에는 외부 수치를 같은 순위 표에 넣지 않는다. “동일 백본 coupling 비교”와 “외부 방법 benchmark”를 분리한다.

## G. 계산 비용과 공정한 예산 비교

### 34. ★ 계산 위치와 시간 측정

**상태: 확인됨 — 구현.** FPS, 거리 행렬, centroid, index gather, randperm은 입력 tensor의 device에서 수행한다. Exact POT solve는 CPU float64 NumPy로 옮겨 수행한다. `exact_optimized`는 batch 전체 전송을 묶지만 CPU solve는 여전히 cloud별이다.

Coupling은 학습 loss 내부에 있으며 DataLoader worker에서 미리 계산하지 않는다. 3D loader는 resume 상태 추적을 위해 workers=0이다. 현재 2D training timer는 학습 loop 입·출구 CUDA synchronize를 포함하지만 model 초기화·최종 파일 저장은 제외한다. 3D는 loop 안의 이전 checkpoint I/O를 포함할 수 있어 타이머 범위를 동일하게 맞춰야 한다.

**근거:** [coupling.py](coupling.py), [train.py](train.py), [train_3d.py](train_3d.py), [benchmark_coupling.py](benchmark_coupling.py).

**범위·조치:** synthetic 3D coupling benchmark의 단계별 동기화 측정은 실제 overlapped training throughput과 다르다. 그것을 그대로 2D 학습 overhead로 가져오지 않는다.

### 35. Independent 대비 실제 추가 비용

**상태: 일부 과거 기록 확인 / 현재 optimized 구현의 정확한 overhead 미확인.** 과거 horse 3-seed 요약은 TG의 상당한 training overhead를 기록한다. 당시 metadata도 POT/network-simplex 기반 `pot_v1`임을 명시한다. 따라서 “당시 Hungarian이어서 느렸다”라고 추측하면 안 되며, 이 시간을 현재 `target_guided_exact_optimized`의 batched-transfer 성능으로 소급해서 써도 안 된다.

**근거:** 증거 목록 E의 training-time 표; [experiment.py](experiment.py)의 `training_seconds`.

**범위·조치:** 같은 코드 버전·microbatch·GPU에서 baseline 대비 `(T_TG/T_independent−1)×100%`를 측정한다. 별도로 coupling-only와 full update 시간을 구분한다. Preprocessing·cache·전송·I/O 제외 여부도 적는다.

### 36. 필요한 예산 비교

**상태: 추가 실험 필요.** 최소한 대표 설정에서 동일 optimizer-update 예산 비교와 동일 wall-clock 예산 비교가 필요하다. Inference는 quality–NFE뿐 아니라 quality–latency 또는 사전 정한 품질 도달 latency를 비교한다.

반복 사용 시 실용 이점은 대략 `추가 학습시간 < 생성 횟수 × 회당 절감 시간`이 성립하는 범위에서 논의할 수 있다. 다만 서로 다른 품질의 샘플 시간을 비교해서 이 식을 채우면 안 된다.

**근거:** 현재 timing fields와 질문 4·34·35.

**범위·조치:** 동일 품질에 도달하지 못한 조건은 “더 빠름” 대신 quality–cost trade-off로 보고한다.

## H. 수학적 정식화와 기전 분석

### 37. ★ 평균 coupling과 centroid 보조정리

**상태: 확인됨 — 고정 유한 집합에 대한 수학적 성질.** Target patch를 $P_k$, source에 할당된 index 집합을 $S_k$, 양쪽 크기를 $n_k>0$라 하자. 각 집합의 uniform empirical measure를 $\mu$로 쓰면, 내부 random bijection을 평균낸 **한 점 쌍의 coupling**은

$$
\bar\pi=\sum_{k=1}^K\frac{n_k}{N}\,
\mu_{S_k}\otimes\mu_{P_k}.
$$

Index 행렬에서는 같은 block에 속한 i,j에 대해 $\bar P_{ij}=1/(Nn_k)$, 나머지는 0이다. 반면 실제 한 번 샘플링한 coupling은 $P^\sigma_{ij}=N^{-1}\mathbf1[j=\sigma(i)]$인 permutation plan이다. 평균 행렬은 rank≤K인 factorization을 갖지만, 실현된 index permutation 행렬의 rank는 N이다. 이 평균 식은 cloud 전체 점들의 독립성을 뜻하지 않는다.

Centroid $c_k=n_k^{-1}\sum_{j\in P_k}y_j$에 대해

$$
\mathbb E_{\sigma}\frac1N\sum_i\|x_i-y_{\sigma(i)}\|^2
=\frac1N\sum_i\|x_i-c_{A(i)}\|^2
+\underbrace{\frac1N\sum_k\sum_{j\in P_k}\|y_j-c_k\|^2}_{C(P)}.
$$

고정 target partition·capacity 아래 $C(P)$는 source assignment와 무관하다. 따라서 exact centroid assignment는 **이 제한된 coupling family 안에서** 기대 제곱 이동 비용을 최소화한다. 증명은 $y_j-c_k$의 patch 평균이 0이어서 교차항이 사라지는 것으로 충분하다.

**근거:** [coupling.py](coupling.py)의 centroid 비용과 패치 내부 random bijection; 위 교차항 소거에 따른 항등식.

**범위·조치:** Target partition까지 바꾸면 $C(P)$도 바뀐다. 전역 pointwise OT, 평균장 근사 오차, 최종 생성 품질의 최적성을 뜻하지 않는다. Factored/low-rank coupling 자체가 새 개념이라는 주장도 피한다. 여기의 상수항은 보조정리의 일부일 뿐, WCSS를 본 방법의 새 최적화 목표로 채택했다는 뜻이 아니다.

### 38. 우선 검증할 기전 가설

**상태: 현재 가설.** 우선 질문은 “TG가 정한 경로의 marginal velocity를 고정된 유한 모델과 적은 Euler step이 더 잘 실현하는가?”이다. 그 답을 이동 비용이나 최종 training MSE 하나로 대신하지 않는다.

관찰을 다음처럼 분리한다: (a) coupling의 기대 이동 비용, (b) held-out FM residual, (c) learned trajectory의 곡률·적분 민감도, (d) 최종 분포 지표. Conditional supervision 평균은 FM의 정상적인 학습 대상이다. 분산이 크다는 사실만으로 blur가 필연이라고 결론내리지 않는다.

**근거:** [diagnose.py](diagnose.py), [CFM 원논문](https://arxiv.org/html/2302.00482v4)의 marginal field/회귀 목적 관계.

**범위·조치:** 기존 `fm_errors`는 label에 대한 residual이지 Bayes 평균장 근사 오차만을 분리한 값이 아니다. Raw kNN velocity 차이를 빼서 근사 오차를 만들지 않는다. 관측 상관과 인과적 설명을 구분한다.

### 39. t=0 분해

**상태: 현재 가설·진단 설계 / 전용 추정기는 미구현.** 본 방법의 확정된 분석 결과로 넣지 않는다. 조건화, MC 보정 및 미구현 항목은 **부록 P.3**에서 별도로 답한다.

**근거·범위·조치:** 현재 `diagnose.py`는 t=0의 raw FM residual을 포함하지만 조건부 평균 추정기는 아니다. 별도 구현·검증 없이 “t=0 decomposition 완료”라고 쓰지 않는다.

### 40. WCSS와 between 항

**상태: partition 진단 도구 구현 / 생성 품질과의 관계는 미검증.** [audit_k.py](audit_k.py)는 같은 target cloud에서 K별 TG partition의 CH·DB·WCSS를 계산한다. Raw WCSS, WCSS/N, WCSS/전체 scatter를 구분해 저장한다. 실행 방법은 [K_SELECTION.md](K_SELECTION.md)에 있다. 정의·정규화·다른 between 항과의 구분은 **부록 P.1–P.2**에 편성한다.

**근거·범위·조치:** 고정 partition의 내부 random pairing에서 직접 도출되는 covariance 항등식이다. 모델 평균장 오차·고주파 보존의 증명이 아니다.

### 41. 실제 생성 trajectory와 적분 오차

**상태: 확인됨 — 진단 도구 / 각 결과의 수렴 여부는 별도 검증.** `diagnose.py`는 학습된 모델을 실제로 적분한다. 경로 길이 / 시작–끝 변위는 각 cloud에서 점들의 길이를 합한 뒤 변위 합으로 나눈다. 점별 비율의 평균이 아니다. 변위 합이 0인 cloud는 제외하고 개수를 기록한다.

같은 noise에서 reference Euler S와 2S endpoint MSE를 비교하고, NFE별 endpoint가 reference에서 얼마나 다른지 기록한다. 이는 수치 수렴 진단이지 reference가 target 분포의 정답이라는 뜻이 아니다.

**근거:** [diagnose.py](diagnose.py)의 `rollout`, `reference_check_mse`; [audit_target_patches.py](audit_target_patches.py)의 chord는 별개의 target–target 선분이다.

**범위·조치:** Pair의 선형 보간, target chord, learned ODE trajectory를 figure에서 구분한다. 배경을 지나간다는 사실만으로 최종 생성 실패라고 판정하지 않는다.

## I. 메인 실험·ablation·figure

### 42. ★ 메인 표와 핵심 figure

**상태: 실험 구성 제안.** 핵심 질문은 “같은 모델·학습 예산에서 coupling만 바꾸었을 때 quality–NFE/latency trade-off가 어떻게 달라지는가?”로 둔다.

2D는 dataset별로 Independent/TG/가까운 OT 대안을 행에, NFE와 Chamfer·leakage·histogram JS·checkerboard mass error를 열에 둔다. Training time도 함께 제시한다. 3D는 별도 표에서 같은 category·split·정규화·표본 수를 맞춘 CD 기반 분포 지표를 보고한다. Pilot과 정식 benchmark 표를 혼합하지 않는다.

**근거:** 현재 평가·summary 도구와 증거 목록 E.

**범위·조치:** NFE–품질 곡선에 high-NFE 손실을 포함한다. 미구현 진단 후보를 검증된 결과 figure로 제시하지 않는다.

### 43. 최소 ablation

**상태: 일부 구현 / 일부 추가 실험 필요.** 최소 구성은 (1) Independent vs TG, (2) 같은 TG partition/source 할당에서 random vs local exact pairing, (3) K=1 포함 K sweep, (4) 비용이 허용되는 N에서 global inner OT 및 가장 가까운 선행 방법 비교다.

**근거:** 질문 23·28; 현재 `pair_within_regions(local='exact')` primitive는 있으나 일반 TG-local method/설정은 아직 없다. Strict-local 결과만으로 (2)를 대체하지 않는다.

**범위·조치:** K=1/N의 unit test와 실제 장기 학습 ablation을 구분한다. 추가할 비교는 한 번에 한 요인만 바꾸고 별도 method·metadata로 명시한다. 이 문서 작성 과정에서는 새 학습 코드를 추가하지 않았다.

### 44. Partition·capacity·soft boundary 실험

**상태: 방법 차이 확인 / 원인 분리는 추가 분석 필요.** FPS-balanced vs nearest는 patch membership뿐 아니라 capacity 목록도 달라지므로 순수 연결성 ablation이 아니다. Sinkhorn+rounding은 최종 대응이 hard bijection이며, 계속 soft한 coupling을 모델에 주는 방법과 다르다.

과거 capacity·separation·source 무작위화 실험은 현재 본 방법으로 채택되지 않은 탐색 이력으로 분리한다.

**근거:** [coupling.py](coupling.py)의 분기; 증거 목록 E의 과거 audit/무작위화 기록.

**범위·조치:** Checkerboard의 각 cell 표본 수는 random이므로, K=8이라도 모든 패치를 32점으로 고정하면 여러 cell을 섞는 일이 불가피할 수 있다. 혼합 비율 자체를 곧바로 구현 오류나 생성 실패의 증거로 읽지 않는다.

### 45. 해상도 확장과 sharpness

**상태: 추가 실험 필요.** N 증가와 K 증가를 구분해야 한다. K 고정은 patch당 점 수가 늘고, N/K 고정은 patch 수가 늘기 때문에 서로 다른 coupling granularity 실험이다.

현재 8192점의 생성 품질·추가 3D category·전체 category 학습이 검증되었다고 볼 수 없다. Synthetic 3D coupling benchmark의 큰 N 실행은 대규모 생성 모델 검증을 대신하지 않는다.

**근거:** [benchmark_coupling.py](benchmark_coupling.py), 실제 설정 파일과 증거 목록 E.

**범위·조치:** “sharpness 보존”을 핵심 주장으로 삼으려면 지표와 영역 정의를 별도로 확정한다. 현 2D leakage/JS는 관련된 분포 지표이지만 고주파 spectrum 보존을 직접 측정하지 않는다. 본 주장에 필요하지 않으면 대규모 확장은 후속 실험으로 둔다.

### 46. 정성 결과와 2D 실험의 역할

**상태: 구현 확인 / 논문용 figure 선택 필요.** 2D도 set-valued $X\in\mathbb R^{N\times2}$ 샘플의 inner coupling 실험이다. 점 하나를 독립적인 데이터 sample로 보는 outer coupling 실험과 동일하지 않다.

좋은 사례와 실패 사례를 같은 generation seed·같은 표시 범위로 제시한다. Checkerboard density 그림은 Gaussian smoothing과 PowerNorm을 사용하므로 시각적 sharpness만으로 결론내리지 않는다. 수치 지표는 원 생성 점에서 계산한다.

**근거:** [eval.py](eval.py)의 `render_density`, [eval_horse.py](eval_horse.py)의 `render_comparison`, [metrics.py](metrics.py).

**범위·조치:** Figure의 임의 sample 선별을 피하고 표시 규칙을 기록한다. 중간 경로의 support 이탈과 endpoint leakage를 분리해 설명한다.

## J. 선행연구·한계·실행 계획

### 47. ★ 관련 연구의 정체와 비교 수준

**상태: 일부 원문 확인 / 전체 동일 프로토콜 재현은 미확인.** 관련 연구 목록 R에 정확한 출처와 확인 수준을 분리한다. 공통 비교 축은 sample unit, inner/outer, partition/plan 구조, 목적함수, randomness, marginal/bijection 보존 조건, offline/online/inference 비용이다.

**범위·조치:** 약어 이름만으로 같은 방법으로 취급하지 않는다. 논문/공개 code가 있어도 현재 저장소에 baseline으로 연결·평가했다는 뜻은 아니다. 특히 MFM-point의 constrained clustering, NSOT의 hybrid coupling, 기존 low-rank/factored coupling과 겹치는 구성요소를 선행 개념으로 인정한다.

### 48. ★ 선행연구 이후 남는 TG 차별점

**상태: 차별화 후보, novelty 확정 아님.** 현재 구체적인 차이는 **각 sampled target cloud가 정하는 균형 patch, centroid에 대한 exact capacity source assignment, 패치 내부 random bijection을 유지하면서 기존 FM·inference를 바꾸지 않는 조합**이다.

“OT와 independent를 섞는다”, “평균 coupling이 low-rank다”, “balanced clustering을 쓴다”만으로 신규성을 주장하지 않는다. Target partition을 사용하는 것과, 그 partition으로 단일-scale source 대응을 제한하는 것은 구분해야 한다.

**근거:** 질문 12·37 및 관련 연구 R.

**범위·조치:** 가장 가까운 방법과 차이 하나씩을 실제 통제 실험에 연결한다. 기여 후보의 문장은 작성할 수 있지만 “최초”, “기존에 없었다”는 결론은 선행연구 조사 완료 전 보류한다.

### 49. 명시할 한계

**상태: 구현상 한계와 미검증 범위를 구분.**

- 정확히 보장하는 것은 입력 set/bijection/capacity와 제한된 family 내 기대 비용 최적화다. 최종 생성 품질·고주파·연결성·population OT 최적성을 보장하지 않는다.
- FPS·exact assignment는 동률과 floating-point 조건에 민감할 수 있다. Exact CPU solve와 전송은 training overhead를 만든다.
- 현재 비교는 제한된 N·K·데이터에 대한 결과이며, 3D pilot 결과를 large-scale 일반성으로 확대하지 않는다.
- Target support를 따라가는 geodesic partition이 아니므로 다른 얇은 부위를 한 patch로 묶을 수 있다.
- 공통 강체 변환에서의 거리 기반 coupling 성질과 backbone의 rotation equivariance는 별개다. Ties/정밀도/coordinate normalization까지 무조건 불변이라고 하지 않는다.
- High-NFE 손실이나 training overhead가 관측되면 이를 trade-off로 명시한다.

**근거:** 알고리즘·백본·solver 코드, 증거 목록 E.

**조치:** 수학적으로 보장하지 않는 항목, 구현 제한, 아직 실행하지 않은 항목을 limitations에서 각각 구분한다.

### 50. ★ 확보된 것과 제출까지의 작업

**상태: 구현 및 일부 평가 기록 확보 / 제출 일정·잔여 예산 미확인.**

| 구분 | 현재 답변 |
|---|---|
| 바로 작성 가능 | 기존 TG Methods, 점 대응 의사코드, centroid 보조정리, 입력 set 보존, 현재 config에 따른 Setup |
| 기존 결과 재분석 | 과거 horse seed-summary, 제한된 source 무작위화, patch audit를 각각 당시 코드/프로토콜과 연결 |
| 필수 추가 작업 | 원 checkpoint·run manifest 확보, 같은 버전의 핵심 비교·반복·비용 측정, 필요한 local-pairing ablation |
| 3D | 연결 코드·설정과 사용자가 보낸 실행 화면은 있으나 현재 접근 가능한 paper-ready 성능 표는 없음; pilot와 benchmark 구분 |
| 선택 실험 | WCSS·CH·DB partition 진단(구현), t=0 조건부 분해(미구현), 추가 geometry, sharpness, 8192점·추가 category 확장 |
| 미확인 자원 | 제출 venue/year/deadline, 사용 가능한 GPU 대수·종류, 현재 진행 중인 run과 남은 GPU-hours |

**범위·조치:** 모든 선택 실험이 끝나야 초안을 쓰는 것은 아니다. 먼저 Methods/Setup/한계를 쓰고, Abstract·성능 주장·novelty 강도는 확인된 결과 범위에 맞춰 확정한다.

## E. 접근 가능한 과거 결과의 증거 목록

이 절의 표는 과거 평가 기록의 정리다. 원 checkpoint가 이 PC에 없으면 JSON에 기록된 hash와 검증 flag를 그대로 표시하되, 이번 문서 작성 중 weight를 다시 로드해 검증했다고 쓰지 않는다.

### E.1. 자료의 위치와 확인 수준

| 자료 | 실제 읽은 범위 | 주의 |
|---|---|---|
| [Horse seed-summary](C:/Users/alstj/Downloads/analysis_results/analysis_results/seed_summary_20260914T153913Z_01b4ed75/summary.json) | Independent, TG K=8, TG K=16; N=256; seeds 0/1/2; 8 NFEs | 3개 완전한 그룹. 요청한 seeds가 부족한 항목 112개는 집계에서 제외됨 |
| [Randomize 평가 폴더](C:/Users/alstj/Downloads/randomize) | 읽을 수 있는 JSON 48개 = 6 checkpoints × 8 NFEs; checkerboard/horse | 모두 training seed=0, eval seed=1. 서로 다른 날짜의 메인 run과 합쳐 seed 평균을 만들면 안 됨 |
| [Checkerboard patch audit](C:/Users/alstj/Downloads/patch_audit.json), [Horse patch audit](<C:/Users/alstj/Downloads/patch_audit (1).json>) | 각각 fresh cloud 16개; balanced vs same-anchor nearest | 모델·source noise·checkpoint를 사용하지 않은 target-only audit |
| [이전 diagnostics 폴더](C:/Users/alstj/Downloads/analysis_results/analysis_results) | 23개 diagnostic JSON 존재; 아래 E.4의 checkerboard 3개를 구체적으로 대조 | 생성 경로 진단이며 새로운 학습 seed 반복이 아님 |
| Capacity / separation / Mahalanobis | 이전 대화의 파일명은 남아 있으나 당시 Downloads의 해당 디렉터리는 현재 접근 불가 | 새 문서에서 수치·순위를 복원하거나 꾸며 넣지 않음 |

Randomize JSON은 `training_config_verified=true`를 기록한다. 그러나 원 checkpoint 경로들은 다른 PC를 가리키고 이 PC에서는 접근되지 않는다. **기록에 적힌 검증 상태**와 **이번 작업에서 원 weight를 재검증한 것**을 구분한다. Seed-summary에 적힌 원 평가 파일들도 다른 PC 경로이므로 아래 표의 직접 출처는 접근 가능한 summary JSON이다.

### E.2. Horse의 과거 3-seed 결과

N=256, training seeds 0/1/2, 동일 horse 백본·10,000 updates 설정. 아래는 **학습 seed 간 mean ± sample SD**, confidence interval이 아니다. 읽기 쉽도록 전체 8개 NFE 중 1/4/8/32/128을 발췌했고, 전체 값은 위 summary에 있다. 지표 정의는 질문 31을 따른다.

| Method | NFE | Chamfer ↓ | Leakage ↓ | Histogram JS ↓ |
|---|---:|---:|---:|---:|
| Independent | 1 | 0.265483 ± 0.023670 | 0.000102 ± 0.000176 | 0.638552 ± 0.022656 |
| Independent | 4 | 0.013238 ± 0.000534 | 0.109355 ± 0.004478 | 0.209112 ± 0.009048 |
| Independent | 8 | 0.005542 ± 0.000161 | 0.101420 ± 0.008602 | 0.121271 ± 0.008904 |
| Independent | 32 | 0.003508 ± 0.000064 | 0.061198 ± 0.020061 | 0.041054 ± 0.008168 |
| Independent | 128 | 0.003467 ± 0.000069 | 0.073344 ± 0.017955 | 0.031592 ± 0.003860 |
| TG K=8 | 1 | 0.016231 ± 0.001613 | 0.136658 ± 0.054321 | 0.333066 ± 0.021872 |
| TG K=8 | 4 | 0.005640 ± 0.000159 | 0.083252 ± 0.008089 | 0.180619 ± 0.010799 |
| TG K=8 | 8 | 0.003762 ± 0.000091 | 0.061239 ± 0.011032 | 0.088523 ± 0.005538 |
| TG K=8 | 32 | 0.003410 ± 0.000053 | 0.050334 ± 0.015447 | 0.030310 ± 0.004109 |
| TG K=8 | 128 | 0.003449 ± 0.000039 | 0.061829 ± 0.015052 | 0.027332 ± 0.004937 |
| TG K=16 | 1 | 0.009790 ± 0.000891 | 0.166423 ± 0.007905 | 0.237971 ± 0.017386 |
| TG K=16 | 4 | 0.004319 ± 0.000142 | 0.089254 ± 0.010222 | 0.131398 ± 0.013349 |
| TG K=16 | 8 | 0.003467 ± 0.000112 | 0.059855 ± 0.014257 | 0.068366 ± 0.010409 |
| TG K=16 | 32 | 0.003413 ± 0.000085 | 0.063924 ± 0.016706 | 0.029200 ± 0.004209 |
| TG K=16 | 128 | 0.003479 ± 0.000090 | 0.078003 ± 0.017799 | 0.029342 ± 0.004463 |

**이 표에서 가능한 해석:** 이 horse 설정에서는 TG의 low-NFE Chamfer/JS 개선이 관측된다. 그렇지만 모든 지표에서 우월하지 않다. Independent는 NFE=1 leakage가 거의 0인데 CD/JS는 매우 나쁘므로, **leakage만으로 분포 복원을 판정할 수 없다**. NFE=128에서 TG K=16의 CD·leakage 평균은 Independent보다 나쁘고 JS 평균은 더 낮다. TG K=8은 세 지표 평균이 더 낮지만, 이 표만으로 통계적 유의성이나 동등성을 주장하지 않는다.

| Method | Training seconds, mean ± SD | Independent 대비 평균 시간 비율 | Checkpoint hash prefix — seed 0 / 1 / 2 |
|---|---:|---:|---|
| Independent | 297.609 ± 20.305 | 1.00× | `84ddb80e0a79` / `8608056032ca` / `f4fb740c841b` |
| TG K=8 | 6598.200 ± 105.657 | 약 22.17× | `2e563999c826` / `ced2c7b6eec4` / `11cfe26d3a18` |
| TG K=16 | 10329.675 ± 3754.875 | 약 34.71× | `7fcad1790c67` / `cafc3bae0009` / `6df569418143` |

기록된 공통 환경은 RTX 6000 Ada, PyTorch 2.5.1+cu121, POT 0.9.7, Python 3.10.21이다. TG metadata는 `implementation=pot_v1`, `exact_solver=POT/network_simplex`, balanced target / exact source / random local이다. 이 시간은 **과거 구현의 측정값**이며 현재 optimized 구현의 overhead나 GPU-hours로 대체할 수 없다.

### E.3. 제한된 source 할당 무작위화 — 채택하지 않은 과거 탐색

이 실험은 현재 TG의 정의에 포함하지 않는다. 두 budget은 exact source assignment 뒤 제한된 random swap을 추가한 변형이다. 당시 `relative_budget=0.01/0.05`, `proposal_sweeps=4`, `implementation=pot_tg_randomized_v1`이며 현재 활성 registry에는 없다.

모두 N=256, K=8, training seed=0, eval seed=1, 평가 64 clouds. 아래 cell은 **Chamfer / Leakage / JS**이고, 전체 NFE 중 1/8/128을 발췌했다.

| Dataset / method | NFE=1 | NFE=8 | NFE=128 |
|---|---|---|---|
| Checkerboard TG | 0.021507 / 0.056824 / 0.409487 | 0.006477 / 0.043640 / 0.087857 | 0.006383 / 0.072876 / 0.051154 |
| Checkerboard budget .01 | 0.019956 / 0.065063 / 0.385271 | 0.006499 / 0.040100 / 0.088877 | 0.006446 / 0.074890 / 0.052963 |
| Checkerboard budget .05 | 0.020517 / 0.087463 / 0.361719 | 0.006446 / 0.041382 / 0.085565 | 0.006366 / 0.072327 / 0.049354 |
| Horse TG | 0.016602 / 0.143127 / 0.341596 | 0.003860 / 0.074219 / 0.093846 | 0.003487 / 0.078003 / 0.032451 |
| Horse budget .01 | 0.016963 / 0.136108 / 0.331734 | 0.003963 / 0.080627 / 0.091913 | 0.003623 / 0.095032 / 0.040843 |
| Horse budget .05 | 0.017256 / 0.119751 / 0.306583 | 0.003887 / 0.078552 / 0.086170 | 0.003594 / 0.085083 / 0.035162 |

| Dataset / method | Training seconds | Checkpoint prefix | 대표 NFE=1 원본 |
|---|---:|---|---|
| Checkerboard TG | 1844.1693 | `fed8703eae36` | [평가 JSON](C:/Users/alstj/Downloads/randomize/nfe_001_20260922T111058Z_b918b5df.json) |
| Checkerboard .01 | 1974.2234 | `1850707fa059` | [평가 JSON](C:/Users/alstj/Downloads/randomize/nfe_001_20260922T111222Z_5ad592a1.json) |
| Checkerboard .05 | 1980.0007 | `242995bf9f13` | [평가 JSON](C:/Users/alstj/Downloads/randomize/nfe_001_20260922T111251Z_4c0db405.json) |
| Horse TG | 2022.7754 | `49715177909c` | [평가 JSON](C:/Users/alstj/Downloads/randomize/nfe_001_20260922T111519Z_abeedd82.json) |
| Horse .01 | 2125.7552 | `ed87d00990b6` | [평가 JSON](C:/Users/alstj/Downloads/randomize/nfe_001_20260922T111652Z_194bff8a.json) |
| Horse .05 | 2140.1523 | `cab71b21b530` | [평가 JSON](C:/Users/alstj/Downloads/randomize/nfe_001_20260922T111724Z_94952e81.json) |

이 기록에서는 일관된 우위가 없다. Horse의 큰 NFE에서는 원 TG보다 손해가 나타난다. **“무작위성 자체는 해롭다”로 일반화하면 안 된다.** 바꾼 것은 coarse source 할당이며, 원 TG의 fine random bijection은 비교 조건 모두에 유지됐다. 단일 training seed이므로 SD·유의성을 제시할 수 없다.

### E.4. 기전·geometry 진단 — 성능 원인의 증명 아님

Checkerboard의 아래 3개 진단은 training seed=0, N=256, diagnostic seed=2026, 128 fresh clouds, Euler reference=128/check=256이다.

| Method | 경로 길이 / 순변위, mean ± cloud SD | NFE=1 endpoint MSE vs 자체 Euler128 | NFE=8 MSE | Euler128–256 MSE |
|---|---:|---:|---:|---:|
| Independent | 1.552766 ± 0.034841 | 0.322586 | 0.00666272 | 3.18225e−5 |
| TG K=8 | 1.058914 ± 0.007714 | 0.0191255 | 0.000824561 | 1.41673e−6 |
| Global OT | 1.006205 ± 0.002126 | 0.00370367 | 0.0000988582 | 1.54716e−7 |

출처: [Independent diagnostics](C:/Users/alstj/Downloads/analysis_results/analysis_results/checkerboard/independent_diagnostic_20260914T154044Z_1759934a/diagnostics.json), [TG diagnostics](C:/Users/alstj/Downloads/analysis_results/analysis_results/checkerboard/target_guided_diagnostic_20260914T154058Z_b4340034/diagnostics.json), [Global OT diagnostics](C:/Users/alstj/Downloads/analysis_results/analysis_results/checkerboard/global_hungarian_diagnostic_20260914T154038Z_74db4d05/diagnostics.json).

이것은 학습된 각 모델의 **자체 finite-step endpoint와의 수치적 차이**다. Ground-truth endpoint 오차나 조건부 분산이 아니다. Global OT가 이 표의 straightness/수치 안정성에서 더 좋다고 해서 최종 분포 품질도 반드시 더 좋다는 결론은 나오지 않는다. 원 checkpoint를 다시 읽지는 않았고, 과거 diagnostics에는 명시적인 `training_config_verified` 필드가 없다는 한계도 있다.

| Target-only patch audit | Balanced chord exit fraction | Same-anchor nearest |
|---|---:|---:|
| Checkerboard | 0.209991 ± 0.066714 | 0.162352 ± 0.044250 |
| Horse | 0.236420 ± 0.029111 | 0.191828 ± 0.032169 |

오차 막대는 16 clouds 사이 SD다. 패치 내부 target–target 선분이 배경을 통과하는 비율이며 **source–target FM 경로의 이탈률이 아니다**. Nearest의 수치가 낮아도, 이 비교는 matched-capacity 생성 실험이 아니므로 nearest로 학습하면 더 잘 생성한다는 뜻은 아니다. 이 자료를 WCSS의 성능 예측력을 입증하는 증거로 사용하지 않는다.

## R. 관련 연구 — 출처와 비교 범위

아래 문헌은 구성요소·비교 수준을 구분하기 위한 자료다. 외부 논문의 성능을 TG 성능에 합산하거나, 논문의 정리를 TG에 조건 없이 이식하지 않는다.

### R.1. 정확한 문헌과 공개 구현 확인 수준

| 약칭 | 확인한 문헌 / 버전 | 공식 구현 확인 수준 |
|---|---|---|
| NSOT | [Not-So-Optimal Transport Flows for 3D Point Cloud Generation](https://arxiv.org/abs/2502.12456v1), ICLR 2025 | [공식 NVIDIA project](https://research.nvidia.com/labs/genair/not-so-ot-flow/) 확인; 이번 조사에서 repository는 확인하지 못함 |
| Equivariant OT-FM | [Equivariant flow matching](https://arxiv.org/abs/2306.15030v2), Klein–Krämer–Noé | 원문이 [bgflow](https://github.com/noegroup/bgflow) 통합 예정이라고 설명; 현재 해당 구현의 완전성 미확인 |
| OT-CFM | [Improving and generalizing flow-based generative models with minibatch optimal transport](https://arxiv.org/abs/2302.00482v4), TMLR | 원문이 [conditional-flow-matching](https://github.com/atong01/conditional-flow-matching)을 연결 |
| Multisample FM | [Multisample Flow Matching: Straightening Flows with Minibatch Couplings](https://arxiv.org/abs/2304.14772v2) | 이번 조사에서 공식 repository 미확인 |
| SD-FM | [Flow Matching with Semidiscrete Couplings](https://arxiv.org/abs/2509.25519v2), ICLR 2026 | 원문이 [ott-jax/ott](https://github.com/ott-jax/ott)을 연결 |
| MAC | [Beyond Optimal Transport: Model-Aligned Coupling for Flow Matching](https://arxiv.org/abs/2505.23346v1) | [공식 project](https://yexionglin.github.io/mac/)의 code 표시가 확인 시점 “Coming Soon” |
| MFM-point | [MFM-point: Multi-scale Flow Matching for Point Cloud Generation](https://arxiv.org/abs/2511.20041v1) | 이번 조사에서 공식 repository 미확인 |
| BSP-OT | BSP-OT: Sparse transport plans between discrete measures in loglinear time, SIGGRAPH Asia 2025 | [공식 repository](https://github.com/baptiste-genest/BSP-OT)와 [POT API](https://pythonot.github.io/gen_modules/ot.bsp.html) 확인; arXiv ID 미확인 |
| Factored coupling | [Statistical Optimal Transport via Factored Couplings](https://proceedings.mlr.press/v89/forrow19a.html), AISTATS 2019 | 논문 확인; 저자 repository는 이번 조사에서 미확인 |
| Low-rank coupling | [Low-Rank Sinkhorn Factorization](https://proceedings.mlr.press/v139/scetbon21a.html), ICML 2021 | 논문 확인; 저자 repository는 이번 조사에서 미확인 |

“미확인”은 code가 없다는 뜻이 아니다. 위 버전은 읽은 원문의 버전이며 최신 버전임을 모두 보장하지 않는다. Equivariant OT라는 이름의 모든 방법을 Klein 등의 해당 논문 하나와 동일시하지 않는다. **위 방법들이 현재 TG 저장소의 baseline으로 구현·평가됐다는 뜻도 아니다.**

### R.2. TG와 비교할 핵심 축

여기서 B는 cloud minibatch 수, N은 cloud당 점 수, K는 TG patch 수다. SD-FM의 dataset 크기는 별도로 D라고 쓴다.

| 방법 | Sample / coupling 수준 | 대응 구성·목적 | 무작위성·보존 조건 | 비용 위치와 TG와의 차이 |
|---|---|---|---|---|
| 기존 TG | 독립 outer cloud 쌍의 inner matching | FPS target balanced patch, centroid source exact assignment, hard block | 매번 patchwise random bijection; 입력 point set 정확히 보존 | online N×K cost 및 CPU exact solve. **solver 시간 전체가 O(NK)라는 뜻은 아님**. FM·inference 불변 |
| NSOT | shape별 noise–surface superset point coupling | offline dense superset OT 후 online paired subsampling; hybrid noise | finite-superset empirical coupling; population marginal 정리는 superset 크기에 대한 점근적 결과 | 비싼 작업을 offline으로 이동. 큰 superset에는 근사법도 사용. TG target patch와 다름 |
| Equivariant OT-FM | outer minibatch OT 및 각 candidate 쌍의 inner symmetry alignment | permutation Hungarian과 Kabsch rotation으로 alignment 근사 | bijection 외에 group action도 사용; 해당 대칭 분포 가정 필요 | inner alignment에 outer B² candidate 비용이 더해짐; patchwise random 방식 아님 |
| OT-CFM / Multisample FM | 주로 dataset-sample의 outer coupling | sample-space 제곱거리의 minibatch OT/entropic OT | 계획된 batch marginal 보존; finite batch가 population OT인 것은 아님 | batch coupling 계산. 이를 pointwise inner OT와 같은 baseline으로 표기하면 안 됨 |
| SD-FM | 연속 noise → 유한 dataset의 sample | offline dual potential, online 검색/샘플링 | noise marginal은 fitted potential에 관계없이 보존; target marginal은 수렴 조건 필요; per-batch bijection 아님 | offline fitting 및 online dataset search. TG cloud 내부 partition과 다른 수준 |
| MAC | sample-level 후보 coupling 선택 | 현재 모델의 endpoint 예측 오차, 낮은 오차 후보 top-k | 무작위 후보 후 선택; 이상적 marginal 제약이 구현된 subset 선택의 정확한 marginal을 자동 보장하지 않음 | 추가 모델 평가 및 **가중 보조 loss**. TG의 loss 불변 조건과 다름 |
| MFM-point | cloud 내부 다중 scale 구성 | FPS 초기화 equal-size constrained k-means와 반복 centroid 갱신 | down/up sampling·단계별 Gaussian alignment; TG patch 내부 random bijection이 아님 | offline clustering과 scale별 flow, coarse-to-fine inference. **반복 balanced clustering은 이미 선행 구성요소** |
| BSP-OT | 두 discrete point set의 직접 matching solver | joint binary-space partition, random proposals, cost-reducing merge | 무작위 후보를 사용하며 최종 bijection 유지 | 정해진 설정에서 loglinear 시간 설계; TG의 hard-block product coupling과 다름 |
| Factored / low-rank OT | 일반 discrete measure; inner/outer FM으로 한정되지 않음 | 중간 mixture factor를 통한 낮은 nonnegative-rank plan 최적화, 일반적으로 soft | 주어진 plan marginal 제약; pair 샘플링이 cloud bijection을 자동 구성하지는 않음 | factorized storage/최적화. TG의 기대 plan과 구조적 관련이 있으나 같은 알고리즘은 아님 |

세부 근거: NSOT [`3.3–3.4](https://arxiv.org/html/2502.12456v1), Equivariant FM [`4·`8](https://arxiv.org/html/2306.15030v2), Multisample FM [Appendix A](https://arxiv.org/html/2304.14772v2), SD-FM [`3–4](https://arxiv.org/html/2509.25519v2), MAC [`3·Eq.7](https://arxiv.org/html/2505.23346v1), MFM-point [`3.1·Algorithm 3](https://arxiv.org/html/2511.20041v1), [BSP-OT API](https://pythonot.github.io/gen_modules/ot.bsp.html), [Low-rank 원문](https://proceedings.mlr.press/v139/scetbon21a/scetbon21a.pdf).

### R.3. 이 저장소에 대한 수학적 비교 해석

다음은 외부 논문이 TG에 대해 주장한 결과가 아니라 **질문 37의 TG 구성에서 직접 도출한 관계**다. 고정 patch의 평균 plan은

$$
\bar P=Q\,\operatorname{diag}(1/g)\,R^\top,\qquad
Q_{ik}=\frac{\mathbf1[i\in S_k]}{N},\quad
R_{jk}=\frac{\mathbf1[j\in P_k]}{N},\quad g_k=\frac{n_k}{N}.
$$

따라서 TG의 **평균** plan은 nonnegative rank≤K인 hard-block factored coupling의 특수한 형태다. 이것은 low-rank 개념 자체의 신규성을 뜻하지 않는다. 실제 매 iteration의 permutation plan을 rank-K라고 부르면 틀리다.

현재 가장 안전한 차별화 문장은 **“기존 point-cloud FM의 모델·loss·inference를 유지하며, target-defined capacity-constrained hard blocks 안에 random bijection을 남기는 online inner coupling”**이다. 이것은 구체적인 방법 설명이지, 조합의 최초성이나 성능 우월성의 증명이 아니다.

## 최종 요약 1. 확정된 방법 설명

TG는 독립적으로 생성한 Gaussian noise cloud와 target point cloud 사이의 training-time point correspondence를 구성한다. 먼저 target cloud에서 결정적 FPS anchor를 선택하고 거의 균등한 정수 capacity 아래 exact balanced assignment로 target patch를 만든다. 각 patch의 centroid를 계산한 후 같은 capacity를 만족하도록 source 점을 centroid에 exact assignment하고, 대응하는 patch 내부에서는 uniform random bijection을 새로 샘플링한다. 입력 점 집합과 source 좌표는 보존되며, 재배열된 target과 원 source의 선형 보간 및 velocity MSE로 기존 모델을 학습한다. 모델 구조와 inference Euler 절차는 coupling 때문에 바뀌지 않는다.

## 최종 요약 2. 주장–근거 대응표

| 후보 주장 | 현재 근거 | 제한·반대 결과 | 안전한 표현 |
|---|---|---|---|
| Training coupling만 바꾼다 | 실제 loss/model/inference 경로 | 데이터셋 사이 백본은 다름 | 같은 실험 조건 내 backbone·FM·sampler를 유지한다 |
| 입력 set·capacity를 보존한다 | exact assignment와 random bijection, 테스트 | ordered target law는 별도 조건 필요 | 양쪽 유한 point set을 그대로 보존한다 |
| Centroid assignment의 수학적 정당성 | 질문 37의 기대 제곱거리 분해 | partition 고정 조건; 성능 보장 아님 | 제한된 coupling family에서 기대 비용을 최소화한다 |
| Low-NFE 품질 개선 | 일부 과거 2D 평가 기록 | 3D 일반화·모든 지표는 미확정 | 확인한 dataset/NFE/지표에 한정해 개선을 보고한다 |
| High-NFE에서도 동등 | 아직 동등성 검증 없음 | 지표별 trade-off 가능 | high-NFE 결과와 불확실성을 함께 보고한다 |
| 효율적이다 | inference에 coupling 추가 없음 | 과거 training overhead 큼 | quality–sampling-cost trade-off를 평가한다 |
| 고주파를 보존하는 원리를 입증 | 현재 없음 | geometry/variance proxy만으로 불충분 | 세부 구조 관련 경험적 결과 또는 가설로 제한한다 |

## 최종 요약 3. 최소 제출 실험 구성

1. **메인 동일 조건 비교:** Independent, 기존 TG, 적절한 OT 대안. 같은 모델·N·학습 예산·sampler로 low/high-NFE를 함께 보고한다.
2. **핵심 ablation:** 같은 TG partition/source assignment의 random vs local OT, K=1을 포함한 K 변화. “Strict-local”로 일반 TG-local을 대체하지 않는다.
3. **반복:** 우선 대응 training seed 0/1/2. 동일 checkpoint의 evaluation seed 반복은 별도로 취급한다. 3 seeds만으로 모든 동등성·일반성을 입증한 것으로 보지 않는다.
4. **비용:** 대표 설정의 full training time, coupling-only time, 동일 NFE/품질 기준 inference latency. 동일 wall-clock 비교를 한 대표 조건에서 확보한다.
5. **3D 주장 시:** 동일 category·split·normalization·표본 수를 갖춘 별도 비교가 필수다. 32-shape pilot만으로 표준 benchmark 주장을 하지 않는다.
6. **선택:** WCSS·t=0 진단, 추가 geometry·large-scale 확장은 메인 결과가 부족한 것을 대신하는 실험이 아니라 별도 탐색이다.

결과가 반대이면 “더 우수” 대신 trade-off를 보고하고, novelty 조사에서 조합까지 선행 방법과 같으면 최초성 문구를 삭제한다.

## 최종 요약 4. 초안 작성 순서

1. Problem setup·기존 TG algorithm·centroid lemma·set-preservation 범위를 먼저 작성한다.
2. 코드로 확인된 dataset/model/training/evaluation setup을 작성하고 실제 run 값이 필요한 칸은 미확인으로 둔다.
3. 접근 가능한 과거 자료의 provenance를 정리하고 메인 비교에 쓸 수 있는 기록만 선별한다.
4. Related Work에서 sample unit과 inner/outer 수준을 맞춰 차별화 후보를 정리한다.
5. 본 실험 표와 한계를 만든 뒤 Abstract·Introduction의 성능 문구를 확정한다.
6. 부록 P의 수학적 정의와 [K_SELECTION.md](K_SELECTION.md)의 partition 진단을 구분한다. 구현 테스트를 생성 품질 검증이나 Contributions로 제시하지 않는다.

## 별도 수학적 진단 부록 P — 생성 품질과의 관계 미검증

### P.0. 편성 원칙과 현재 상태

이 부록은 기존 TG의 내부 random pairing에서 도출되는 WCSS 관계와 t=0 진단 후보를 정리한다. [audit_k.py](audit_k.py)는 WCSS·CH·DB를 계산하지만, t=0 Bayes 평균장 추정기는 아니다. 진단 도구·수학적 항등식과 모델의 생성 품질에 대한 검증은 구분한다.

### P.1. WCSS의 정확한 의미 — 질문 40

Raw WCSS는 patch 내부 제곱거리의 총합이고, 아래 $W(P)$는 이를 N으로 나눈 **점당 평균**이다.

$$
W(P)=\frac1N\sum_k\sum_{y\in P_k}\|y-c_k\|^2
=\frac1N\sum_k n_k\operatorname{tr}C_k,
\quad C_k=\frac1{n_k}\sum_{y\in P_k}(y-c_k)(y-c_k)^\top.
$$

Covariance의 분모는 $n_k$이며 unbiased sample covariance의 $n_k-1$이 아니다. 고정 X0,Y,partition/source assignment에서 내부 random pairing만 평균내면

$$
W(P)=\frac1N\sum_i\operatorname{tr}\operatorname{Cov}(U_i\mid X_0,Y).
$$

좌표는 합산하고 점에 대해 평균한다. 같은 partition·capacity에서는 source assignment만 바꿔도 이 평균 내부 분산은 변하지 않는다. 개별 위치에 어떤 분산이 배치되는지는 달라질 수 있다. 전체 random bijection의 점 사이 공분산을 0이라고 가정한 식도 아니다; 이 식은 diagonal block trace들의 합이다.

### P.2. 서로 다른 두 between 항과 해석 한계

고정 target의 전체 분산은

$$
\frac1N\sum_j\|y_j-\bar y\|^2
=W(P)+\sum_k\frac{n_k}{N}\|c_k-\bar y\|^2.
$$

여기의 두 번째 항은 **한 cloud 안의 centroid scatter**다. Shape/target를 다시 뽑을 때 생기는 $\operatorname{Cov}_Y$ 평균 supervision 변동과 다르다. 고정 target에서 partition 변화로 WCSS가 감소하면 이 고정-total identity에서 centroid scatter는 증가한다. 이것만으로 평균장이 쉬워졌거나 어려워졌다고 결론내릴 수는 없다.

평가 원칙은 **WCSS 감소 ≠ 고주파 보존 증명 ≠ 생성 품질 향상**이다. 같은 WCSS에서 결과가 다르더라도 원인을 곧바로 연결성 하나로 확정하지 않는다. 분산 방향·patch 위치·marginal field 복잡도도 달라질 수 있다.

### P.3. t=0 조건부 분해 — 질문 39

**현재 상태: 설계만 있으며 별도 Bayes-mean 추정기·결과는 없다.** 모델이 보는 관측을 전체 X0로 두고, target Y는 실제 학습 분포에서 다시 샘플링한다. 현재 FPS와 source assignment는 입력이 주어지면 결정적이며, 추가 partition randomness를 도입한다면 그 변수도 평균/조건화에 포함해야 한다.

고정 X0,Y에서 내부 pairing을 해석적으로 평균내면

$$
m_i(X_0,Y)=c_{A(i;X_0,Y)}-x_i,
\qquad \operatorname{Cov}(U_i\mid X_0,Y)=C_{A(i;X_0,Y)}.
$$

실제 t=0 Bayes mean은 $v_i^*(X_0,0)=\mathbb E_Y m_i(X_0,Y)$이며,

$$
\frac1N\sum_i\operatorname{tr}\operatorname{Cov}(U_i\mid X_0)
=\mathbb E_Y W(P(Y))
+\frac1N\sum_i\operatorname{tr}\operatorname{Cov}_Y(m_i(X_0,Y)).
$$

2D에서는 Y 변화가 같은 silhouette/checkerboard에서의 **점 집합 재샘플링**이고, 여러 3D shape 간 변화라고 부르면 안 된다. 3D에서는 shape 선택·point subsampling·그에 따른 partition/source assignment가 함께 포함된다. Between 항을 순수한 “coarse 오류”라고 이름 붙이지 않는다.

M개의 독립 target로 $\hat m_i$를 추정하고 model/checkpoint를 고정하자. $S_i$는 **M개의 $m_i(X_0,Y^{(m)})$에 대한** unbiased sample covariance(분모 M−1)다. 내부 random velocity $U_i$의 covariance가 아니다. 이때

$$
\frac1N\sum_i\left(\|v_{\theta,i}(X_0,0)-\hat m_i\|^2
-\frac{\operatorname{tr}S_i}{M}\right)
$$

는 MC 평균 추정 noise를 보정한 평균장 오차 추정량이다. 유한 M에서는 음수가 나올 수 있으며 0으로 잘라 unbiased라고 주장하면 안 된다. 여러 X0와 M 증가에 따른 안정성·불확실성을 확인해야 한다. X0 수, M, checkpoint 및 seed는 아직 정하지 않았다.

한 Y를 고정한 centroid label과 모델 출력을 비교한 오차는 target를 모르는 모델의 전체 Bayes-mean 근사 오차가 아니다. 또한 t=0 분석만으로 늦은 시간의 세부 구조 복원을 설명할 수 없다.

## 재현성 기록

- 이 답변의 연구 결과는 기존 코드·기록을 읽어 정리한 것이다. 2026-09-30 정리에서는 프로젝트에서 제거한 실험의 설명·링크와 구현 hash를 갱신했다.
- PSF 기준 commit: `c74b39e1200513039cfb8d776505fb75da599e68`; [prepare_psf.py](prepare_psf.py)는 적용 patch와 working diff hash를 추적한다.
- 현재 `coupling.py` SHA256: `FF7DF22C35D95050C4DCFDAD1FB1212716B3A57CA747968AF839F5CBE855981B`.
- 현재 `train.py` SHA256: `0130AFD92C38B2768A1B6FF8998418CD378BBB49A3ED7F04B6189330C72AA59F`.
- 과거 run의 코드·환경·데이터가 위 snapshot과 같다는 보장은 없다. 논문용 manifest는 당시 checkpoint/config/environment/source hash를 기준으로 별도 작성한다.
