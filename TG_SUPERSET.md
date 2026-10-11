# Dense superset TG 비교

TG의 coarse 규칙(FPS balanced target partition + 실제 patch centroid에 대한 exact source 배정)은 유지하되, 이 계산을 N=256 cloud마다 하는 대신 M개의 dense 점 풀에서 수행한다. Fine exact OT, frozen fine permutation, affine search, subpatch, anchor prior, ID conditioning은 사용하지 않는다. 모델, FM loss, optimizer, 학습 update 수, 표준 Gaussian 추론은 유지한다.

## 비교 조건

| 이름 | coarse 구조 | Gaussian hybrid beta |
|---|---|---:|
| stream | 독립 cloud 640,000개, 각각 N=256에서 계산, 재사용 없음 | 0 |
| stream_hybrid | 위와 같은 stream cache | 0.1 |
| superset | M=8,192인 dense 풀 8개에서 계산, 매번 N=256 부분집합 | 0 |
| superset_hybrid | 위와 같은 dense cache | 0.1 |

각 dataset은 기존 설정의 batch 64 × 10,000 updates를 사용한다. 말·체커보드 전체 실행은 8개 학습이다. Stream의 64만 개는 **cloud 수**, superset의 M은 **풀 하나의 점 수**다. 두 수를 혼동하면 안 된다. Dense 조건의 기본값은 실험 시작점이지 최적값이 아니다. `--superset-size`, `--supersets`, `--beta`로 바꿀 수 있다.

## 샘플링과 보존 범위

1. Cloud마다 dense 풀 쌍 하나를 균등 선택한다.
2. 해당 source 풀 전체에서 N개 인덱스를 비복원 균등 추출한다.
3. 선택된 source의 patch별 개수 n_k를 센다. n_k=32 같은 고정 quota를 사용하지 않는다.
4. 각 target patch에서 n_k개를 비복원 균등 추출하고, 선택된 source 위치에 무작위로 배정한다.
5. 필요하면 **배정 후** `x0' = sqrt(1-beta)*x0 + sqrt(beta)*epsilon`을 적용한다. Patch를 재배정하지 않는다.
6. `xt=(1-t)*x0'+t*y`, 정답 속도 `y-x0'`로 기존 FM loss를 계산한다.

Dense source/target patch capacity가 같으므로 n_k는 multivariate hypergeometric 법칙을 따른다. 임의의 target N-subset의 확률은 `prod C(c_k,n_k)/C(M,N) × prod 1/C(c_k,n_k) = 1/C(M,N)`이다. 따라서 source뿐 아니라 **target도 풀 전체의 균등 N-subset 주변분포**를 가진다. 선택된 source/target 인덱스는 cloud 안에서 중복되지 않고, fine 대응은 매번 새로 생성된다. 모든 M개 점을 매 update 사용하는 것은 아니다.

보존되는 것은 **유한 풀의 부분집합 법칙**이다. 고정된 유한 풀, 비복원 추출, cloud별 공통 풀 선택 때문에 실제 iid 모집단 cloud 법칙과 정확히 같다고 주장하지 않는다. Gaussian hybrid도 유한 cache를 정확한 Gaussian으로 만들지 않는다. Beta>0에서는 source 좌표가 바뀌지만 target 좌표는 그대로다. 모든 조건의 추론은 새 iid 표준 Gaussian에서 시작하며 cache나 patch 정보가 필요 없다.

Dense에서 coarse 배정을 공유하는 것은 기존 N-point TG와 **다른 coupling**이다. 표본 수 증가만을 검증하는 실험이 아니다. Dense의 공유 구조가 학습을 쉽게 만들지, 고정 partition 편향을 키울지는 결과로 확인해야 한다. NSOT의 superset/hybrid 아이디어를 TG에 적용한 ablation이며, 새로운 Gaussian perturbation 방법이라고 주장하지 않는다.

## 실행

기존 실험 장비에서 프로젝트 폴더로 이동하고 기존 학습 환경을 활성화한다. 입력 YAML은 수정하지 않고, effective config와 정확한 run 경로를 manifest에 저장한다.

```powershell
# 읽기 전용: 8개 작업의 설정 확인
python run_tg_superset.py --dry-run

# 작은 CPU 검증: 2 datasets × 4 conditions, 각 2 updates와 NFE 1/2 평가
# 전체 학습용 cache와 분리되며, 생성 품질 판단용 결과가 아니다.
python run_tg_superset.py --stage smoke

# 실제 비교: cache 준비 → 학습 → NFE 1,2,4,8,16,32,64,128 평가
python run_tg_superset.py

# 한 dataset만 실행
python run_tg_superset.py --dataset horse
```

`python run_tg_superset.py`는 **평가 전용 명령이 아니다. 실제 학습까지 수행한다.** 기본 stream cache는 dataset당 약 3.68 GiB가 필요하고, dense cache는 기본값에서 약 1.5 MiB다. Dense coarse 비용은 M×K이며 M×M fine OT는 계산하지 않는다. 기존 cache는 hash와 설정을 검증해 재사용하고 덮어쓰지 않는다. Beta 조건끼리는 geometry cache를 공유하며, 별도의 geometry RNG로 같은 subset/fine draw 순서를 사용한다.

단계별 실행 및 재개:

```powershell
python run_tg_superset.py --stage prepare
# 위 명령이 출력한 실제 manifest 경로 사용. 'latest' 검색은 하지 않는다.
python run_tg_superset.py --stage train --manifest "experiment_manifests/실제_폴더/manifest.json"
python run_tg_superset.py --stage eval --manifest "experiment_manifests/실제_폴더/manifest.json"
```

완료된 학습/평가는 재개할 때 건너뛰되 기록된 config·checkpoint·JSON hash를 검증한다. 평가를 다시 하려면 `--rerun-eval`. 실행 옵션을 바꾸어 새 실험을 하려면 기존 manifest를 수정하지 말고 새 manifest를 만든다. Smoke manifest를 전체 실험으로 재사용할 수 없다.

## 평가와 해석

기존 eval.py/eval_horse.py를 그대로 사용하므로 CD, histogram JS, leakage 및 horse thin/gap ROI 지표와 렌더링이 저장된다. Run config에 beta가 표시되고, JSON에 사전 계산·cache setup·학습·추론 시간이 기록된다. 공유 cache의 준비 시간을 beta 조건마다 새로 계산한 비용처럼 합산하지 않는다.

비교는 낮은 NFE뿐 아니라 NFE 64/128의 수렴 품질, 배경 누출, 얇은 구조의 질량 오차를 함께 본다. 동일 evaluation seed로 생성 source/target을 맞춘다. 한 training seed 결과는 예비 결과이며 `--seed 1`, `--seed 2` 등으로 재현해야 한다. NSOT를 넘는다는 결론에는 동일 조건의 별도 NSOT 기준선이 필요하다.

Jacobian/Dirichlet 진단도 새 sampler를 지원한다:

```powershell
python audit_jacobian.py "runs/horse/실제_run/config.yaml" --dataset horse --scope both --clouds 10 --probes 10
```

`coupling` scope는 저장된 풀에서 새로 뽑은 학습 coupling의 interpolant를, `rollout` scope는 새 Gaussian에서 실제 생성한 경로를 측정한다. 전자는 unseen-pool 검증이 아니다. FM residual과 Jacobian이 낮아져도 최종 분포가 정확하다는 보장은 없으며 생성 지표를 우선한다.

TG_LOCAL 전용 affine/subpatch 코드·설정·테스트·문서는 제거했다. 원본 TG baseline, 기존 NSOT 구현, 일반 진단 도구는 보존했다.
