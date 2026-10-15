# MS-W2: 겹친 경계 영역 재대응 × 후반 시간 가중치

기존 2D MS-W2 `[4,20,100]`을 유지한 채 coupling 변경과 학습 가중치 변경을
분리해서 비교하는 2×2 실험이다. 새로운 3D 구현이나 제거한 cross-leaf refinement
실험을 복원하는 작업은 아니다. 기존 MS-W2 설정과 기본 학습 경로는 그대로 유지한다.

## 비교하는 네 가지 모델

| 실험 | Coupling | 학습 시간 가중치 | 비교 목적 |
| --- | --- | --- | --- |
| A | 기존 MS-W2 | 균등 | 이번 실험의 기준 |
| B | 기존 MS-W2 | 후반 강조 | B−A: 동일한 대응을 후반에 더 잘 학습하는 효과 |
| C | 겹친 경계 영역 재대응 | 균등 | C−A: coupling 변경 자체의 효과 |
| D | 겹친 경계 영역 재대응 | 후반 강조 | D−C와 B−A: 가중치 효과 및 두 변경의 상호작용 |

기본값은 checkerboard와 horse에 A/B/C/D를 각각 학습하므로 **seed 하나당 8개 모델**이다.
학습 cloud는 256점, bank는 10,000점이며, 모델과 optimizer 등은 각 데이터셋의 기존
`multiscale_experiments/*.yaml`을 기준으로 한다. 기존 체크포인트를 A/B에 복사하는
실험이 아니라 네 모델을 모두 새로 학습한다.

## 랩 컴퓨터에서 실행

이 저장소를 Git으로 갱신하고 기존 Python 환경을 활성화한 뒤 **프로젝트 최상위 폴더**에서
실행한다. 별도 ZIP, 집 컴퓨터의 파일 경로, 새로운 3D repo는 필요 없다.

먼저 작은 CPU smoke test로 준비→학습→평가 연결을 확인할 수 있다.

```powershell
python run_msw2_boundary_study.py --smoke --stage all
```

Smoke는 각 데이터셋의 네 실험을 M=64, N=16, 분할 `[2,4,8]`, 작은 모델, 두 번의 학습
update, NFE 1/2로 실행한다. 경계 창은 8/16점, anchor 상한 4개, 이웃 수 4개, sweep 1회다.
**성능을 판단할 본실험이 아니며 full 결과와 섞지 않는다.** `--smoke`에는 `--device cuda`,
`--steps`, 기존 full-size bank 옵션을 함께 사용하지 않는다.

본실험 전체 실행:

```powershell
python run_msw2_boundary_study.py --stage all --device cuda
```

기본 NFE는 **1, 2, 4, 8, 16, 32, 128**이다. 기존 evaluator를 호출하므로 이미지도 기존
`eval_results` 스타일로 저장한다. 모델을 병렬 실행하거나 여러 GPU로 자동 분산하는
runner는 아니며, 선택한 장치에서 모델들을 순차 학습한다.

학습 전에 bank 준비와 검증만 먼저 하려면 다음 명령을 사용한다. 인수 없는 기본 실행은
`--stage prepare`이며, **bank OT 계산은 실행하지만 모델 학습은 시작하지 않는다.**

```powershell
python run_msw2_boundary_study.py
```

출력된 `experiment_manifest=...` 경로를 보관한다. 아래 `study_TIMESTAMP_ID`를 **실제
출력된 폴더 이름**으로 바꿔서 준비된 실험을 이어서 실행한다.

```powershell
python run_msw2_boundary_study.py --manifest "eval_results/msw2_boundary_study/study_TIMESTAMP_ID/manifest.json" --stage all
```

학습과 평가를 나누어 실행할 수도 있다.

```powershell
python run_msw2_boundary_study.py --manifest "eval_results/msw2_boundary_study/study_TIMESTAMP_ID/manifest.json" --stage train
python run_msw2_boundary_study.py --manifest "eval_results/msw2_boundary_study/study_TIMESTAMP_ID/manifest.json" --stage eval
```

완료한 bank·학습·평가 작업은 hash를 확인하고 재사용한다. 학습 도중 중단한 모델에는
optimizer-state resume이 없으므로 그 모델은 처음부터 다시 학습한다. 같은 manifest를
여러 프로세스에서 동시에 실행하지 않는다. 준비한 YAML·bank·체크포인트를 수정해서
재개하지 말고, 설정을 변경하려면 새 실험을 만든다. `--manifest` 재개에서는 `--stage`와
추가 `--nfe`만 지정할 수 있으며 장치·seed·학습 횟수는 기존 설정을 유지한다.

다른 실행 예시:

```powershell
python run_msw2_boundary_study.py --datasets horse --stage all --device cuda
python run_msw2_boundary_study.py --stage all --device cuda --seeds 0 1 2
python run_msw2_boundary_study.py --stage all --device cuda --bank-seed 1
python run_msw2_boundary_study.py --stage all --device cuda --nfe 1 2 4 8 16 32 64 128
python run_msw2_boundary_study.py --manifest "eval_results/msw2_boundary_study/study_TIMESTAMP_ID/manifest.json" --stage eval --nfe 64
```

두 데이터셋에서 seed 0/1/2를 지정하면 24개 모델을 학습한다. 마지막 명령은 저장된
NFE 목록에 64를 추가하며 기존 평가를 삭제하거나 새로 덮어쓰지 않는다.

### 랩 컴퓨터의 기존 baseline bank를 재사용


기존 bank를 다시 만들고 싶지 않다면 랩 저장소에 있는 **original MS-W2의 config.yaml**을
지정한다. 다음 경로의 `YOUR_*_RUN` 부분만 실제 run 폴더 이름으로 바꾼다.

```powershell
python run_msw2_boundary_study.py --stage all --device cuda --base-checkerboard-config "runs/checkerboard/YOUR_MSW2_RUN/config.yaml" --base-horse-config "runs/horse/YOUR_MSW2_RUN/config.yaml"
```

말만 기존 bank로 실행할 때:

```powershell
python run_msw2_boundary_study.py --datasets horse --stage all --device cuda --base-horse-config "runs/horse/YOUR_MSW2_RUN/config.yaml"
```

지정한 config의 bank 파일이 실제로 존재해야 한다. M=10,000, N=256, `[4,20,100]`, 동일
scale weight와 bank seed가 일치해야 하며, 기본 bank seed가 0이 아니면 `--bank-seed`도
맞춰 지정한다. 기존 모델·학습 설정을 가져오므로 데이터셋의 baseline 설정을 확인한다.
기존 checkpoint는 재사용하지 않고 bank 좌표와 baseline permutation만 재사용한다.
다른 컴퓨터의 절대 경로는 임의로 추측해서 고치지 않는다.

## 저장되는 결과

실험 관리 경로는 `eval_results/msw2_boundary_study/study_TIMESTAMP_ID/`다.

| 경로 | 내용 |
| --- | --- |
| `manifest.json` | A/B/C/D 작업, config·bank·checkpoint·평가 파일 경로와 hash, endpoint 검증 |
| `configs/` | 각 데이터셋·seed·실험별 YAML |
| `banks/` | 새 baseline bank 및 경계 재대응 bank; 기존 bank를 지정했다면 baseline은 원래 위치에서 재사용 |
| `summary.json` | 평가·시간·bank 진단, B−A/C−A/D−C/D−B 및 D−C−B+A 차이 |
| `comparison.csv` | 데이터셋·seed·실험·NFE별 CD·누출률·JS 및 기타 평가 지표와 시간 |
| `runs/<dataset>/...` | 기존 형식의 checkpoint, `config.yaml`, `training.json` |
| `eval_results/<dataset>/...` | 기존 형식의 NFE별 JSON·PNG; manifest와 summary에서 연결 |

`summary.json`의 bank metadata에는 목적값 전후, 변경된 점 대응 비율, target leaf를
바꾼 비율, scale/공간별 창 coverage, 시도·수락된 local solve 수와 전처리 시간이 포함된다.
CSV의 학습·전처리 시간은 NFE 행마다 반복되므로 행 전체를 합산하면 안 된다.
`inference_ms_per_cloud`는 batch 적분 시간을 cloud 수로 나눈 처리량 지표이며,
cloud 하나를 따로 실행한 latency가 아니다.

Paired difference가 음수이면 lower-is-better 지표에서는 앞쪽 실험이 좋아졌다는 뜻이다.
이 차이는 유의성 검정이나 신뢰구간이 아니고, 서로 다른 의미의 ROI raw mass 같은 값까지
모두 작을수록 좋다고 해석해서는 안 된다. 학습 fitting의 시간대별 상세 진단은 별도의
진단 코드로 확인하며, 이 runner의 기본 평가를 시간대별 MSE 검사와 혼동하지 않는다.

## 통제하는 조건

동일한 데이터셋·bank seed에서는 다음을 고정한다.

- A/B와 C/D의 source·target bank 좌표는 정확히 동일하다. C/D에서는 permutation만 바뀐다.
- A/B는 같은 baseline bank를 공유하고, C/D는 같은 재대응 bank를 공유한다.
- 같은 training seed의 네 모델은 같은 초기화, source bank index 추출 순서와 시간 추출
  순서를 사용한다. index 난수는 `seed + 1`, 시간 전용 난수는 `seed + 2`이다.
- 시간은 모든 실험에서 균등하게 추출한다. B/D에서만 loss의 시간 가중치를 바꾼다.
- 평가에는 공통 설정과 seed를 사용하며, inference 입력은 bank가 아닌 fresh Gaussian이다.

시간 전용 난수 분리는 이번 A에도 적용된다. 따라서 과거에 저장한 baseline 체크포인트의
학습 궤적과 이번 A가 동일하다고 가정하면 안 된다. 기존 bank를 재사용하더라도 A/B/C/D는
이번 runner에서 함께 새로 학습하여 비교한다. 같은 난수 조건은 CUDA 연산까지 환경을
넘어 bitwise 재현한다는 보장은 아니다.

Training seed와 bank seed는 별개다. 여러 training seed에서 같은 bank를 사용하면
네트워크 학습 변동을 확인하는 것이며, bank 자체의 변동까지 확인하려면 별도의 bank seed
실험도 필요하다.

## 후반 시간 가중치

기본 threshold는 `0.75`, late multiplier는 `2.0`이다. 한 cloud의 좌표별 MSE를 먼저
평균하고, 그 cloud의 시간에 해당하는 가중치를 곱한 다음 batch 평균을 계산한다.

```text
t ~ Uniform(0, 1)
w(t) = [1 + (multiplier - 1) * 1(t >= threshold)]
       / [threshold + multiplier * (1 - threshold)]
loss = mean_over_clouds(w(t) * mean_over_points_and_coordinates(squared_velocity_error))
```

기본값에서 초기 구간의 가중치는 0.8, 후반 구간은 1.6이며 시간 전체의 기대 가중치는 1이다.
가중치를 minibatch의 관측 시간 비율로 다시 정규화하지 않는다. 큰 이동, 큰 속도 또는
Jacobian에 직접 벌점을 추가하지 않는다. 가중 loss 자체는 A/C와 B/D 사이의 fitting 비교
지표로 사용할 수 없다. 학습 적합도는 **가중하지 않은 MSE**로 비교해야 한다.

## 겹친 경계 영역 재대응이 하는 일


먼저 기존 MS-W2의 전체 permutation을 만든다. 그다음 기존의 각 scale 분할에서 경계를
가로지르는 공간적 이웃을 찾아 여러 크기의 겹친 창을 구성한다. Source 공간과 target
공간 모두에서 후보 창을 만든다. 각 창의 순차 처리는 다음 형태다.

```text
Source 창: I = 창 안의 source index, J = 현재 permutation[I]
Target 창: J = 창 안의 target index, I = 현재 inverse_permutation[J]
현재 I와 J 사이의 multiscale point cost로 exact linear assignment를 푼다.
실제 비용이 감소할 때만 현재 permutation을 갱신한다.
다음 창은 갱신된 permutation과 inverse_permutation을 사용한다.
```

Point cost는 **마지막 scale 이전의 패치 W₂ 비용을 점 쌍에 상속한 항**과 **마지막 scale의
실제 점 간 제곱거리 항**을 기존 scale weight로 합친 것이다. 마지막 scale 패치 비용을
추가로 중복 합산하지 않는다. 초기 block-constrained permutation에서 평균한 값은 기존
MS-W2 목적값과 일치하지만, 이후에는 일부 점이 원래 배정된 target leaf를 벗어날 수 있다.

창 하나의 재대응은 exact이지만, 제한된 창을 순차 방문하는 전체 알고리즘이 전역 최적
permutation을 찾는다는 뜻은 아니다. 모든 창이 동시에 동일한 계획을 갖게 하는 합의
최적화나 이웃 속도 차이에 대한 smoothing penalty도 아니다. **같은 점들을 공동으로
재대응할 여지를 경계 주변에서 추가로 검사하는 실험**이다.

Source/target 좌표 이동, 노이즈, target 평균화는 없다. 모든 단계에서 target index를
한 번씩 사용하는 전역 bijection을 유지하므로 유한 bank의 경험분포는 변하지 않는다.
Population Gaussian/target 분포나 fresh-source 생성 밀도가 정확히 보존된다는 뜻은 아니다.

기본 창 크기는 64/128점, 각 공간·각 scale의 anchor 상한은 32개, 경계 탐색 이웃 수는
16개, sweep은 2회다. 같은 anchor에서 두 크기의 창을 만든다. 최대 local solve 횟수는
`2 spaces × 3 scales × 2 widths × 32 anchors × 2 sweeps = 768`이며, 기본 안전 상한은
1,000회다. 경계 후보 부재, 중복 창 제거, 개선 없는 sweep에서의 조기 종료로 실제 횟수는
더 작을 수 있다. 모든 경계점을 빠짐없이 처리하거나 모든 가능한 겹친 영역을 최적화하는
방식은 아니다. 이 설정은 첫 대조실험을 위한 계산 예산이며 최적 설정으로 주장하지 않는다.

현재 baseline cache에는 패치별 전체 비용행렬이 없으므로, 재대응 전처리에서 nonleaf
비용표를 재계산한다. 기본 `[4,20,100]`에서는 **큰·중간 scale의 exact patch OT 416회**가
local solve 예산과 별도로 필요하다. 따라서 작은 창만 푼다는 이유로 전처리 전체가 매우
가볍다고 가정하면 안 된다. 이 전처리는 CPU SciPy를 사용하며 `--device cuda`는 모델
학습·평가 장치를 지정하는 것이지 이 OT solver를 GPU로 전환하는 옵션이 아니다.

기존 refinement 역시 비용 감소가 생성 품질 개선으로 이어지지 않았으므로, 이번 실험도
비용 감소나 대응 변경 비율만으로 성공을 판정하지 않는다.

## 결과를 판단하는 기준

- 먼저 같은 데이터셋·seed·NFE에서 A/B/C/D를 비교한다. 1NFE와 128NFE를 함께 보고,
  CD뿐 아니라 누출률·histogram JS·체커보드 cell mass·말 ROI를 확인한다.
- C−A는 coupling 실험, B−A는 학습 가중치 실험이다. D가 A보다 좋다는 사실만으로
  coupling 변경이 효과적이었다고 결론 내리지 않는다.
- Transport 목적값과 Jacobian 감소를 성공 조건으로 두지 않는다. 후반 fitting 개선과
  fresh-source 생성 품질 개선이 실제로 함께 나타나는지 확인한다.
- 단일 seed는 pilot이다. 모델 seed를 반복하고 이후 bank seed도 바꿔서 재현성을 확인한다.
- Bank 전처리, 추가 경계 재대응, 온라인 학습, inference 시간을 구분한다. 공유 bank의
  생성 비용을 A/B 또는 C/D의 모델 수만큼 중복 계산하지 않는다.

