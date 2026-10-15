# MS-W2 / Leaf-only / QAT coupling comparison

기존 2D **checkerboard와 horse**에서 coupling만 비교합니다. QAT 논문의 전체
모델/학습 설정 재현이 아니라 **QAT coupling + 이 프로젝트의 동일한
PointSetTransformer backbone** 비교입니다. 기존 MS-W2 설정과 평가 렌더링은
그대로 유지합니다.

| Variant | Coupling | Source used during training |
| --- | --- | --- |
| `multiscale` | MS-W2 `[4,20,100]`, weights `[1,1,1]` | Fixed Gaussian bank |
| `leaf_only` | Same nested partitions, weights `[0,0,1]` | The identical fixed Gaussian bank |
| `qat` | Quantile AlignTree coupling, singleton target leaves | Fresh continuous conditional Gaussian samples |

Leaf-only는 `[100]`으로 새로 분할하는 실험이 아닙니다. `[4,20,100]`으로 만든
동일한 계층과 작은 패치를 유지하고, 작은 패치 비용만으로 leaf assignment를
계산합니다. 선택한 leaf 쌍 내부의 exact point OT도 유지합니다. 가중치가 0인
큰/중간 패치의 OT 비용 계산은 생략하므로 준비 시간도 비교할 수 있습니다.

QAT는 target의 최대 분산 좌표를 선택하고 그 좌표의 평균에서 재귀 분할합니다.
각 target 분할의 점 개수 비율을 Gaussian의 조건부 분위수 경계로 옮깁니다.
target point를 균등하게 선택하고 해당 singleton leaf의 Gaussian rectangle에서
독립적으로 source를 샘플링합니다. 따라서 QAT에는 유한 source bank, jitter,
leaf 내부 Hungarian matching이 없습니다. MS-W2의 100개 OT leaf와 QAT의
singleton leaf는 각 coupling의 설계 차이이며, QAT에 MS-W2의 leaf 개수를
강제로 적용하지 않습니다. 완전히 동일한 target 좌표가 중복되면 기하적으로
분리할 수 없으므로 같은 terminal leaf에 묶고 중복 횟수만큼 질량을 유지합니다.
서로 다른 좌표는 singleton까지 나누며, float64 정밀도로 분할할 수 없는
경우에는 조용히 근사하지 않고 오류로 알립니다.

참고: [QAT-FM paper, §3.2 and Appendix B, Algorithms 1–2](https://arxiv.org/html/2609.03443v1).

## Controlled protocol

- Dataset별 세 방법의 **target 좌표를 byte-identical하게 검증**합니다.
  MS-W2와 leaf-only의 source bank 좌표도 동일합니다. QAT의 source는 연속
  Gaussian이므로 두 bank와 같은 유한 source를 강제로 쓰지 않습니다.
- 기본 `M=10000`, cloud당 `N=256`, 기존 model/data/training/evaluation 설정을
  공유합니다. 기본 학습은 10000 updates입니다. bank seed는 기본 0으로
  고정하고, 학습 seed별로 세 모델을 모두 실행합니다.
- 같은 학습 seed에서는 모델 초기화와 시간 sampling RNG 조건을 맞춥니다.
  pair-index RNG는 별도 stream이며, QAT의 conditional source sampling도
  독립된 stream을 사용하여 시간 sampling 순서를 바꾸지 않습니다.
- 평가에서는 세 방법 모두 **같은 fresh standard Gaussian 입력**, 같은 평가
  seed/batch/점 개수/target 기준을 사용합니다. QAT tree는 추론에 필요하지
  않습니다.
- 기본 Euler NFE는 `1,2,4,8,16,32,64,128`입니다. 학습 후 기존 evaluator의
  CD, histogram JS, leakage 및 dataset별 metric을 저장합니다.
- bank 준비 시간, 학습 loop 시간, inference integration 시간을 분리합니다.
  inference 시간에는 checkpoint loading, warmup, metric 및 rendering이
  포함되지 않습니다. smoke timing은 성능 비교용이 아닙니다.
- 단일 seed 결과는 pilot입니다. 최종 비교는 최소 여러 학습 seed를 반복하고,
  고정 bank seed를 바꾼 실험은 별도로 구분하는 것이 좋습니다.

## Lab computer commands

repo root에서 기존 프로젝트 환경을 활성화한 뒤 실행합니다. 아래 명령에는
집/랩 컴퓨터에 종속된 절대 경로가 없습니다. zip이나 checkpoint 복사는
필요하지 않으며 코드는 평소처럼 git으로 옮기면 됩니다.

빠른 CPU smoke: 두 데이터셋, 세 방법, 작은 bank/model, 2 updates와 NFE 1/2로
준비 → 학습 → 평가 → 원본 그림 비교를 검사합니다.

```powershell
python run_multiscale.py --coupling-study --smoke --stage all
```

본실험 한 seed: `--device cuda`는 사용 중인 환경에 맞게 지정합니다.

```powershell
python run_multiscale.py --coupling-study --stage all --device cuda --seeds 0
```

최종 여러 seed 비교:

```powershell
python run_multiscale.py --coupling-study --stage all --device cuda --seeds 0 1 2
```

사용자가 처음 요청한 NFE만 저장하려면:

```powershell
python run_multiscale.py --coupling-study --stage all --device cuda --seeds 0 --nfe 1 2 4 8 16 32 128
```

말/체커보드 중 하나만 실행하려면 `--datasets horse` 또는
`--datasets checkerboard`를 추가합니다. runner는 모델들을 순차 실행하며,
위 명령 자체가 다중 GPU 병렬 학습을 켜지는 않습니다.

준비만 먼저 실행하려면 `--stage`를 생략합니다. 기본 동작은 **prepare only**이며
학습이 자동으로 시작되지 않습니다.

```powershell
python run_multiscale.py --coupling-study --device cuda --seeds 0
```

출력에 표시된 `experiment_manifest=.../manifest.json` 경로를 사용해 이어서
실행합니다. 아래 `<experiment>`는 실제 생성된 폴더 이름으로 바꾸세요.

```powershell
python run_multiscale.py --manifest eval_results/coupling_study/<experiment>/manifest.json --stage all
```

resume에서는 `--coupling-study`, `--seeds`, `--device` 등 새 실험을 만드는
옵션을 다시 주지 않습니다. 완료된 bank/train/eval artifact는 hash 검증 후
재사용합니다. 평가 NFE 추가는 `--stage eval --nfe ...`로 할 수 있습니다.
checkpoint 없이 평가부터 시작할 수는 없습니다.

## Outputs and interpretation

study별 설정/manifest/summary/비교 산출물은
`eval_results/coupling_study/<experiment>/`에 저장됩니다.

- `manifest.json`: 실제 config, bank, checkpoint, evaluation artifact와 hash.
- `summary.json`: 모든 모델의 진행 상태와 기록된 결과.
- `bank_comparison.csv`: coupling 준비 단계 비교.
- `comparison.csv`: dataset/seed/variant/NFE별 metric 및 시간.
- `comparison_by_seed.csv`: 여러 학습 seed의 평균과 표본 표준편차. 단일
  seed의 표준편차는 0으로 오해하지 않도록 빈칸으로 기록합니다.
- `<dataset>_seed0_nfe_001_comparison.png` 등: 기존 evaluator가 만든 PNG를
  MS-W2 → leaf-only → QAT 순서로 그대로 이어 붙인 그림입니다. 새 density나
  scatter renderer로 다시 그리지 않습니다.

원본 checkpoint와 `training.json`은 기존 `runs/` 구조를 유지하고, 원본
평가 PNG/JSON은 기존 `eval_results/<dataset>/...`에 저장됩니다. 재사용하는
coupling cache는 `coupling_cache/` 아래에 있습니다. 비교 CSV의 timing은
같은 하드웨어/환경에서 실행한 값끼리 해석하세요. QAT의 이론적 Gaussian
marginal은 유한 정밀도 inverse-CDF sampling 오차와 구분해야 하며, 생성
품질 우위를 구현 단계에서 가정하지 않습니다.
