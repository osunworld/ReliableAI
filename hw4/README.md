# HW4 실행 안내

이 폴더의 `test.py`는 Assignment #4 Problem 2를 재현하기 위한 실행 스크립트입니다.
외부 모델을 학습한 뒤 `digits_linf.yaml` 설정 파일을 이용해 alpha-beta-CROWN으로
local robustness를 검증합니다.

실행 과정은 크게 다음과 같습니다.

- `sklearn digits` 데이터셋 로드
- 작은 PyTorch CNN 모델 학습
- 모델 checkpoint를 `artifacts/digits_cnn.pth`로 저장
- `digits_linf.yaml` 설정을 이용해 alpha-beta-CROWN 실행
- 검증 결과를 `results/` 폴더에 저장

## 1. 사용 모델과 데이터셋

### 1.1 모델

본 과제에서는 직접 정의한 `SmallDigitsCNN`을 외부 모델로 사용합니다.

모델 구조는 다음과 같습니다.

```text
Conv2d(1 -> 2) + ReLU
Conv2d(2 -> 4) + ReLU
Flatten
Linear(256 -> 32) + ReLU
Linear(32 -> 10)
```

이 모델은 alpha-beta-CROWN의 `complete_verifier/models/` 안에 있는 기본 모델이 아니라,
`test.py` 안에서 직접 정의한 PyTorch 모델입니다. 학습된 가중치는 `.pth` 형식으로 저장되므로
과제에서 요구하는 PyTorch 포맷 조건을 만족합니다.

### 1.2 데이터셋

데이터셋은 `sklearn.datasets.load_digits`를 사용합니다.

- 8x8 grayscale digit image
- class 수: 10
- 입력 범위: `[0, 1]`로 정규화
- train/test split seed: `7`
- 검증 sample index: `348`
- 해당 sample의 true label과 predicted label: `6`

Assignment #3에서도 같은 `sklearn digits` 기반 CNN을 Marabou로 검증했기 때문에,
이번 과제에서는 같은 모델 계열과 같은 sample을 사용해 두 verifier를 비교할 수 있도록 하였습니다.

## 2. Conda 환경 준비

Python 3.10 환경을 권장합니다.

```bash
cd ReliableAI/hw4
conda create -n rai python=3.10 pip -y
conda activate rai
```


## 3. alpha-beta-CROWN 저장소 받기

`test.py`는 현재 폴더 아래에 `alpha-beta-CROWN/` 저장소가 있다고 가정합니다.
이 저장소는 크기가 크고 외부 도구이므로 `.gitignore`에 포함되어 제출 커밋에는 들어가지 않습니다.

```bash
cd ReliableAI/hw4
git clone --recursive https://github.com/Verified-Intelligence/alpha-beta-CROWN.git
```

만약 submodule이 제대로 받아지지 않았다면 다음 명령을 실행합니다.

```bash
cd alpha-beta-CROWN
git submodule update --init --recursive
cd ..
```

## 4. Python 패키지 설치

`requirements.txt`에는 본 실험 실행에 필요한 최소한의 패키지를 정리했습니다.

```bash
cd ReliableAI/hw4
python -m pip install -r requirements.txt
```

`requirements.txt` 안에는 `auto_LiRPA`를 로컬 alpha-beta-CROWN 저장소에서 설치하는 줄이 포함되어 있습니다.
따라서 패키지 설치 전에 반드시 `alpha-beta-CROWN/` 저장소를 먼저 받아야 합니다.

## 5. alpha-beta-CROWN 설치 확인

alpha-beta-CROWN에서 제공하는 tutorial config 하나를 실행해 설치가 잘 되었는지 확인합니다.

```bash
cd ReliableAI/hw4
python test.py --check-install-only --device cpu --timeout 5
```

정상 실행되면 다음과 비슷한 출력이 나옵니다.

```text
[install-check] returncode=0
```

설치 확인에 사용한 예제 설정 파일은 다음입니다.

```text
alpha-beta-CROWN/complete_verifier/exp_configs/tutorial_examples/custom_box_data_example.yaml
```

HPC 환경에서는 `/tmp` 경로가 노드마다 다를 수 있으므로,
alpha-beta-CROWN 저장소는 `/tmp`가 아니라 현재 과제 폴더 아래에 두는 것이 안전합니다.

## 6. 과제 코드 실행

기본 실행 명령은 다음과 같습니다.

```bash
cd ReliableAI/hw4
python test.py --device cpu
```

이 명령은 다음을 순서대로 수행합니다.

1. `sklearn digits` 데이터셋 로드
2. `SmallDigitsCNN` checkpoint 확인
3. checkpoint가 없으면 모델 학습 후 `artifacts/digits_cnn.pth` 저장
4. `digits_linf.yaml` 설정 파일을 이용해 alpha-beta-CROWN 실행
5. 결과를 `results/` 폴더에 저장

모델을 다시 학습하고 싶으면 `--retrain` 옵션을 사용합니다.

```bash
python test.py --device cpu --retrain
```

alpha-beta-CROWN timeout을 바꾸고 싶으면 다음처럼 실행합니다.

```bash
python test.py --device cpu --timeout 60
```

## 7. YAML 설정 파일

검증 설정은 `digits_linf.yaml`에 들어 있습니다.

주요 내용은 다음과 같습니다.

- 모델: `Customized("test.py", "digits_cnn")`
- 데이터 로더: `Customized("test.py", "digits_test_data", sample_index=348, seed=7)`
- 입력 크기: `[-1, 1, 8, 8]`
- 검증 property: `L_inf` 반경 `epsilon = 0.01` 안에서 label 유지
- solver: alpha-CROWN, beta-CROWN, branch-and-bound
- timeout: 30초

즉 검증하고자 하는 성질은 다음과 같이 해석할 수 있습니다.

```text
선택한 입력 x에 대해 ||x' - x||_inf <= 0.01 인 모든 x'가
원래 class 6으로 계속 분류되는지 확인한다.
```

## 8. 출력 결과 해석

실행 후 주요 결과 파일은 `results/` 아래에 생성됩니다.

- `digits_setup.json`: 모델 정확도, sample index, label 정보
- `abcrown_stdout.log`: alpha-beta-CROWN 전체 출력 로그
- `abcrown_run.json`: 실행 명령과 전체 실행 시간
- `abcrown_digits_linf.pkl`: alpha-beta-CROWN 내부 결과 파일
- `abcrown_result_summary.json`: `.pkl` 결과를 읽기 쉽게 정리한 JSON 파일

예상되는 결과 예시는 다음과 같습니다.

```json
{
  "summary": {
    "safe-incomplete": [0]
  },
  "results": [
    ["safe-incomplete", 0.46886253356933594]
  ],
  "bab_ret": []
}
```

여기서 `safe-incomplete`는 alpha-beta-CROWN의 raw result입니다.
과제에서 요구하는 표현으로 바꾸면 다음처럼 해석할 수 있습니다.

- `safe-incomplete`: Verified
- `unsafe` 또는 `sat`: Falsified
- `unknown` 또는 `timeout`: Timeout

따라서 위 결과는 0번 verification instance가 `epsilon=0.01` 범위에서 Verified되었다는 의미입니다.

## 9. 폴더 구조


```text
hw4/
├── README.md
├── requirements.txt
├── test.py
├── digits_linf.yaml
└── results/
    └── .gitkeep
```

로컬 실행 중에는 다음 폴더와 파일도 생성됩니다.

```text
hw4/
├── alpha-beta-CROWN/
├── artifacts/
│   └── digits_cnn.pth
└── results/
    ├── abcrown_stdout.log
    ├── abcrown_run.json
    ├── abcrown_digits_linf.pkl
    ├── abcrown_result_summary.json
    └── digits_setup.json
```

`alpha-beta-CROWN/`, `artifacts/`, `results/*`는 `.gitignore`에 의해 커밋되지 않도록 설정되어 있습니다.


