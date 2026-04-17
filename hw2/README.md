# HW2: CIFAR-10 Differential Testing with DeepXplore-Style PyTorch Pipeline

## 1. 개요

이 디렉토리는 DeepXplore의 핵심 아이디어인 **differential testing**과 **neuron coverage**를
CIFAR-10용 ResNet50 모델 2개에 적용하기 위한 실험 코드와 결과를 담고 있다.

원본 `deepxplore/` 저장소는 Python 2.7, TensorFlow 1.x, standalone Keras 환경을 전제로 작성되어
현재 환경에서 그대로 실행하기 어렵다. 따라서 본 과제에서는 원본 저장소를 참고하되,
현대 PyTorch 환경에서 재현 가능하도록 `CIFAR10_torch/` 경로를 별도로 구현하였다.

이 구현은 다음 목적을 가진다.

- 동일한 CIFAR-10 입력을 두 모델에 넣고 예측 불일치를 찾는다.
- 불일치가 없을 경우 gradient 기반 입력 변형으로 disagreement를 유도한다.
- hidden state를 이용해 neuron coverage를 추적한다.
- 생성된 결과 이미지를 `results/` 폴더에 저장한다.

## 2. 사용 모델

### 2.1 모델 출처

현재 기본 실험에는 Hugging Face의 다음 두 모델을 사용한다.

- 모델 1: [`jialicheng/cifar10_resnet-50`](https://huggingface.co/jialicheng/cifar10_resnet-50)
- 모델 2: [`jialicheng/unlearn-so_cifar10_resnet-50_salun_10_100`](https://huggingface.co/jialicheng/unlearn-so_cifar10_resnet-50_salun_10_100)

두 모델 모두 기본 backbone은 `microsoft/resnet-50`이지만, 학습 절차와 하이퍼파라미터가 다르다.

### 2.2 학습 단계에서의 차이

#### 모델 1: `jialicheng/cifar10_resnet-50`

- `microsoft/resnet-50`을 CIFAR-10에 일반 fine-tuning한 모델
- Hugging Face model card 기준 주요 설정
- learning rate: `1e-5`
- train batch size: `128`
- eval batch size: `256`
- seed: `42`
- optimizer: `Adam`
- lr scheduler: `linear`
- epoch: `300`
- 평가 정확도: `0.9672`

#### 모델 2: `jialicheng/unlearn-so_cifar10_resnet-50_salun_10_100`

- `microsoft/resnet-50`을 기반으로 CIFAR-10에서 추가적인 **unlearning 계열 절차**가 반영된 모델
- 모델 이름의 `unlearn-so`, `salun_10_100`이 그 변형 학습 절차를 나타낸다
- Hugging Face model card 기준 주요 설정
- learning rate: `1e-4`
- train batch size: `128`
- eval batch size: `256`
- seed: `100`
- optimizer: `adamw_torch`
- lr scheduler: `linear`
- epoch: `10`
- 평가 정확도: `0.9667`

## 3. 모델 불러오기 및 실행 방식

### 3.1 환경 설치

현재 과제 실행 기준 패키지는 [requirements.txt](/Users/Sun/Documents/ReliableAI/hw2/requirements.txt)에 정리되어 있다.

```bash
cd ~/ReliableAI/hw2
pip install -r requirements.txt
```



### 3.2 test 실행

[test.py](/Users/Sun/Documents/ReliableAI/hw2/test.py)는 모델과 전처리가 정상 동작하는지 확인하는 간단한 점검 코드다.

```bash
cd ~/Documents/ReliableAI/hw2
python test.py
```

이 스크립트는 다음만 확인한다.

- 모델 2개가 정상 로드되는지
- CIFAR-10 샘플 1개가 정상 전처리되는지
- 두 모델이 예측값과 confidence를 출력하는지

### 3.3 실제 differential testing 실행

실제 실험 코드는 [CIFAR10_torch/gen_diff.py](/Users/Sun/Documents/ReliableAI/hw2/CIFAR10_torch/gen_diff.py)이다.

본 실험은 아래의 설정으로 수행되었다.

```bash
cd ~/Documents/ReliableAI/hw2
python CIFAR10_torch/gen_diff.py occl 2.0 0.01 0.03 180 40 0.5 --occlusion_size 12 12
```

인자 의미:

- `transformation`: `light`, `occl`, `blackout` 중 하나
- `weight_diff`: 두 모델의 출력 차이를 유도하는 loss 가중치
- `weight_nc`: neuron coverage 증가를 유도하는 loss 가중치
- `step`: 입력 업데이트 step 크기
- `seeds`: 시작 입력 개수
- `grad_iterations`: seed당 최대 gradient 반복 횟수
- `threshold`: coverage 판정 threshold

실행 결과는 기본적으로 [results](/Users/Sun/Documents/ReliableAI/hw2/results)에 저장된다.

## 4. 원본 DeepXplore에 가한 수정사항

### 4.1 왜 새 경로가 필요한가

원본 [deepxplore](/Users/Sun/Documents/ReliableAI/hw2/deepxplore)는 다음 환경을 전제로 한다.

- Python 2.7
- TensorFlow 1.x
- standalone Keras

반면 본 과제는 현재 로컬 환경에서 재현 가능한 PyTorch 기반 실험이 필요하므로,
원본을 직접 수정해 끼워 맞추기보다 `CIFAR10_torch/`를 별도 구현하였다.

### 4.2 `CIFAR10_torch/`에서 달라진 점

원본 DeepXplore와 비교했을 때 주요 차이는 다음과 같다.

- 프레임워크를 Keras/TensorFlow에서 **PyTorch**로 변경
- 모델 로딩을 로컬 `.h5` 대신 **Hugging Face pretrained checkpoint** 기반으로 변경
- 입력 데이터를 ImageNet/MNIST 대신 **CIFAR-10**으로 변경
- neuron coverage 계산을 Keras layer output 대신 **PyTorch hidden states** 기반으로 변경
- 결과 저장 경로를 `generated_inputs/` 대신 **`hw2/results/`**로 변경
- 코드 전체를 Python 3 환경에서 동작하도록 재구성

## 5. 입력 전처리 일관성을 위한 수정사항

- CIFAR-10 `32x32` 입력을 모델 processor 기준 크기로 resize
- processor의 mean/std로 정규화 통일
- perturbation 후 입력 범위를 clip
- 저장 시에는 정규화를 되돌려 사람이 읽을 수 있게 변환

## 6. 폴더 구조와 역할

전체 구조는 아래처럼 이해하면 된다.

```text
hw2/
├── README.md
├── requirements.txt
├── test.py
├── CIFAR10_torch/
│   ├── configs.py
│   ├── models.py
│   ├── utils.py
│   └── gen_diff.py
├── results/
├── data/
└── deepxplore/
```

### 6.1 상위 디렉토리

#### [README.md](/Users/Sun/Documents/ReliableAI/hw2/README.md)

- 현재 과제 실험 방법과 코드 구조 설명 문서

#### [requirements.txt](/Users/Sun/Documents/ReliableAI/hw2/requirements.txt)

- 현재 PyTorch 기반 실험을 위한 패키지 목록

#### [test.py](/Users/Sun/Documents/ReliableAI/hw2/test.py)

- smoke test 실행 파일

### 6.2 `CIFAR10_torch/`

현재 과제에서 **실제로 사용하는 메인 코드**가 들어 있는 디렉토리다.

#### [configs.py](/Users/Sun/Documents/ReliableAI/hw2/CIFAR10_torch/configs.py)

- 콘솔 출력 색상 설정 등 간단한 설정값 보관

#### [models.py](/Users/Sun/Documents/ReliableAI/hw2/CIFAR10_torch/models.py)

- 모델 및 processor 로딩

#### [utils.py](/Users/Sun/Documents/ReliableAI/hw2/CIFAR10_torch/utils.py)

- 전처리, 후처리, coverage 계산, 저장 유틸리티

#### [gen_diff.py](/Users/Sun/Documents/ReliableAI/hw2/CIFAR10_torch/gen_diff.py)

- differential testing 메인 실행 스크립트

### 6.3 `results/`

- 실험 결과 저장 폴더
- `*_orig.png`: 원본 이미지
- `*_gen.png`: perturbation 적용 후 이미지
- `*.json`: 예측 결과, confidence, coverage 메타데이터

또한 [requirements-legacy.txt](/Users/Sun/Documents/ReliableAI/hw2/deepxplore/requirements-legacy.txt)는
원본 DeepXplore 실행을 위한 legacy 의존성을 따로 기록한 파일이다.

## 7. 결과 저장 형식

한 개의 disagreement 사례가 저장될 때 다음 파일이 함께 생성된다.

- `case_prefix_orig.png`: 원본 입력
- `case_prefix_gen.png`: 생성된 입력
- `case_prefix.json`: 메타데이터

JSON에는 다음 정보가 포함된다.

- 사용한 모델 ID
- true label
- 각 모델의 예측 라벨과 confidence
- 모델별 coverage
- 평균 coverage

