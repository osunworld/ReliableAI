# HW3 실행 안내

이 폴더의 `test.py`는 다음 과정을 한 번에 수행합니다.

- `sklearn digits` 데이터셋 로드
- 작은 `2-layer CNN` 학습
- 모델을 `ONNX` 형식으로 변환
- Marabou Python API(`maraboupy`)로 local robustness 검증

예를 들어 아래 명령으로 실험을 다시 재현할 수 있습니다.

```bash
python test.py --epsilon 0.1
```

## 1. Conda 환경 준비

```bash
cd hw3
conda create -n hw3 python=3.10 -y
conda activate hw3
pip install -r requirements.txt
```

## 2. Marabou 저장소 받기

`test.py`는 현재 폴더 아래에 `Marabou/` 저장소가 있다고 가정합니다.  
따라서 아래처럼 clone 해야 합니다.

```bash
cd hw3
git clone https://github.com/NeuralNetworkVerification/Marabou.git
```

## 3. Marabou 빌드 전 준비

Marabou 소스 빌드 중에는 시스템 `wget` 명령이 필요합니다.

macOS(Homebrew):

```bash
brew install wget
```

Ubuntu/Debian:

```bash
sudo apt-get update
sudo apt-get install -y wget
```

## 4. Marabou 빌드

`requirements.txt`에는 `cmake<4`가 들어 있으므로, 가상환경 안에서 Marabou를 빌드하면 됩니다.

```bash
cd hw3/Marabou
mkdir build
cd build
cmake ../
cmake --build . -j 4
```

## 5. macOS에서 필요한 코드 수정

macOS 환경에서는 `Engine.cpp`에서 비표준 iterator 타입 때문에 빌드가 실패할 수 있습니다.  
만약 빌드 중 iterator 관련 오류가 나면 아래처럼 수정합니다.

파일:

`Marabou/src/engine/Engine.cpp`

다음 줄을 찾습니다.

```cpp
std::_List_const_iterator<unsigned int> it = entry->lemma->getCausingVars().begin();
```

아래처럼 바꿉니다.

```cpp
List<unsigned>::const_iterator it = entry->lemma->getCausingVars().begin();
```

그 다음 다시 빌드합니다.

```bash
cd hw3/Marabou/build
cmake --build . -j 4
```

## 6. 설치 확인

Marabou가 정상적으로 설치되었는지 확인하려면 예제 하나를 실행합니다.

```bash
cd hw3/Marabou
export PYTHONPATH=$PWD:$PYTHONPATH
cd maraboupy/examples
python -u 1_ONNXExample.py
```

정상이라면 `SAT`, `UNSAT`, 그리고 ONNX evaluation 비교 결과가 출력됩니다.

## 7. 과제 코드 실행

프로젝트 폴더로 돌아와서 `test.py`를 실행합니다.

기본 예시:

```bash
cd hw3
python test.py --epsilon 0.1
```

다른 예시:

```bash
python test.py --epsilon 0.01
python test.py --epsilon 0.2
python test.py --epsilon 0.2 --timeout 20
python test.py --epsilon 0.01 --sample-index 348
```

## 8. 출력 결과 해석

- `UNSAT`: 주어진 `epsilon` 범위 안에서 반례를 찾지 못함
- `SAT`: adversarial input(반례)을 찾음

예를 들어 `epsilon = 0.01`에서 모든 target class가 `UNSAT`이면, 선택한 샘플은 그 범위 안에서 locally robust하다고 해석할 수 있습니다.

반대로 `SAT`가 나오면:

- 콘솔에 adversarial 8x8 입력이 출력되고
- `counterexample.png` 파일이 생성됩니다

이 그림에는 다음 3개가 포함됩니다.

- 원본 이미지
- adversarial 이미지
- 두 이미지의 차이(`difference`)

## 9. 제출 파일

이 과제 제출에 사용한 주요 파일은 아래와 같습니다.

- `requirements.txt`: Python 의존성 목록
- `test.py`: 외부 모델 학습, ONNX 변환, Marabou 검증
- `report.pdf`: 결과 보고서
- `README.md`: 실행 방법
