# Reliable and Trustworthy AI - Assignment #1

이 저장소는 Assignment #1을 위한 신경망(MNIST 및 CIFAR-10)에 대한 적대적 공격(FGSM 및 PGD) 알고리즘 구현 코드를 포함하고 있습니다.

## 환경 설정 (Environment Setup)
코드 실행 환경을 동일하게 재현하고 필요한 외부 모듈을 설치하려면 터미널에서 아래 명령어를 실행하세요:

```bash
pip install -r requirements.txt
```

## 코드 실행 방법 (How to Run the Code)

1. MNIST 모델을 훈련하고, Pretrained CIFAR-10 모델을 불러온 뒤, Targeted/Untargeted FGSM/PGD 공격의 성공률을 평가하고 시각화 이미지를 생성하려면 아래 명령어를 실행하세요:
```bash
python test.py
```
참고: 생성된 이미지(original, adversarial attack, perturbation)는 results/ 디렉토리에 자동으로 저장됩니다.
2. ε 값에 따른 Attack Success Rate 표 생성을 위해 다양한 ε 값(0.01, 0.05, 0.1, 0.2, 0.3, 0.5)에 대해 모델을 평가하고, 요약 Table을 출력하려면 아래 명령어를 실행하세요:
```bash
python eps_table.py
```

## 저장소 파일 구조 (Repository Structure)
- mnist.py & cifar10.py: 모델 아키텍처 정의 및 데이터셋 로드.
- fgsm.py & pgd.py: Adversarial attack 알고리즘 구현.
- visualize.py: 원본, 공격받은 이미지, 노이즈를 비교하는 시각화 및 저장 로직.
- test.py: 훈련, 적대적 공격, 시각화를 순차적으로 실행하는 메인 스크립트.
- results/: 시각화된 결과 이미지 파일(PNG)들이 저장되는 디렉토리.
- report.pdf: 적대적 공격 결과와 엡실론 값에 따른 트레이드오프를 분석한 최종 보고서.
