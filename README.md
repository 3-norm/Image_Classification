## Image_Classification
2024 2학기 시각지능학습 프로젝트_이미지 분류 모델

## 주의사항
본 프로젝트에서는 피라미드넷 (PyramidNet)과 와이드 레즈넷 (Wide ResNet)을 사용하여 CIFAR-100 분류 문제를 해결합니다.

모델 학습 시, 두 모델을 동시에 서버에서 학습시키면 학습 시간이 24시간을 초과할 수 있습니다. 따라서 학습 시간을 관리하기 위해 각 모델을 개별적으로 학습하는 것을 권장합니다.

## 목차
- [프로젝트 개요](#프로젝트-개요)
- [필수 조건](#필수-조건)
- [데이터셋](#데이터셋)
- [모델 아키텍처](#모델-아키텍처)
- [학습 설정](#학습-설정)
- [사용법](#사용법)
- [결과](#결과)
- [참고 문헌](#참고-문헌)

## 프로젝트 개요
이 프로젝트는 CIFAR-100 데이터셋의 이미지를 분류하는 딥러닝 CNN 모델 ShakePyramidNet과 wide-resnet을 통한 앙상블 학습을 구현한 것입니다. 

ShakePyramidNet과 모델은 CutMix, MixUp, ShakeDrop 등의 기술을 활용하여 일반화 및 성능을 향상시킵니다.



## 필수 조건
이 프로젝트의 종속 라이브러리는 requirements.txt 파일에 정의되어 있습니다. 다음 명령어를 사용하여 모든 필수 라이브러리를 설치할 수 있습니다.
pip install -r requirements.txt



## 데이터셋
CIFAR-100 데이터셋은 100개의 클래스에 걸쳐 총 60,000개의 32x32 컬러 이미지로 구성되어 있습니다. 각 클래스는 600개의 이미지로 이루어져 있으며, 50,000개의 학습 이미지와 10,000개의 테스트 이미지로 나뉩니다. 이 프로젝트에서는 CIFAR-100 데이터를 로드하고 전처리하는 커스텀 Dataset 클래스를 사용합니다.

데이터 변환
학습에 사용되는 데이터 변환은 다음과 같습니다


- RandomHorizontalFlip: 이미지를 무작위로 좌우 반전하여 데이터 증강
- RandomCrop: 패딩을 포함하여 이미지를 무작위로 자름
- CutMix: 두 개의 이미지를 랜덤하게 혼합하여 일부 영역을 교환하고 레이블도 혼합함



## 모델 아키텍처
ShakePyramidNet은 ResNet 계열의 모델로, ShakeDrop과 Residual Connection을 사용하여 성능을 향상시킵니다. 

모델은 다음과 같은 레이어들로 구성됩니다.
- Conv2D: 입력 이미지에서 특징을 추출하는 합성곱 레이어
- Batch Normalization: 학습을 안정화하고 속도를 향상시키는 배치 정규화 레이어
- ShakeDrop: 학습 중에 랜덤으로 특정 활성화를 드롭하는 기술로, 정규화 효과를 극대화
- Fully Connected Layer: 클래스 예측을 위한 최종 출력 레이어



## 학습 설정
- ShakePyramidNet의 학습은 다음과 같은 설정으로 이루어집니다
- Optimizer: SGD(모멘텀 및 Nesterov 사용)
- Learning Rate Scheduler: ReduceLROnPlateau, 성능이 개선되지 않을 때 학습률을 감소
- Loss Function: CrossEntropyLoss
- Epochs: 300회
- Batch Size: 128
- CutMix와 MixUp 데이터 증강 기법을 선택적으로 적용



## 사용법
설치
먼저, 저장소를 클론하세요.
```python
git clone https://github.com/3-norm/Image_Classification.git
```

필요한 패키지를 설치합니다. requirements.txt 파일을 이용해 아래 명령어로 설치할 수 있습니다.

```python
pip install -r requirements.txt
```


서버 접속 및 JupyterLab 실행
개인적으로 부여받은 서버 주소로 접속합니다.
서버에 접속한 후, clone받은 .ipynb 파일을 엽니다.
작업을 각 셀에서 순차적으로 실행하세요.


## 결과



## 참고 문헌
ShakeDrop / PyramidNet : https://github.com/dyhan0920/PyramidNet-PyTorch/tree/master, https://github.com/zxcvfd13502/DDAS_code
CIFAR-100 데이터셋: [CIFAR-100 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)
