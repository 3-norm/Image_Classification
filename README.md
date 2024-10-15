2024 2학기 시각지능학습 프로젝트: 이미지 분류 모델
PyramidNet과 Wide ResNet을 사용하여 CIFAR-100 이미지 분류 문제를 해결합니다.<br/> 학습 시간 관리 차원에서 각 모델을 개별적으로 학습하는 것이 권장됩니다.

목차
프로젝트 개요
필수 조건
데이터셋
모델 아키텍처
학습 설정
사용법
결과
참고 문헌
프로젝트 개요
이 프로젝트는 ShakePyramidNet 모델을 사용하여 CIFAR-100 데이터셋을 분류하는 딥러닝 CNN 모델입니다. CutMix, MixUp, ShakeDrop 등의 데이터 증강 기법을 활용하여 성능을 높이고자 합니다.<br/><br/>

필수 조건
프로젝트의 필수 라이브러리는 requirements.txt에 명시되어 있습니다. 다음 명령어로 필수 라이브러리를 설치하세요:

bash
코드 복사
pip install -r requirements.txt
<br/>
데이터셋
CIFAR-100 데이터셋은 100개 클래스에 걸쳐 60,000개의 32x32 컬러 이미지로 구성되어 있습니다. 커스텀 Dataset 클래스를 사용하여 데이터를 전처리하고 로드합니다.<br/>

데이터 변환
변환 방법	설명
RandomHorizontalFlip	좌우 반전하여 데이터 증강
RandomCrop	패딩과 함께 이미지 무작위 자르기
CutMix	두 이미지 혼합 및 레이블 결합
<br/>
모델 아키텍처
ShakePyramidNet은 ResNet 계열의 모델로, ShakeDrop과 Residual Connection을 사용하여 성능을 개선합니다.<br/>

Conv2D: 특징 추출<br/>
Batch Normalization: 학습 안정화<br/>
ShakeDrop: 랜덤 드롭<br/>
Fully Connected Layer: 최종 클래스 예측<br/><br/>
학습 설정
Optimizer: SGD (모멘텀과 Nesterov 사용)
Learning Rate Scheduler: ReduceLROnPlateau
Loss Function: CrossEntropyLoss
Epochs: 300
Batch Size: 128
Data Augmentation: CutMix 적용
<br/>
사용법
bash
코드 복사
python train.py
데이터셋 경로가 올바른지 확인하고 실행하십시오.

<br/>
결과
학습이 완료되면 CIFAR-100 테스트셋에서의 성능 평가가 가능합니다. 결과는 정확도와 손실로 나타납니다.<br/><br/>

참고 문헌
ShakeDrop / PyramidNet: PyramidNet GitHub, DDAS Code GitHub
