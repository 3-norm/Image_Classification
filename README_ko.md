# Image_Classification
2024 2학기 시각지능학습 프로젝트_이미지 분류 모델


## 목차
- [프로젝트 개요](#프로젝트-개요)
- [주의사항](#주의사항)
- [데이터셋](#데이터셋)
- [재현성](#재현성)
- [실행 방법](#실행-방법)
- [결과](#결과)
- [참고 문헌](#참고-문헌)<br><br>

## 프로젝트 개요
이 프로젝트는 CIFAR-100 데이터셋의 이미지를 분류하는 딥러닝 CNN 모델 ShakePyramidNet과 wide-resnet을 통한 앙상블 학습을 구현한 것입니다.<br><br>




## 주의사항
본 프로젝트에서는 피라미드넷 (PyramidNet)과 와이드 레즈넷 (Wide ResNet)을 사용하여 CIFAR-100 분류 문제를 해결합니다.

모델 학습 시, 두 모델을 동시에 서버에서 학습시키면 학습 시간이 24시간을 초과할 수 있습니다. 따라서 학습 시간을 관리하기 위해 각 모델을 개별적으로 학습하십시오. <br><br> 




## 데이터셋
CIFAR-100 데이터셋은 100개의 클래스에 걸쳐 총 60,000개의 32x32 컬러 이미지로 구성되어 있습니다. 각 클래스는 600개의 이미지로 이루어져 있으며, 50,000개의 학습 이미지와 10,000개의 테스트 이미지로 나뉩니다. 이 프로젝트에서는 CIFAR-100 데이터를 로드하고 전처리하는 커스텀 Dataset 클래스를 사용합니다.

데이터 변환
학습에 사용되는 데이터 변환은 다음과 같습니다


- RandomHorizontalFlip: 이미지를 무작위로 좌우 반전하여 데이터 증강
- RandomCrop: 패딩을 포함하여 이미지를 무작위로 자름
```python
transformtrain = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
])

transformtest = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
])

train_dataset = CIFAR100Dataset(train_images, train_labels, transform=transformtrain)
test_dataset = CIFAR100Dataset(test_images, test_labels, transform=transformtest)
```

- CutMix: 두 개의 이미지를 랜덤하게 혼합하여 일부 영역을 교환하고 레이블도 혼합함 <br><br>  






## 재현성
본 프로젝트에서는 딥러닝 모델 학습 시 재현성을 보장하기 위해 난수 시드를 고정하였습니다.

이 프로젝트는 기본적으로 시드 값을 327로 설정하였으며, 이를 통해 데이터 샘플링, 가중치 초기화 등의 과정에서 일관된 학습 결과를 얻을 수 있습니다.


```python
def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_random_seed(327)
```

위 코드를 통해 학습 과정에서 난수로 인해 발생할 수 있는 불일치를 최소화하여 실험의 재현성을 유지할 수 있습니다.


### Random seed 수정
위의 함수에서 원하는 시드 값으로 `set_random_seed()`의 매개변수를 수정하여 사용할 수 있습니다.

시드 값을 원하는 값으로 변경하세요. 예 : set_random_seed(42)<br><br>




## 실행 방법
### 저장소를 클론하세요.
```python
git clone https://github.com/3-norm/Image_Classification.git
```
#### 파일 구조
```
├── .gitignore             # Git에서 제외할 파일과 디렉토리 설정
├── ensemble.ipynb         # 앙상블 기법을 사용한 모델 결합 및 테스트
├── pyramidnet.ipynb       # PyramidNet 모델 구현 및 학습 코드
├── README.md              # 프로젝트 설명 및 가이드
├── README_ko.md           # 한글 버전의 프로젝트 설명 및 가이드
├── requirements.txt       # 프로젝트 실행에 필요한 파이썬 패키지 목록
└── wide-resnet.ipynb      # Wide ResNet 모델 구현 및 학습 코드
```
### 필요한 패키지를 설치합니다. requirements.txt 파일을 이용해 아래 명령어로 설치할 수 있습니다.

```python
pip install -r requirements.txt
```

#### CIFAR-100 데이터셋은 코드 실행 시 자동으로 다운로드 및 로드됩니다.

### 코드 실행 

1. **Wide ResNet 모델 실행**
   - `wide-resnet.ipynb` 파일을 Jupyter Notebook에서 열고, 셀을 순차적으로 실행하세요.  
        랜덤 시드는 기본적으로 327로 설정되어 있으며, 필요에 따라 변경할 수 있습니다.
   
2. **PyramidNet 모델 실행**
   - `pyramidnet.ipynb` 파일을 Jupyter Notebook에서 열고, 셀을 순차적으로 실행하세요.  
        랜덤 시드는 기본적으로 327로 설정되어 있으며, 필요에 따라 변경할 수 있습니다.

3. **Ensemble 모델 실행**
   - `ensemble.ipynb` 파일에서 각각 학습된 모델들을 결합하여 최종 테스트를 진행할 수 있습니다.

    **모델 파일 경로 설정**

    모델을 불러올 때, 각자의 로컬 환경에 맞는 모델 파일 경로를 설정해야 합니다.
   
    또한, 학습이 완료된 후 생성된 .pth 파일 이름으로 해당 경로의 파일명을 수정해야 합니다.
   
    다음과 같은 경로 설정 부분에서 `your_path`를 모델 파일이 저장된 위치로 바꾸고, .pth 파일 이름도 실제 생성된 파일 이름으로 수정하세요
   
    ```python
    model1_path = "your_path/wide-resnet_best_model.pth"
    model2_path = "your_path/pyramidnet_best_model.pth"

    ```




<br><br>
## 결과

### 모델 파라미터
#### PyramidNet
> - **Epochs**: 200
> - **Learning Rate (LR)**: 0.1
> - **Weight Decay**: 5e-4
> - **Momentum**: 0.9
> - **Nesterov**: True
> - **Scheduler**: ReduceLROnPlateau (Patience: 10, Factor: 0.2, Min LR: 1e-6)
>
>
#### WideResNet
> - **Epochs**: 200
> - **Learning Rate (LR)**: 0.1
> - **Weight Decay**: 5e-4
> - **Momentum**: 0.9
> - **Scheduler**: MultiStepLR (Milestones: [60, 120, 160], Gamma: 0.2)

### 앙상블 기법
앙상블 모델은 WideResNet과 ShakePyramidNet을 소프트 보팅(Soft Voting) 방식으로 결합합니다:
- **WideResNet weight**: 0.4
- **ShakePyramidNet weight**: 0.6


### Seed별 성능 비교

#### Random Seed: 327
|               | Top1 Accuracy | Top5 Accuracy | Superclass Accuracy |
|---------------|---------------|----------------|---------------------|
| PyramidNet    | 84.69%        | 97.19%         | 91.69%              |
| WideResNet    | 82.74%        | 96.09%         | 90.19%              |
| **Ensemble**  | **85.76%**    | **97.40%**     | **92.38%**          |


#### Random Seed: 817
|               | Top1 Accuracy | Top5 Accuracy | Superclass Accuracy |
|---------------|---------------|----------------|---------------------|
| PyramidNet    | 84.31%        | 97.37%         | 91.62%              |
| WideResNet    | 82.78%        | 96.18%         | 90.04%              |
| **Ensemble**  | **85.47%**    | **97.55%**     | **92.18%**          |

#### 최고 점수 : Random Seed 112
|               | Top1 Accuracy | Top5 Accuracy | Superclass Accuracy |
|---------------|---------------|----------------|---------------------|
| PyramidNet    | 84.44%        | 97.30%         | 92.46%              |
| WideResNet    | 82.58%        | 95.89%         | 90.14%              |
| **Ensemble**  | <mark>**85.71%**</mark>    | <mark>**97.45%**</mark>     | <mark>**92.46%**</mark>          |

<br><br>
## 참고 문헌
ShakeDrop / PyramidNet : https://github.com/dyhan0920/PyramidNet-PyTorch/tree/master, https://github.com/zxcvfd13502/DDAS_code
CIFAR-100 데이터셋: [CIFAR-100 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)
