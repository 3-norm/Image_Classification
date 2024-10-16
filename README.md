## Image_Classification
2024 2학기 시각지능학습 프로젝트_이미지 분류 모델


## 목차
- [프로젝트 개요](#프로젝트-개요)
- [주의사항](#주의사항)
- [데이터셋](#데이터셋)
- [모델 아키텍처](#모델-아키텍처)
- [학습 설정](#학습-설정)
- [사용법](#사용법)
- [결과](#결과)
- [참고 문헌](#참고-문헌)

## 프로젝트 개요
이 프로젝트는 CIFAR-100 데이터셋의 이미지를 분류하는 딥러닝 CNN 모델 ShakePyramidNet과 wide-resnet을 통한 앙상블 학습을 구현한 것입니다. 



## 주의사항
본 프로젝트에서는 피라미드넷 (PyramidNet)과 와이드 레즈넷 (Wide ResNet)을 사용하여 CIFAR-100 분류 문제를 해결합니다.

모델 학습 시, 두 모델을 동시에 서버에서 학습시키면 학습 시간이 24시간을 초과할 수 있습니다. 따라서 학습 시간을 관리하기 위해 각 모델을 개별적으로 학습하십시오.



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

- CutMix: 두 개의 이미지를 랜덤하게 혼합하여 일부 영역을 교환하고 레이블도 혼합함
```python
def cutmix_data(x, y, alpha=1.0):
    lam = np.random.beta(alpha, alpha)
    rand_index = torch.randperm(x.size()[0]).to(x.device)
    target_a = y
    target_b = y[rand_index]
    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    x[:, :, bbx1:bbx2, bby1:bby2] = x[rand_index, :, bbx1:bbx2, bby1:bby2]
    return x, target_a, target_b, lam

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    return bbx1, bby1, bbx2, bby2
```


## 모델 아키텍처
ShakePyramidNet은 ResNet 계열의 모델로, ShakeDrop과 Residual Connection을 사용하여 성능을 향상시킵니다. 

모델은 다음과 같은 레이어들로 구성됩니다.
- Conv2D: 입력 이미지에서 특징을 추출하는 합성곱 레이어
- Batch Normalization: 학습을 안정화하고 속도를 향상시키는 배치 정규화 레이어
- ShakeDrop: 학습 중에 랜덤으로 특정 활성화를 드롭하는 기술로, 정규화 효과를 극대화
- Fully Connected Layer: 클래스 예측을 위한 최종 출력 레이어



## 학습 설정
### ShakePyramidNet
- Batch Size: 128
```python
train_loader = DataLoader(train_data, batch_size=128, shuffle=True)
test_loader = DataLoader(test_data, batch_size=128, shuffle=False)
```
- CutMix와 MixUp 데이터 증강 기법을 선택적으로 적용 해당 코드에서는 cutmix만 사용하였습니다.
```python
train(model, train_loader, optimizer, criterion, device, use_cutmix=True)
```

- 하이퍼파라미터
```python
config = {
    'epoch': 200,
    'lr': 0.1,
    'weight_decay': 5e-4,
    'momentum': 0.9,
    "nesterov": True,
    'patience': 10,
    'factor': 0.2,   
    'min_lr': 1e-6   
}

optimizer = optim.SGD(
    model.parameters(),
    lr=config['lr'],
    weight_decay=config['weight_decay'],
    momentum=config['momentum'],
    nesterov=config["nesterov"]
)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=config['factor'],
    patience=config['patience'],
    min_lr=config['min_lr']
)
```



## 사용법
### 설치
1. 저장소를 클론하세요.
```python
git clone https://github.com/3-norm/Image_Classification.git
```

필요한 패키지를 설치합니다. requirements.txt 파일을 이용해 아래 명령어로 설치할 수 있습니다.

```python
pip install -r requirements.txt
```

### 코드 실행과정
2. 데이터셋 준비 및 전처리
CIFAR-100 데이터셋을 자동으로 다운로드하고 로드합니다.
데이터셋은 학습과 테스트 셋으로 나누어져 있으며, 각각 데이터 증강을 위해 RandomHorizontalFlip과 RandomCrop을 포함한 변환을 적용합니다.
```python
# CIFAR-100 데이터셋 불러오기
train_data = datasets.CIFAR100(root='./data', train=True, download=True, transform=transformtrain)
test_data = datasets.CIFAR100(root='./data', train=False, download=True, transform=transformtest)

train_loader = DataLoader(train_data, batch_size=128, shuffle=True)
test_loader = DataLoader(test_data, batch_size=128, shuffle=False)
```

### 모델 구성
3. 모델 실행

* <mark>ShakeDropFunction</mark>

  학습 중 일부 활성화를 무작위로 드롭하여 정규화 효과를 극대화합니다.

* <mark>ShakeDrop</mark>

  ShakeDropFunction을 포함하는 클래스이며, 특정 확률로 드롭아웃을 적용합니다.

* <mark>ShakeBasicBlock</mark>

  ResNet 블록에 ShakeDrop 기법을 적용한 ShakePyramidNet의 기본 블록입니다.

* <mark>ShakePyramidNet</mark>
  ShakeDrop 기법을 사용한 피라미드 네트워크입니다. 모델 깊이에 따라 채널 수를 점진적으로 증가시키며 Residual Connection을 활용합니다.

### 학습 및 평가
```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ShakePyramidNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4, nesterov=True)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=10, min_lr=1e-6)

for epoch in range(300):
    train_loss, train_acc = train(model, train_loader, optimizer, criterion, device, use_cutmix=True)
    val_loss, val_top1_acc, val_top5_acc, val_super_class_acc = evaluate(model, test_loader, criterion, device)
    print(f'Epoch {epoch+1}: Loss={val_loss:.4f}, Top-1 Accuracy={val_top1_acc:.2f}%, Top-5 Accuracy={val_top5_acc:.2f}%')
```


## 결과
||pyramidnet|wide-resnet|Ensemble|
|------|---|---|---|
|Top1_acc|%|%|%|
|Top5_acc|%|%|%|
|Superclass_acc|%|%|%|

## 참고 문헌
ShakeDrop / PyramidNet : https://github.com/dyhan0920/PyramidNet-PyTorch/tree/master, https://github.com/zxcvfd13502/DDAS_code
CIFAR-100 데이터셋: [CIFAR-100 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)
