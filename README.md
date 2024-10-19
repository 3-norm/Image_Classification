<div align="center">
  <a href="README.md">English</a> | <a href="README_ko.md">한국어</a>
</div>
<br><br>
<div align="center">
  <img src="https://github.com/user-attachments/assets/3aa896d4-3f2c-44d3-895b-1936a1ad0c22" alt="image"width="60%">
</div>



## Table of Contents
- [Project Overview](#project-overview)
- [Notes](#notes)
- [Dataset](#dataset)
- [Reproducibility](#reproducibility)
- [How to Run](#how-to-run)
- [Results](#results)
- [References](#references)<br><br>

## Project Overview
This project implements deep learning CNN models ShakePyramidNet and wide-resnet for image classification using the CIFAR-100 dataset.<br><br>


## Notes
In this project, we solve the CIFAR-100 classification problem using PyramidNet and Wide ResNet.

When training the models, training both models simultaneously on the server may exceed the training time of 24 hours. Therefore, to manage training time, please train each model separately. <br><br> 


## Dataset
The CIFAR-100 dataset consists of a total of 60,000 32x32 color images across 100 classes. Each class contains 600 images, divided into 50,000 training images and 10,000 test images. In this project, we use a custom Dataset class to load and preprocess the CIFAR-100 data.

Data Transformations
The data transformations used for training are as follows:


- RandomHorizontalFlip: Randomly flips the image horizontally for data augmentation
- RandomCrop: Randomly crops the image, including padding
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

- CutMix: Randomly mixes two images, exchanging parts of the image and also mixing the labels <br><br>  



## Reproducibility
To ensure reproducibility during model training in this project, we fixed the random seed.

By default, the seed value is set to 327, which allows us to obtain consistent training results in processes such as data sampling and weight initialization.


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

By minimizing inconsistencies caused by random numbers during the training process, we can maintain the reproducibility of the experiments.


### Modifying the Random Seed
In the above function, you can modify the parameter of `set_random_seed()` to the desired seed value.

Change the seed value to the desired value. Example: set_random_seed(42)<br><br>


## How to Run
### Clone the repository.
```python
git clone https://github.com/3-norm/Image_Classification.git
```
#### File Structure
``` 
├── .gitignore             # Specifies files and directories to ignore in version control
├── ensemble.ipynb         # Model combination and testing using Ensemble technique
├── pyramidnet.ipynb       # PyramidNet model implementation and training code
├── README.md              # Project description and guide
├── README_ko.md           # Korean version of the project description and guide
├── requirements.txt       # List of Python packages required for project execution
└── wide-resnet.ipynb      # Wide ResNet model implementation and training code
```
### Install the required packages. You can use the following command to install using the requirements.txt file.

```python
pip install -r requirements.txt
```

#### The CIFAR-100 dataset will be automatically downloaded and loaded when running the code.

### How to Train the Models

1. **Run the Wide ResNet model**
   - Open the `wide-resnet.ipynb` file in Jupyter Notebook and execute the cells sequentially.  
        The random seed is set to 327 by default, but you can change it as needed.
   
2. **Run the PyramidNet model**
   - Open the `pyramidnet.ipynb` file in Jupyter Notebook and execute the cells sequentially.  
        The random seed is set to 327 by default, but you can change it as needed.

3. **Run the Ensemble model**
   - In the `ensemble.ipynb` file, you can combine the individually trained models for final testing.

    **Setting the Model File Paths**

    When loading the models, you need to set the file paths according to your local environment.

    Additionally, after training is complete, you should update the .pth filenames to match the actual names of the generated files.

    In the following code, replace `your_path` with the location where the model files are saved, and update the .pth filenames to the actual generated filenames:

    ```python
    model1_path = "your_path/wide-resnet_best_model.pth"
    model2_path = "your_path/pyramidnet_best_model.pth"

    ```




<br><br>
## Results

### Model Parameters
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

### Ensemble Method
The ensemble model combines WideResNet and ShakePyramidNet using Soft Voting:
- **WideResNet weight**: 0.4
- **ShakePyramidNet weight**: 0.6


### Our Best Score
|               | Top1 Accuracy | Top5 Accuracy | Superclass Accuracy |
|---------------|---------------|----------------|---------------------|
| PyramidNet    | 84.69%        | 97.19%         | 91.69%              |
| WideResNet    | 82.74%        | 96.09%         | 90.19%              |
| **Ensemble**  | **85.76%**    | **97.40%**     | **92.38%**          |

<br><br>
## References
ShakeDrop / PyramidNet : https://github.com/dyhan0920/PyramidNet-PyTorch/tree/master, https://github.com/zxcvfd13502/DDAS_code  
CIFAR-100 dataset: [CIFAR-100 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)