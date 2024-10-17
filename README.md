[English](README.md) | [한국어](README_ko.md)


# Image Classification
2024 2nd Semester Visual Intelligence Learning Project


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
├── wide-resnet.ipynb      # Wide ResNet model implementation and training code
├── pyramidnet.ipynb       # PyramidNet model implementation and training code
├── ensemble.ipynb         # Model combination and testing using Ensemble technique
├── requirements.txt       # List of Python packages required for project execution
└── README.md              # Project description and guide
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

    When loading the models, you need to set the model file paths according to your local environment. 

    Modify the following part with `your_path` to the path of the saved model files.

    ```python
    model1_path = "your_path/wide-resnet_best_model.pth"
    model2_path = "your_path/pyramidnet_best_model.pth"

    ```




<br><br>
## Results
||pyramidnet|wide-resnet|Ensemble|
|------|---|---|---|
|Top1_acc|84.69%|82.74%|85.76%|
|Top5_acc|97.19%|96.09%|97.40%|
|Superclass_acc|91.69%|90.19%|92.38%|

<br><br>
## References
ShakeDrop / PyramidNet : https://github.com/dyhan0920/PyramidNet-PyTorch/tree/master, https://github.com/zxcvfd13502/DDAS_code
CIFAR-100 Dataset: [CIFAR-100 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)