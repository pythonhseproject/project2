# Image classifier

The program is created to classify images. 

## Prerequisites

The program demands numpy, keras and tensorflow

## How to use

### First model  
The first model uses dense layers. It takes grayscale 28*28 images. It's trained on fashion MNIST dataset. You can run it by

```
python classification.py --input data/classification
```
where `data/classification` is direction with input images and you can change it. The model outputs what is on the image.  
Classes are: 'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'

### Second model
The second model uses mainly convolutional layers. It takes color images. It's trained on cifar10 dataset. You can run it by

```
python cnn.py --input data/cnn
```
where `data/cnn` is direction with input images and you can change it. The model outputs what is on the image.  
Classes are: 'airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'
