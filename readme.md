There are 2 models for image classification. For both you need to give directory path with input images (after --image)  
Run:  
    python classification.py --input data/classification  
or  
    python cnn.py --input data/cnn  

1) Classification by dense layers. Takes grayscale images 28*28  
File name of python script: classification.py  
Dataset: fashion MNIST  
Classes are: 'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'  
To run the script on prepared data you need to run:  
    python classification.py --input data/classification  

2) Classification by convolutional layers. Take color images  
File name of python script: cnn.py  
Dataset: cifar10  
Classes are: 'airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'  
To run the script on prepared data you need to run:  
    python cnn.py --input data/cnn  
