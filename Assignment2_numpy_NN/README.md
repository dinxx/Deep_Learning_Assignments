# Assignment 2 - Cloth Classification Neural Network from scratch (Numpy)

## Task
The task is to build a 2-layer neural network using only numpy and other basic python libraries for a cloth classification task.
Specifics:
1. No of hidden layers : 1
2. Non-linearity in all layers : ReLU
3. Forward pass equation : y = softmax((W​2​.(relu(W​1​.x + b​1​)) + b​2​). It gives the probability that x belongs to a particular class.
4. Loss function: categorical cross entropy
5. Regularization: L2 regularization to prevent overfitting.

## Data
The data is to be stored in a folder with path "../data_Ass23/" with respect to the directory in which the file train.py is located. Please do not extract the data, since the dataloader.py module does exactly that.

### Labels
Each training and test example is assigned to one of the following labels:
0 T-shirt/top	1 Trouser	2 Pullover	3 Dress		4 Coat
5 Sandal	6 Shirt		7 Sneaker	8 Bag		9 Ankle boot

## Implementation
Following modules are built:
1. data_loader.py: Loads all datasets from zip file and creates random shuffled minibatches.
2. module.py: Contains the Neural Network class. Weights initialized using He initialization in the constructor. Forward and Backward pass implemented in numpy.
3. train.py: Implements minibatch gradient descent to train the neural network. 

## Libraries used
numpy
matplotlib
zipfile

## Authors
Kunal Jain
