# Assignment 3 - Cloth Classification Neural Network using Tensorflow

## Task
1. The task is to build a 4-layer neural network using Tensorflow library for a cloth classification task. Use ReLU non-linearity in all layers.
2. Once the network is trained, get the activations from all three hidden layers, and use them individually as features to train a logistic regression network defined using the scikit-learn library.
3. Report results on test set for both the DNN and the 3 logistic regression networks.
4. Do not use tf.nn, tf.layers or tf.contrib.layers. Define own functions for activations and other functions.

## Data
The data is the same as Assignment 2. Again, it is to be stored in a folder with path "../data_Ass23" with respect to the directory in which the file train.py is located. Please do not extract the data, since the dataloader.py module does exactly that.

### Labels
Each training and test example is assigned to one of the following labels:
0 T-shirt/top	1 Trouser	2 Pullover	3 Dress		4 Coat
5 Sandal	6 Shirt		7 Sneaker	8 Bag		9 Ankle boot

## Implementation
Following modules are built:
1. data_loader.py: Loads all datasets from zip file and creates random shuffled minibatches.
2. modules.py: Contains the Neural Network class. All parameters are defined as tf.Variable and initialized using Xavier initialization in the constructor. Forward pass is defined using basic tf functions like matmul. Softmax loss is implemented as the cost function. tf.train.AdamOptimizer() is used as the optimizer. Logistic regression function is defined.
3. assignment3.py: main file for the assignment. Searches for command line arguments so as to decide which operation to perform.

## Usage
1. python assignment3.py --train: Trains the model and reports final cost, learning curve and train set accuracy.
2. python assignment3.py --test: Tests the model using weights saved from the training.
3. python assignment3.py --layer=1: Gets activation of the 1st layer of the trained network and trains a logistic regression network with the given outputs.
4. python assignment3.py --layer=2: Gets activation of the 2nd layer of the trained network and trains a logistic regression network with the given outputs.
5. python assignment3.py --layer=3: Gets activation of the 3rd layer of the trained network and trains a logistic regression network with the given outputs.

## Report
model_parameter_explanation.txt gives a summary of how the network was tuned and what accuracies were obtained.

## Libraries used
numpy
matplotlib
tensorflow
scikit-learn

## Authors
Kunal Jain
