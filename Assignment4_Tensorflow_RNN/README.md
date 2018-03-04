# Assignment 4 - Cloth Classification Recurrent Neural Network (LSTM/GRU)

## Task
1. The task is to build a recurrent neural network using Tensorflow library for a cloth classification task. LSTM and GRU cells have to be custom designed, and not used directly from tf.contrib.rnn.
2. The network is trained, and the weights are saved for both the LSTM and GRU RNNs for hidden layer sizes 32/64/128/256.
3. Report results on test set for all possible cases and compare with results from Assignment 3.

## Data
The data is the same as Assignment 2/3. Again, it is to be stored in a folder with path "../data_Ass23" with respect to the directory in which the file train.py is located. Please do not extract the data, since the dataloader.py module does exactly that.

### Labels
Each training and test example is assigned to one of the following labels:
0 T-shirt/top	1 Trouser	2 Pullover	3 Dress		4 Coat
5 Sandal	6 Shirt		7 Sneaker	8 Bag		9 Ankle boot

## Implementation
Following modules are built:
1. data_loader.py: Loads all datasets from zip file and creates random shuffled minibatches.
2. modules.py: Contains the Recurrent Neural Network class. All parameters are defined as tf.Variable and initialized using Xavier initialization in the constructor. Basic tf functions like matmul, add, tanh, sigmoid and softmax are used to calculate the states. Softmax Cross Entropy loss is take as the cost function. tf.train.AdamOptimizer() is used as the optimizer. Two custom cells are built, corresponding to the basic LSTM and GRU cells without peephole connections.
__The custom cells are implemented as an extension of the RNNCell class in tf.contrib.rnn, which means, overriding atleast the state\_size @property (tuple with the lengths of whichever states you’re keeping track of), output\_size @property (length of the output of the cell) OR the \_\_call\_\_ method.__ The call method accepts parameters input and state and returns the hidden state along with all the states (only hidden) OR (hidden+cell state as a tuple).
3. main.py: main file for the assignment. Parses command line arguments so as to decide which operation to perform.

## Usage
python main.py (--train/--test) --hidden_unit=(32/64/128/256) --model=('lstm'/'gru')

1. --hidden_unit: Takes an integer corresponding to the number of hidden units in the cell. (32/64/128/256/others)
2. --model: Takes the type of RNN Cell used. Possible: 'lstm' or 'gru'
3. --train: Trains the model over the above chosen conditions and saves weights.
4. --test: Tests model over the above chosen conditions with the saved weights from (3).

## Report
model_accuracy.txt gives a summary of how the network accuracies depended on the RNN Cell and hidden unit size.

## Libraries used
numpy
matplotlib
tensorflow
argparse
zipfile

## Authors
Kunal Jain
