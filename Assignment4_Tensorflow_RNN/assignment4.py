#!/usr/bin/env python

import numpy as np
import sys
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

from data_loader import DataLoader 
from modules import NN

load = DataLoader()

# y obtained in one-hot representation from DataLoader
X_train,y_train = load.load_data(mode = 'train')
X_train = X_train.astype(np.float32)
X_test,y_test = load.load_data(mode = 'test') 
X_test = X_test.astype(np.float32)
# X_train, X_val, y_train, y_val = train_test_split(X_train.T, y_train.T, test_size=0.1, random_state=1)
# X_train,y_train = X_train.T,y_train.T 
m = X_train.shape[1]
print X_train.shape,y_train.shape,X_test.shape,y_test.shape,m
# print X_val.shape,y_val.shape


learning_rate = 0.001
regL2 = 0.005
num_epochs = 50
mbatch_sz = 128

# Layer sizes (nh will be taken as a command line input)
nx = 28	
ny = 10

#RNN timesteps
T_x = 28










