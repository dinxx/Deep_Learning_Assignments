#!/usr/bin/env python

import numpy as np
import sys
from matplotlib import pyplot as plt

from data_loader import DataLoader 
from modules import NN

load = DataLoader()

# y obtained in one-hot representation from DataLoader
X_train,y_train = load.load_data(mode = 'train')
X_train = X_train.astype(np.float32)
X_test,y_test = load.load_data(mode = 'test') 
X_test = X_test.astype(np.float32)
m = X_train.shape[1]
# print X_train.shape,y_train.shape,X_test.shape,y_test.shape,m


layers_dims = [784, 300, 10]
learning_rate = 0.001
regL2 = 0.005
num_epochs = 50
mbatch_sz = 128

narg=len(sys.argv)
s=(sys.argv)
mode=s[1][2:]
print 'Running Mode: '+ mode

Net = NN(layers_dims,learning_rate,regL2,num_epochs,mbatch_sz)

if mode=='train':
	epoch_costs = Net.trainNNModel(X_train,y_train)
	# plt.plot(range(num_epochs),epoch_costs,"r-")
	# plt.show()
elif mode=='test':
	Net.NNaccuracy(X_test,y_test)
elif mode=='layer=1':
	Net.logistic(X_train,y_train,X_test,y_test,1)
elif mode=='layer=2':
	Net.logistic(X_train,y_train,X_test,y_test,2)
elif mode=='layer=3':
	Net.logistic(X_train,y_train,X_test,y_test,3)
else:
	print "Wrong mode. Please run again."







