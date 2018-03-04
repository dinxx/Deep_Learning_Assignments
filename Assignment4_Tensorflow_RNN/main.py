#!/usr/bin/env python

import numpy as np
from matplotlib import pyplot as plt
# from sklearn.model_selection import train_test_split
from modules import RNN
from data_loader import DataLoader 
import argparse

load = DataLoader()

# y obtained in one-hot representation from DataLoader
X_train,y_train = load.load_data(mode = 'train')
X_train = X_train.astype(np.float32)
X_test,y_test = load.load_data(mode = 'test') 
X_test = X_test.astype(np.float32)
# X_train, X_val, y_train, y_val = train_test_split(X_train.T, y_train.T, test_size=0.1, random_state=1)
# X_train,y_train = X_train.T,y_train.T 
m = X_train.shape[0]
print X_train.shape,y_train.shape,X_test.shape,y_test.shape,m
# print X_val.shape,y_val.shape


learning_rate = 0.0008
num_epochs = 50
mbatch_sz = 128

# Layer sizes (nh will be taken as a command line input)
nx = 28	
ny = 10
io_dims = [nx,ny]

#RNN timesteps
T_x = 28

ap = argparse.ArgumentParser()
ap.add_argument("--model",required = True, help = "RNN Cell Type", default = 'lstm')
ap.add_argument("--hidden_unit",required = True, help = "No. of hidden units", default = 32)
ap.add_argument("--train",action = 'store_true', help = "Train RNN")
ap.add_argument("--test",action = 'store_true', help = "Test accuracy for RNN")
args = vars(ap.parse_args())

model = args["model"]
nh = int(args["hidden_unit"])
flag_train = args["train"]
flag_test = args["test"]

print "Training"if(flag_train)else"Testing", "on", model,"cell RNN with %i hidden units--\n"%(nh)

Net = RNN(model = model, nh = nh, Tx = T_x, io_dims = io_dims, learning_rate = learning_rate, epochs = num_epochs, mbatchsz = mbatch_sz)

if flag_train:
	epoch_costs = Net.trainRNNmodel(X_train,y_train)
	# plt.plot(range(num_epochs),epoch_costs,"r-")
	# plt.show()
elif flag_test:
	Net.test_accuracy(X_test,y_test)
else:
	print "Wrong mode. Please run again."







