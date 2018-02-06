#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from data_loader import DataLoader
from module import NN

load = DataLoader()

X_train,y_train = load.load_data(mode = 'train')
X_train = X_train.astype(np.float32)
X_test,y_test = load.load_data(mode = 'test')
X_test = X_test.astype(np.float32)
# print X_train.shape,y_train.shape,X_test.shape,y_test.shape
m = X_train.shape[0]

num_epochs = 50
mbatch_sz = 128
num_batches = int(m/mbatch_sz)
epoch_seed = 1

layers_dims = [784, 40, 10] #  2-layer model
learning_rate = 0.0001
regL2 = 0.0001
np.random.seed(100)
Net = NN(layers_dims,learning_rate,regL2)



costs = []
epochs = []
testaccs = []

for epoch in range(num_epochs):
	epoch_seed += 1
	epochcost = 0
	minibatches = load.create_batches(X_train,y_train,num_batches,epoch_seed,mbatch_sz)
	for minibatch in minibatches:
		mbX,mby = minibatch

		y_pred,cache = Net.forward_pass(mbX)
		# print y_pred
		# print mbX.shape,mby.shape,y_pred.shape

		epochcost += Net.cross_entropy_loss(mby,y_pred)

		Net.backward_pass(mbX,mby,cache)

	# if epoch%100 == 0:
	epochs.append(epoch)
	costs.append(epochcost/num_batches)

	print "Cost for epoch "+str(epoch)+" ==> "+ str(epochcost/num_batches)

	testacc = Net.accuracy(X_test,y_test)
	testaccs.append(testacc)
	print "Accuracy for epoch "+str(epoch)+" ==> "+ str(testacc)

plt.plot(epochs,costs,'ro')
plt.show()

accuracy = Net.accuracy(X_test,y_test)
print 'Final test accuracy: ' + str(accuracy)

with open("accuracy.txt","w") as outfile :
	outfile.write("The final accuracy is : "+str(accuracy))











# Net = NN()







