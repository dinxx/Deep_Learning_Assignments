#!/usr/bin/env python
import numpy as np 

layers_dims = [784, 400, 10] #  2-layer model


class NN:
	def __init__(self, layers_dims = layers_dims, learning_rate = 0.0005, regL2 = 0.01):
		self.learning_rate = learning_rate
		self.regL2 = regL2
		(self.n_x,self.n_h,self.n_y) = layers_dims
		# He Initialization
		self.W1 = np.random.randn(self.n_x,self.n_h)/np.sqrt(self.n_x)
		self.b1 = np.zeros((1,self.n_h))
		self.W2 = np.random.randn(self.n_h,self.n_y)/np.sqrt(self.n_h)
		self.b2 = np.zeros((1,self.n_y))
		# print self.W1.shape,self.b1.shape,self.W2.shape,self.b2.shape

	def ReLU(self,x):
		# x[x<0] = 0
		return np.maximum(0,x)
	
	def d_ReLU(self,x):
		x[x<=0] = 0
		x[x>0] = 1
		return x

	def softmax(self,x):
		exp_ = np.exp(x-np.amax(x))
		return exp_/np.sum(exp_, axis=1, keepdims=True)

	def forward_pass(self,X):
		X = X/255.
		z1 = np.dot(X,self.W1) + self.b1
		a1 = self.ReLU(z1)

		z2 = np.dot(a1,self.W2) + self.b2
		a2 = self.softmax(z2)
		# print z2[:,0],a2[:,0]

		cache = (z1,a1,z2,a2)
		return a2,cache

	def cross_entropy_loss(self,y,y_pred):
		m = y.shape[0]
		logprobs = -np.log(y_pred[range(m),np.argmax(y,axis=1)])
		# print logprobs, np.sum(logprobs)
		regterm = (self.regL2/2)*(np.sum(np.square(self.W1))+np.sum(np.square(self.W2)))
		cost = (1.0/m)*(np.sum(logprobs)+regterm)
		return cost

	def backward_pass(self,X,y,cache):
		(z1,a1,z2,a2) = cache
		# print z1.shape,a1.shape,z2.shape,a2.shape,y.shape
		m = X.shape[0]
		# print a2.shape
		# print m

		dz2 = a2 - y
		dW2 = (1.0/m)*np.dot(a1.T,dz2)
		db2 = (1.0/m)*np.sum(dz2,axis=0,keepdims=True)


		dz1 = np.dot(dz2,self.W2.T)*self.d_ReLU(z1)
		dW1 = (1.0/m)*np.dot(X.T,dz1)
		db1 = (1.0/m)*np.sum(dz1,axis=0,keepdims=True)
		# print dW1.shape,db1.shape,self.W1.shape,self.b1.shape

		self.W1 -= self.learning_rate*(dW1 + (self.regL2)*self.W1/m)
		self.b1 -= self.learning_rate*(db1 + (self.regL2)*self.b1/m)
		self.W2 -= self.learning_rate*(dW2 + (self.regL2)*self.W2/m)
		self.b2 -= self.learning_rate*(db2 + (self.regL2)*self.b2/m)

	def accuracy(self,X_test,y_test):
		count = 0
		y_pred,_ = self.forward_pass(X_test)
		for i in range(X_test.shape[0]):
			index = np.argmax(y_pred[i])
			if index == np.argmax(y_test[i]):
				count +=1
		return count*100/len(X_test)




# N = NN()
# X = np.random.randn(784,7)*20
# y = np.random.randn(10,7)*10
# y_pred,cache = N.forward_pass(X)
# print N.cross_entropy_loss(y,y_pred)
# N.backward_pass(X,y,cache)

















