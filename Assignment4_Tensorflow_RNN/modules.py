#!/usr/bin/env python
import numpy as np 
import tensorflow as tf
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from data_loader import DataLoader

ip_dims = [28, 10] #  4-layer model (nx nh nh nh ny)


class LSTMCell:
	def __init__(self, nh = 32, Tx = 28, ip_dims = ip_dims, learning_rate = 0.001, regL2 = 0.005, epochs = 50, mbatchsz = 128):
		self.learning_rate = learning_rate
		self.epochs = epochs
		self.mbatchsz = mbatchsz
		self.regL2 = regL2
		self.Tx = Tx

		self.n_h = nh
		(self.n_x,self.n_y) = ip_dims

		tf.set_random_seed(1) 

		self.X,self.y = self.create_placeholders(self.n_x,self.n_y)

		# New Cell state weight
		self.Wc = tf.get_variable("Wc",[self.n_h,self.n_h+self.n_x],initializer = tf.contrib.layers.Xavier_initializer(seed=1))
		self.bc = tf.get_variable("bc",[self.n_h,1],initializer = tf.zeros_initializer)
		# Update gate weight
		self.Wu = tf.get_variable("Wu",[self.n_h,self.n_h+self.n_x],initializer = tf.contrib.layers.Xavier_initializer(seed=1))
		self.bu = tf.get_variable("bu",[self.n_h,1],initializer = tf.zeros_initializer)
		# Forget gate weight
		self.Wf = tf.get_variable("Wf",[self.n_h,self.n_h+self.n_x],initializer = tf.contrib.layers.Xavier_initializer(seed=1))
		self.bf = tf.get_variable("bf",[self.n_h,1],initializer = tf.zeros_initializer)
		# Output gate weight
		self.Wo = tf.get_variable("Wo",[self.n_h,self.n_h+self.n_x],initializer = tf.contrib.layers.Xavier_initializer(seed=1))
		self.bo = tf.get_variable("bo",[self.n_h,1],initializer = tf.zeros_initializer)
		
	def create_placeholders(self,nx,ny):
		"""
		creates placeholders for current session
		Args:
		input output layer sizes
		Returns:
		X -- placeholder for the data input, of shape [n_x, None] and dtype "float"
		Y -- placeholder for the input labels, of shape [n_y, None] and dtype "float"
		"""
		X = tf.placeholder(tf.float32, shape = [nx,None], name = 'X')
		y = tf.placeholder(tf.float32, shape = [ny,None], name = 'y')
		return X,y

	def _concat(h,x):
		m = h.shape[1]
		concat = tf.concat([h,x],0)
		return concat

	def forward_pass(self):


	def cross_entropy_loss(self,y_pred):




def NNaccuracy(self,X_test,y_test):		

	y_pd,_ = self.forward_pass()
	saver = tf.train.Saver()

	with tf.Session() as sess:
		saver.restore(sess,"./weights/weights.cpkt")
		count = tf.equal(tf.argmax(y_pd), tf.argmax(self.y))
		accuracy = tf.reduce_mean(tf.cast(count,"float"))
		print "Test Accuracy for 4 layer neural network: ", accuracy.eval({self.X: X_test, self.y: y_test})


# N = NN()
# L = DataLoader()

# X_train,y_train = L.load_data(mode = 'train')
# X_train = X_train.astype(np.float32)
# X_test,y_test = L.load_data(mode = 'test')
# X_test = X_test.astype(np.float32)
# m = X_train.shape[1]
# # print X_train.shape,y_train.shape,X_test.shape,y_test.shape,m

# _ = N.trainNNModel(X_train,y_train)
# N.NNaccuracy(X_test,y_test)














