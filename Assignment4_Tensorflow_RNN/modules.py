#!/usr/bin/env python
import numpy as np 
import tensorflow as tf

from tensorflow.contrib.rnn import RNNCell,static_rnn

from sklearn.metrics import accuracy_score
from data_loader import DataLoader

ip_dims = [28, 10] #  4-layer model (nx nh nh nh ny)

class GRU_Cell(RNNCell):
	def __init__(self, nh = 32, ip_dims = ip_dims):
		super(GRU_Cell, self).__init__()
		self.n_h = nh
		(self.n_x,self.n_y) = ip_dims

		tf.set_random_seed(1) 

		# Gate weights
		self.W = tf.get_variable("Wg",[self.n_h+self.n_x,2*self.n_h],initializer = tf.contrib.layers.xavier_initializer(seed=1))
		self.b = tf.get_variable("bg",[2*self.n_h],initializer = tf.zeros_initializer)
		# New activation weights
		self.Wc = tf.get_variable("Wc",[self.n_h+self.n_x,self.n_h],initializer = tf.contrib.layers.xavier_initializer(seed=1))
		self.bc = tf.get_variable("bc",[self.n_h],initializer = tf.zeros_initializer)


	def _concat(self,x,h):
		return tf.concat([x,h],1)

	def __call__(self, x, h_prev, scope = None):
		with tf.variable_scope(scope or type(self).__name__):
			# u = input_gate, f = forget_gate, o = output_gate
			gates = tf.sigmoid(tf.add(tf.matmul(self._concat(h_prev,x),self.W),self.b))
			z,r = tf.split(gates, num_or_size_splits = 2, axis = 1)

			h_cell = tf.tanh(tf.add(tf.matmul(self._concat(tf.multiply(r,h_prev),x),self.Wc),self.bc))
			h_next = tf.add(tf.multiply(1-z,h_prev), tf.multiply(z,h_cell))
		return h_next, h_next

	@property
	def state_size(self):
		return self.n_h

	@property
	def output_size(self):
		return self.n_h


class LSTM_Cell(RNNCell):
	def __init__(self, nh = 32, ip_dims = ip_dims):
		super(LSTM_Cell, self).__init__()

		self.n_h = nh
		(self.n_x,self.n_y) = ip_dims

		tf.set_random_seed(1)

		# Gate weights
		self.W = tf.get_variable("W",[self.n_h+self.n_x,4*self.n_h],initializer = tf.contrib.layers.xavier_initializer(seed=1))
		self.b = tf.get_variable("b",[4*self.n_h],initializer = tf.zeros_initializer)

	def _concat(self,x,h):
		return tf.concat([x,h],1)

	def __call__(self, x, state, scope = None):
		with tf.variable_scope(scope or type(self).__name__):
			c_prev,h_prev = state
			x = self._concat(x,h_prev)

			# u = input_gate, f = forget_gate, o = output_gate
			gates = tf.add(tf.matmul(x,self.W),self.b)
			u,f,o,c = tf.split(gates, num_or_size_splits = 4, axis = 1)

			c_cell = tf.tanh(c)
			c_next = tf.add(tf.multiply(tf.sigmoid(f),c_prev), tf.multiply(tf.sigmoid(u),c_cell))

			h_next = tf.multiply(tf.sigmoid(o),tf.tanh(c_next))
		# return h_next, self._concat(c_next,h_next)
		return h_next, (c_next,h_next)

	@property
	def state_size(self):
		return (self.n_h,self.n_h)

	@property
	def output_size(self):
		return self.n_h


class RNN(object):
	def __init__(self, mode = 'LSTM', nh =32, Tx = 28, ip_dims = ip_dims, learning_rate = 0.0005, epochs = 50, mbatchsz = 128):
		self.learning_rate = learning_rate
		self.epochs = epochs
		self.mbatchsz = mbatchsz
		self.Tx = Tx
		self.mode = mode

		self.n_h = nh
		(self.n_x,self.n_y) = ip_dims

		self.X,self.y = self.create_placeholders(self.n_x*self.n_x,self.n_y)
		#check
		self.keep_prob = tf.placeholder(tf.float32)

		self.input = tf.reshape(self.X,[-1,self.Tx,self.n_x])

		tf.set_random_seed(1)

		# Softmax weights
		self.Wo = tf.get_variable("Wo",[self.n_h,self.n_y],initializer = tf.contrib.layers.xavier_initializer(seed=1))
		self.bo = tf.get_variable("bo",[self.n_y],initializer = tf.zeros_initializer)

		# Prepare data shape to match `rnn` function requirements
		# Current data input shape: (batch_size, timesteps, n_input)
		# Required shape: 'timesteps' tensors list of shape (batch_size, n_input)
		# Unstack to get a list of 'timesteps' tensors of shape (batch_size, n_input)
		self.x = tf.unstack(self.input,self.Tx,1)

		if mode == 'LSTM':
			self.cell = LSTM_Cell(self.n_h,ip_dims)
		elif mode == 'GRU':
			self.cell = GRU_Cell(self.n_h,ip_dims)

		outputs,states = static_rnn(self.cell,self.x,dtype = tf.float32)
		self.logits = tf.add(tf.matmul(outputs[-1],self.Wo),self.bo)
		self.pred = tf.nn.softmax(self.logits)

	def trainRNNModel(self,X_train,y_train):
		tf.set_random_seed(1)
		costs = []
		load = DataLoader()
		m = X_train.shape[0]

		cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits = self.logits, labels = self.y))
		optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(cost)

		correct_pred = tf.equal(tf.argmax(self.pred,1),tf.argmax(self.y,1))
		accuracy = tf.reduce_mean(tf.cast(correct_pred,tf.float32))

		seed = 1

		with tf.Session() as sess:
			sess.run(tf.global_variables_initializer())
			saver = tf.train.Saver()

			for epoch in range(self.epochs):
				epoch_cost = 0.
				n_mb = int(m/self.mbatchsz)
				seed += 1
				minibatches = load.create_batches(X_train,y_train,n_mb,seed,self.mbatchsz)

				for minibatch in minibatches:
					mbX,mby = minibatch
					_,mb_Cost = sess.run([optimizer,cost], feed_dict = {self.X: mbX, self.y: mby})
					epoch_cost += mb_Cost/n_mb

				print "Cost after epoch %i: %f" % (epoch, epoch_cost)
				costs.append(epoch_cost)

			finalCost,trainAcc = sess.run([cost,accuracy], feed_dict = {self.X: mbX, self.y: mby})
			print "Final training cost after epoch %i: %f"%(self.epochs,finalCost)
			print "Train Accuracy for ", self.mode, " recurrent neural network: ", trainAcc
			savepath = saver.save(sess,"./weights/weights.cpkt")

		return costs

	def create_placeholders(self,nx,ny):
		"""
		creates placeholders for current session
		Args:
		input output layer sizes
		Returns:
		X -- placeholder for the data input, of shape [n_x, None] and dtype "float"
		Y -- placeholder for the input labels, of shape [n_y, None] and dtype "float"
		"""
		X = tf.placeholder(tf.float32, shape = [None,nx], name = 'X')
		y = tf.placeholder(tf.float32, shape = [None,ny], name = 'y')
		return X,y

	def test_accuracy(self,X_test,y_test):		
		saver = tf.train.Saver()

		with tf.Session() as sess:
			saver.restore(sess,"./weights/weights.cpkt")
			count = tf.equal(tf.argmax(self.pred,1), tf.argmax(self.y,1))
			accuracy = tf.reduce_mean(tf.cast(count,tf.float32))
			print "Test Accuracy: ", accuracy.eval({self.X: X_test, self.y: y_test})


N = RNN(mode = 'GRU', nh =128)
L = DataLoader()

X_train,y_train = L.load_data(mode = 'train')
X_train = X_train.astype(np.float32)
X_test,y_test = L.load_data(mode = 'test')
X_test = X_test.astype(np.float32)
m = X_train.shape[0]
print X_train.shape,y_train.shape,X_test.shape,y_test.shape,m

_ = N.trainRNNModel(X_train,y_train)
N.test_accuracy(X_test,y_test)













