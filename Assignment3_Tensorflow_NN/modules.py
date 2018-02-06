#!/usr/bin/env python
import numpy as np 
import tensorflow as tf
from tensorflow.python.framework import ops
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from data_loader import DataLoader

layers_dims = [784, 300, 10] #  4-layer model (nx nh nh nh ny)


class NN:
	def __init__(self, layers_dims = layers_dims, learning_rate = 0.001, regL2 = 0.005, epochs = 50, mbatchsz = 128):
		self.learning_rate = learning_rate
		self.epochs = epochs
		self.mbatchsz = mbatchsz
		self.regL2 = regL2

		(self.n_x,self.n_h,self.n_y) = layers_dims

		tf.set_random_seed(1) 

		self.X,self.y = self.create_placeholders(self.n_x,self.n_y)

		# He Initialization		
		self.W1 = tf.Variable(tf.random_uniform((self.n_h,self.n_x),-1,1), name="W1")/np.sqrt(self.n_x)
		self.b1 = tf.Variable(tf.zeros((self.n_h,1)), name="b1")
		self.W2 = tf.Variable(tf.random_uniform((self.n_h,self.n_h),-1,1), name="W2")/np.sqrt(self.n_h)
		self.b2 = tf.Variable(tf.zeros((self.n_h,1)), name="b2")
		self.W3 = tf.Variable(tf.random_uniform((self.n_h,self.n_h),-1,1), name="W3")/np.sqrt(self.n_h)
		self.b3 = tf.Variable(tf.zeros((self.n_h,1)), name="b3")
		self.W4 = tf.Variable(tf.random_uniform((self.n_y,self.n_h),-1,1), name="W4")/np.sqrt(self.n_h)
		self.b4 = tf.Variable(tf.zeros((self.n_y,1)), name="b4")
		# print self.W1.shape,self.b1.shape,self.W2.shape,self.b2.shape

	def ReLU(self,z):
		res = tf.maximum(z,0.)
		return res

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

	def forward_pass(self):
		"""
		Forward Propagation: X -> LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> RELU -> LINEAR ->SOFTMAX
		Returns:
		z4 -- Output of the last LINEAR unit
		a_cache -- Activation cache of the 3 hidden layers
		"""

		# H1
		z1 = tf.matmul(self.W1,self.X)+self.b1
		a1 = self.ReLU(z1)

		# H2
		z2 = tf.matmul(self.W2,a1)+self.b2
		a2 = self.ReLU(z2)

		# H3
		z3 = tf.matmul(self.W3,a2)+self.b3
		a3 = self.ReLU(z3)

		# Output
		z4 = tf.matmul(self.W4,a3)+self.b4

		a_cache = (a1,a2,a3)
		return z4,a_cache

	def cross_entropy_loss(self,y_pred):
		"""
		Computes the cost
		Args:
		y_pred -- output of forward propagation (z4)
		Returns:
		cost - Tensor of the cost function
		"""
		# print str(y_pred)
		logits = tf.transpose(y_pred)
		labels = tf.transpose(self.y)
		# print str(logits), str(labels)
		loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits = logits, labels = labels) + 
			self.regL2*(tf.nn.l2_loss(self.W1) + tf.nn.l2_loss(self.W2) + tf.nn.l2_loss(self.W3) + tf.nn.l2_loss(self.W4)))
		
		return loss

	def trainNNModel(self, X_train, y_train):
		"""
		Trains a four-layer tensorflow neural network: X->LINEAR->RELU->LINEAR->RELU->LINEAR->RELU->LINEAR->SOFTMAX.
		Args:
		X_train -- training set data
		y_train -- training set labels
		Returns:
		cost -- Final cost after training
		"""
		# ops.reset_default_graph()  # to be able to rerun the model without overwriting tf variables
		tf.set_random_seed(1)
		costs = []
		load = DataLoader()

		(n_x,m) = X_train.shape
		n_y = y_train.shape[0]

		# X,y = self.create_placeholders(n_x,n_y)

		y_pred,_ = self.forward_pass()
		cost = self.cross_entropy_loss(y_pred)
		# cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits = tf.transpose(y_pred), labels = tf.transpose(self.y)))

		optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(cost)

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

			finalCost = costs[-1]
			yp,_ = self.forward_pass()
			count = tf.equal(tf.argmax(yp), tf.argmax(self.y))
			accuracy = tf.reduce_mean(tf.cast(count,"float"))
			trainAcc = accuracy.eval({self.X: X_train, self.y: y_train})
			print "Final training cost after epoch %i: %f"%(self.epochs,finalCost)
			print "Train Accuracy for 4 layer neural network: ", trainAcc
			savepath = saver.save(sess,"./weights/weights.cpkt")

		return costs

	def NNaccuracy(self,X_test,y_test):		

		y_pd,_ = self.forward_pass()
		saver = tf.train.Saver()

		with tf.Session() as sess:
			saver.restore(sess,"./weights/weights.cpkt")
			count = tf.equal(tf.argmax(y_pd), tf.argmax(self.y))
			accuracy = tf.reduce_mean(tf.cast(count,"float"))
			print "Test Accuracy for 4 layer neural network: ", accuracy.eval({self.X: X_test, self.y: y_test})

	def logistic(self,X_train,y_train,X_test,y_test,L=1):

		_,a_cache = self.forward_pass()
		saver = tf.train.Saver()

		with tf.Session() as sess:
			saver.restore(sess,"./weights/weights.cpkt")
			cache_train = sess.run(a_cache, feed_dict={self.X:X_train,self.y:y_train})
			cache_test = sess.run(a_cache, feed_dict={self.X:X_test,self.y:y_test})

		LR = LogisticRegression(C = 1e5)
		# print cache_train[L-1].shape, y_train.shape
		LR.fit(cache_train[L-1].T,np.argmax(y_train.T,1))
		ypred = LR.predict(cache_test[L-1].T)

		print "Test Accuracy for Logistic Regression on activations of layer %i (%i neurons): "%(L,self.n_h), accuracy_score(ypred,np.argmax(y_test.T,1))


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

# N.logistic(X_train,y_train,X_test,y_test,1)
# N.logistic(X_train,y_train,X_test,y_test,2)
# N.logistic(X_train,y_train,X_test,y_test,3)














