'''
	Implementation of the Restrict Boltzmann Machine algorithm.
	@author: https://github.com/ViniciusLima94
	Repository: 
'''

import numpy as np

class RBM:

	def __init__(self, X = None, n_hidden = None, epochs = 10, mini_batch = None, k = None):
		'''
			Constructor method.
			Inputs:
			Outputs:
		'''
		self.k         = k           # Number of Gibbs samplings.
		self.X         = X           # Data.
		self.epochs    = epochs      # Number of epochs.
		self.n_samples  = X.shape[0] # Number of data samples.
		self.n_visible = X.shape[1]  # Number of visible nodes (= number of features).          
		# Number of hidden nodes.
		if n_hidden == None:
			self.n_hidden = self.n_visible
		else:
			self.n_hidden = n_hidden
		if mini_batch == None:
			self.mini_batch = self.n_samples
		else:
			self.mini_batch = mini_batch # Size of the mini batch.
		# Weigth of the connections
		self.W = np.random.normal(0, 0.01, size=(self.n_visible, self.n_hidden))
		# Bias of the visible layer
		self.pi = self.X.copy()
		self.pi[self.pi<0] = 0
		self.pi = self.pi.mean(axis=0) + 0.0001
		self.a = np.log( self.pi / (1-self.pi) )
		# Bias of the hidden layer
		self.b = np.zeros(self.n_hidden)

	def fit(self, ):
		for epoch in range(self.epochs):
			loss = 0
			n    = 0
			for mb in range(0, int(self.n_samples), self.mini_batch):
				count = 0
				v0      = self.X[mb:mb+self.mini_batch]               # Setting visible layer
				n1, n2  = np.where(v0<0)                              # Index of visible units with value -1
				p_h0_v0 = self.sample_h(v0)                           # Sampling hidden layer
				h0      = self.bernoulli_sampling(p_h0_v0) # Bernoulli sampling hidden layer
				count = count + 1
				# Iteration for the rest of Gibbis sampling
				h_k     = h0[:]
				while count <= self.k:
					p_v_h = self.sample_v(h_k)
					v_k   = self.bernoulli_sampling(p_v_h)
					# Setting v_k where v0 was -1 to -1
					v_k[n1, n2] = -1
					p_h_v =  self.sample_h(v_k) 
					h_k   = self.bernoulli_sampling(p_h_v)
					count += 1
				# Computing loss
				loss += np.mean( np.abs( v_k[v0>=0] - v0[v0>=0] ) )
				n    += 1.0
				# Contrastive divergence
				delta_W, delta_a, delta_b = self.contrastive_divergence(v0, v_k, p_h0_v0, p_h_v)
				# Updating weights and biases.
				self.W += delta_W
				self.a += delta_a
				self.b += delta_b
			print 'epoch: ' + str(epoch+1) + ', loss: ' + str(loss/n)

	def inference(self, X_infer):
		v0      = X_infer
		p_h0_v0 = self.sample_h(v0)   
		h0      = self.bernoulli_sampling(p_h0_v0)
		h_k     = h0[:]
		p_v_h = self.sample_v(h_k)
		self.v_k   = self.bernoulli_sampling(p_v_h)
		test_loss  = np.mean( np.abs( self.v_k[v0>=0]-v0[v0>=0] ) )
		print 'test loss: ' + str(test_loss)

	def contrastive_divergence(self, v0, v_k, p_h0_v0, p_h_v):
		delta_W = np.zeros([self.W.shape[0], self.W.shape[1]])
		for i in range(v0.shape[0]):
			delta_W += ( np.outer(v0[i, :], p_h0_v0[i, :]) - np.outer(v_k[i, :], p_h_v[i, :]) ) / float(self.mini_batch)
		delta_b = (p_h0_v0 - p_h_v).mean(axis=0)
		delta_a = (v0 - v_k).mean(axis=0)
		return delta_W, delta_a, delta_b

	def bernoulli_sampling(self, layer):
		if len(layer.shape) == 2:
			r, c = layer.shape
			eta = np.random.rand(r, c)
		elif len(layer.shape) == 1:
			r    = layer.shape[0]
			eta = np.random.rand(r)
		return (eta < layer).astype(int)

	def sample_h(self, v):
		z = self.b + np.dot(v, self.W) 
		return 1.0 / ( 1 + np.exp(-z) ) 

	def sample_v(self, h):
		z = self.a + np.dot(h, self.W.T) 
		return 1.0 / ( 1 + np.exp(-z) ) 
		
		


