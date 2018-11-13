import numpy as np 
import matplotlib.pyplot as plt 

class ANN:

	def __init__(self, layers, X, y, epochs=10000, alpha=1, seed=1):
		'''
			Constructor method. Build the ANN
			Inputs:
			layer: Tuple with number of neurons in each hidden layer
			input: Input values to the ANN
			output: Output used to train the weights
			alpha: Learning rate
			seed:  Random seed to be used
			epochs: Number of epochs
		'''
		# Set seed
		np.random.seed(seed)
		self.layers = layers                   # How many neurons in each hidden layer, the last one is the output layer (tuple)
		self.epochs = epochs                   # Number of epochs (iteractions)
		self.alpha  = alpha                    # Learning rate
		self.X      = X                        # Input values
		self.y      = y                        # Output values (labels)
		self.n_layers    = len(self.layers)        # Number of hidden layers
		self.n_input_layers   = self.X.shape[1]    # Number of neurons in the input layer
		self.a                = {}                 # Dictionary to store neurons activation in each layer
		for i in range(self.n_layers+1):           # Initializing one entrie for each hidden layer
			if i == 0:
				self.a[i] = self.X
			else:
				self.a[i] = []
		# Creating random weights
		self.weights = {}                          # Weights dictionary
		# Initializing for the input layer
		self.weights[0] = np.random.normal( 0, 1, size=(self.n_input_layers, self.layers[0]) )
		# If there is more then one hidden layer, initialize the weights between then
		if self.n_layers > 1:
			for i in range(self.n_layers - 1):
				self.weights[i+1] = np.random.normal( 0, 1, size=(self.layers[i], self.layers[i+1]) )

	def fit(self,):
		'''	
			Fit the weights in the neural net.
		'''
		for i in range(self.epochs):
			for j in range(self.n_layers):
				self.a[j+1] = self.sigmoid( np.dot( self.a[j], self.weights[j] ) )  # Compute input in the jth in the hidden layer
			# Backpropagation algorithm
			self.delta = self.cost_function_prime(self.y, self.a[self.n_layers]) * self.sigmoid_prime(self.a[self.n_layers])            # Compute error [delta = dC/da]
			self.weights[self.n_layers-1] -= self.alpha * np.dot(self.a[self.n_layers-1].T, self.delta)  # Update weights
			for L in range(2, self.n_layers+1):
				self.delta = np.dot( self.delta, self.weights[self.n_layers-(L-1)].T ) *  self.sigmoid_prime(self.a[self.n_layers-(L-1)])
				self.weights[self.n_layers-L] -= self.alpha * np.dot(self.a[self.n_layers-L].T, self.delta)  # Update weights

	def sigmoid(self, x):
		'''
			Sigmoid function, computes the output of f(x), where x in the input
			and f is the sigmoid function f(x) = 1 / (1 + exp(-x)).
			Inputs:
			x: Input value
			Outputs:
			Return f(x)
		'''
		return 1 / (1 + np.exp(-x) )

	def sigmoid_prime(self, output):
		'''
			The first derivative of the sigmoid function, computes the output of f(x), where x in the input
			and f is the sigmoid function f'(x) = f(x)*(1-f(x)).
			Inputs:
			x: Input value
			Outputs:
			Return f'(x)
		'''
		return output * ( 1 - output )

	def cost_function(self, y, yprime):
		'''
			Return the quadratic cost function  C = 1/2 * (y-yprime)^2, between the expected value and estimated value.
			Inputs:
			y: Expected value
			yprime: Estimated value
			Outputs:
			Return C
		'''
		return 0.5 * (y - yprime)**2

	def cost_function_prime(self, y, yprime):
		'''
			Return the first derivative of the quadratic cost function  C' = (y-yprime), 
			between the expected value and estimated value.
			Inputs:
			y: Expected value
			yprime: Estimated value
			Outputs:
			Return C'
		'''
		return (yprime - y)
