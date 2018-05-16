import numpy as np

from ..layers.convolutional import Convolutional
from ..layers.pooling import MaxPooling
from ..layers.dense import Dense
from src.loss import softmax_loss


class ThreeLayerConvNet(object):
	"""
	A three-layer convolutional network with the following architecture:

	conv - relu - 2x2 max pool - affine - relu - affine - softmax

	The network operates on minibatches of data that have shape (N, C, H, W)
	consisting of N images, each with height H and width W and with C input
	channels.
	"""

	def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
				 hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
				 dtype=np.float32):
		"""
		Initialize a new network.

		Inputs:
		- input_dim: Tuple (C, H, W) giving size of input data
		- num_filters: Number of filters to use in the convolutional layer
		- filter_size: Width/height of filters to use in the convolutional layer
		- hidden_dim: Number of units to use in the fully-connected hidden layer
		- num_classes: Number of scores to produce from the final affine layer.
		- weight_scale: Scalar giving standard deviation for random initialization
		  of weights.
		- reg: Scalar giving L2 regularization strength
		- dtype: numpy datatype to use for computation.
		"""
		self.reg = reg
		self.dtype = dtype
		
		C,H,W = input_dim
		
		conv_param = {'stride': 1, 'pad': (filter_size - 1) // 2}
		pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}
		stage_params = {'conv_param':conv_param, 'pool_param':pool_param}
		self._layers = []
		self._layers.append(Convolutional(weight_scale, dtype, filter_size, num_filters, C, stage_params, 1))
		self._layers.append(Dense(num_filters*(H//2)*(W//2), hidden_dim, weight_scale, dtype, 2))
		self._layers.append(Dense(hidden_dim, num_classes, weight_scale, dtype, 3))
		self.num_layers = len(self.layers)

	@property
	def layers(self):
		return self._layers

	@layers.setter
	def layers(self, l):
		self._layers = l
	

	@property
	def params(self):
		'''
		Return weights and coefficients of each layer in a parameter
		dictionary.
		Returns:
		params: dictionary of weights W and coefficients b
		'''
		par = {}
		for l in range(len(self.layers)):
			layer = self.layers[l]
			par.update(layer.params)
		return par

	@params.setter
	def params(self, new_params):
		for l in range(len(self.layers)):
			p = l+1
			d = {k.replace(str(p),''):v for k, v in new_params.items() if str(p) in k}
			self.layers[l].params = d


	def loss(self, X, y=None):
		"""
		Evaluate loss and gradient for the three-layer convolutional network.

		Input / output: Same API as TwoLayerNet in fc_net.py.
		"""
		# W1, b1 = self.params['W1'], self.params['b1']
		# W2, b2 = self.params['W2'], self.params['b2']
		# W3, b3 = self.params['W3'], self.params['b3']

		# pass conv_param to the forward pass for the convolutional layer
		# Padding and stride chosen to preserve the input spatial size
		filter_size = self.layers[0].W.shape[2]
		conv_param = {'stride': 1, 'pad': (filter_size - 1) // 2}

		# pass pool_param to the forward pass for the max-pooling layer
		pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

		scores = None
		############################################################################
		# TODO: Implement the forward pass for the three-layer convolutional net,  #
		# computing the class scores for X and storing them in the scores          #
		# variable.                                                                #
		#                                                                          #
		# Remember you can use the functions defined in cs231n/fast_layers.py and  #
		# cs231n/layer_utils.py in your implementation (already imported).         #
		############################################################################
		X = X.astype(self.dtype)
		mode = 'test' if y is None else 'train'
		params = {'mode':mode}

		scores = self._feed_forward(X, mode)

		if y is None:
			return scores

		loss, grads = self._feed_backward(scores, y)
		return loss, grads

	def _feed_forward(self, X, mode):
		'''
		Do forward pass through the network to compute the scores
		X: Array of input data
		'''
		scores = X
		for layer in self.layers:
			scores = layer.feed_forward(scores, mode)
		return scores

	def _feed_backward(self, scores, y):
		'''
		Compute loss and gradients using backpropagation
		scores: Loss computed using forward propagation
		y: array of labels

		Returns regularized loss and gradients
		'''
		loss, dscores = softmax_loss(scores, y) 
		for layer in self.layers:
			loss += 0.5*self.reg*np.sum(layer.W**2)
		grads = {}
		
		delta, grad = self.layers[-1].feed_backward(dscores) 
		grads.update(grad)
		grads['W%d'%self.num_layers] += self.reg*self.layers[-1].W

		for l in range(2, self.num_layers+1):
			delta, grad = self.layers[-l].feed_backward(delta)
			grads.update(grad)
			grads['W%d'%(self.num_layers-l+1)] += self.reg*self.layers[-l].W
		return loss, grads
