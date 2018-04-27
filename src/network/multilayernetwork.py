import numpy as np
import random

from src.layers import ConnectedLayer, OutputLayer
from src.loss import softmax_loss

class MLP(object):
	def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10, weight_scale=1e-2, reg=0.0, dtype=np.float32):
		self.hidden_dims = hidden_dims
		self.num_layers = len(hidden_dims)+1
		self.layers = self.__initialize_layers(hidden_dims, input_dim, num_classes, weight_scale)
		self.dtype = dtype
		self.reg = reg

	def loss(self, X, y=None):
		'''
		Compute loss and gradient for network. If y is None then run a forward pass
		and return an array of classification scores where scores[i,c] is the
		classification score for X[i] and class c. Otherwise run forward and
		backward pass an return a tuple (loss, grads) where loss is the scalar
		value of the loss function and grads contains the gradients of the loss
		for each layer.

		X: array of input data
		y: array of labels where y[i] gives the label for X[i]

		'''
		X = X.astype(self.dtype)

		scores = self.forward_prop(X)
		# If test mode return early
		if y is None:
			return scores
		# Use backpropagation
		loss, grads = self.backward_prop(scores, y)
		return loss, grads


	def forward_prop(self, X):
		'''
		Do forward pass through the network to compute the scores
		X: Array of input data
		'''
		scores = X
		for layer in self.layers:
			scores = layer.feed_forward(scores)
		return scores

	def backward_prop(self, scores, y):
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

		delta, grads['W%d'%self.num_layers], grads['b%d'%self.num_layers] = self.layers[-1].feed_backward(dscores) 
		grads['W%d'%self.num_layers] += self.reg*self.layers[-1].W

		for l in range(2, self.num_layers+1):
			delta, grads['W%d'%(self.num_layers-l+1)], grads['b%d'%(self.num_layers-l+1)] = self.layers[-l].feed_backward(delta) # regularization
			grads['W%d'%(self.num_layers-l+1)] += self.reg*self.layers[-l].W
		return loss, grads

	def update_params(self, params):
		'''
		Update layers with new weights and biases
		params: List of (W,b) tuples
		'''
		for layer, param in zip(self.layers, params):
			layer.update(*param)


	def __initialize_layers(self, hidden_dims, input_dim, num_classes, weight_scale):
		dims = [input_dim] + hidden_dims + [num_classes]
		layers = []

		for l in range(self.num_layers-1):
			layer = ConnectedLayer(dims[l], dims[l+1], weight_scale)
			layers.append(layer)

		layer = OutputLayer(dims[-2], dims[-1], weight_scale)
		layers.append(layer)

		return layers