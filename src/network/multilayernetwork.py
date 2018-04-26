import numpy as np
import random

from layers import ConnectedLayer, OutputLayer
from loss import softmax_loss

class MPL(object):
	def __init__(self, hidden_size, input_size, n_classes, weight_scale=1e-2):
		self.hidden_size = hidden_size
		self.num_layers = len(hidden_size)+1
		self.layers = __initialize_layers(hidden_size, input_size, n_classes, weight_scale)

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
		scores = forward_prop(X)
		# If test mode return early
		if y == None:
			return scores
		# Use backpropagation
		loss, grads = backward_prop(scores, y)
		return loss, grads


	def forward_prop(self, X):
		'''
		Do forward pass through the network to compute the scores
		X: Array of input data
		'''
		scores = X
		for layer in self.layers:
			scores = layer.feed_forward(scores)

	def backward_prop(self, scores, y):
		'''
		Compute loss and gradients using backpropagation
		scores: Loss computed using forward propagation
		y: array of labels

		Returns regularized loss and gradients
		'''
		loss, dscores = softmax_loss(scores, y) #TODO add regularization
		grads = {}

		delta, grads['W%d'%self.num_layers], grads['b%d'%self.num_layers] = self.layers[-1].feed_backward(dscores) #add regularization

		for l in xrange(2, self.num_layers):
			delta, grads['W%d'%(self.num_layers-l+1)], grads['b%d'%(self.num_layers-l+1)] = self.layers[-l].feed_backward(delta) # regularization
		return loss, grads

	def update_params(self, params):
		'''
		Update layers with new weights and biases
		params: List of (W,b) tuples
		'''
		for layer, param in zip(self.layers, params):
			layer.update(*param)


	def __initialize_layers(self, hidden_size, input_size, n_classes, weight_scale):
		dims = [input_size] + hidden_size + [n_classes]
		layers = []

		for l in range(self.num_layers-1):
			layer = ConnectedLayer(dims[l], dims[l+1], weight_scale)
			layers.append(layer)

		layers.append(OutputLayer(dims[-2], dims[-1], weight_scale))