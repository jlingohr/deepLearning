import numpy as np 
import random
import abc

from activation import *


class AbstractLayer(object):
	@abc.abstractmethod
	def __init__(input_size, outout_size, activation, weight_scale):
		'''
		Initialize each layer in the network with a weight matrix
		W chosen from a Gaussian distribution and an intercept
		of 0
		'''
		self.W = np.random.normal(0, scale=weight_scale, size=(input_size, outout_size))
		self.b = np.zeros(outout_size)

	@abc.abstractmethod
	def feed_forward(self, x):
		'''
		Compute forward pass for a single layer
		and returns the output

		x: input to the layer

		out: output of layer

		side-effect: caches result of forward pass
		'''
		return

	@abc.abstractmethod
	def feed_backward(self, dscores):
		'''
		Compute backpropagation for a single layer

		dscores: upstream derivative
		'''
		return

	def update(self, W, b):
		'''
		Updates weights and biases
		'''
		self.W = W
		self.b = b


class ConnectedLayer(AbstractLayer):
	def __init__(input_size, outout_size, activation, weight_scale):
		Layer.__init__(input_size, outout_size, activation, weight_scale)

	def feed_forward(self, x):
		out, cache = affine_relu_forward(x, self.w, self.b)
		self.cache = cache
		return out

	def feed_backwards(self, dscores):
		return affine_relu_backward(dscores, self.cache)

class OutputLayer(AbstractLayer):
	def __init__(input_size, outout_size, activation, weight_scale):
		Layer.__init__(input_size, outout_size, activation, weight_scale)

	def feed_forward(self, x):
		out, cache = affine_forward(x, self.w, self.b)
		self.cache = cache
		return out

	def feed_backward(self, dscores):
		return affine_backward(dscores, self.cache)


