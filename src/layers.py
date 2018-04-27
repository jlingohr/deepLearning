import numpy as np 
import random
from abc import ABCMeta, abstractmethod

from src.activation import *


class AbstractLayer(metaclass=ABCMeta):

	def __init__(self, input_size, output_size, weight_scale, dtype):
		'''
		Initialize each layer in the network with a weight matrix
		W chosen from a Gaussian distribution and an intercept
		of 0
		'''
		#super().__init__()
		self.W = np.random.randn(input_size, output_size) * weight_scale
		self.W = self.W.astype(dtype)
		self.b = np.zeros(output_size)
		self.b = self.b.astype(dtype)

	@abstractmethod
	def feed_forward(self, x):
		'''
		Compute forward pass for a single layer
		and returns the output

		x: input to the layer

		out: output of layer

		side-effect: caches result of forward pass
		'''
		pass

	@abstractmethod
	def feed_backward(self, dscores):
		'''
		Compute backpropagation for a single layer

		dscores: upstream derivative
		'''
		pass

	def update(self, W, b):
		'''
		Updates weights and biases
		'''
		#print(self.W.shape, W.shape, self.b.shape, b.shape)
		self.W = W
		self.b = b


class ConnectedLayer(AbstractLayer):
	def __init__(self, input_size, output_size, weight_scale, dtype):
		AbstractLayer.__init__(self, input_size, output_size, weight_scale, dtype)

	def feed_forward(self, x):
		out, cache = affine_relu_forward(x, self.W, self.b)
		self.cache = cache
		return out

	def feed_backward(self, dscores):
		return affine_relu_backward(dscores, self.cache)

class OutputLayer(AbstractLayer):
	def __init__(self, input_size, output_size, weight_scale, dtype):
		AbstractLayer.__init__(self, input_size, output_size, weight_scale, dtype)

	def feed_forward(self, x):
		out, cache = affine_forward(x, self.W, self.b)
		self.cache = cache
		return out

	def feed_backward(self, dscores):
		return affine_backward(dscores, self.cache)


