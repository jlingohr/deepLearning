import numpy as np 
import random
from abc import ABCMeta, abstractmethod, abstractproperty

from src.activation import *


class AbstractLayer(metaclass=ABCMeta):

	def __init__(self, input_size, output_size, weight_scale, dtype):
		'''
		Initialize each layer in the network with a weight matrix
		W chosen from a Gaussian distribution and an intercept
		of 0
		'''
		self.W = np.random.randn(input_size, output_size) * weight_scale
		self.W = self.W.astype(dtype)
		self.b = np.zeros(output_size)
		self.b = self.b.astype(dtype)

	@abstractproperty
	def activation_forward(self):
		return self.activation_forward

	@abstractproperty
	def activation_back(self):
		return self.activation_back

	#@abstractmethod
	def feed_forward(self, x):
		'''
		Compute forward pass for a single layer
		and returns the output

		x: input to the layer

		out: output of layer
		side-effect: caches result of forward pass
		'''
		out, cache = self.activation_forward(x, self.W, self.b)
		self.cache = cache
		return out

	#@abstractmethod
	def feed_backward(self, dscores):
		'''
		Compute backpropagation for a single layer

		dscores: upstream derivative
		'''
		return self.activation_back(dscores, self.cache)

	def update(self, W, b):
		'''
		Updates weights and biases
		'''
		#print(self.W.shape, W.shape, self.b.shape, b.shape)
		self.W = W
		self.b = b


class ConnectedLayer(AbstractLayer):
	activation_forward = None
	activation_back = None
	
	def __init__(self, input_size, output_size, weight_scale, dtype):
		AbstractLayer.__init__(self, input_size, output_size, weight_scale, dtype)
		self.activation_forward = affine_relu_forward
		self.activation_back = affine_relu_backward

class BatchNorm(ConnectedLayer):
	activation_forward = None
	activation_back = None

	def __init__(self, input_size, output_size, weight_scale, dtype):
		ConnectedLayer.__init__(self, input_size, output_size, weight_scale, dtype)
		self.activation_forward = batchnorm_forward
		self.activation_back = batchnorm_backward


class OutputLayer(AbstractLayer):
	activation_forward = None
	activation_back = None
	
	def __init__(self, input_size, output_size, weight_scale, dtype):
		AbstractLayer.__init__(self, input_size, output_size, weight_scale, dtype)
		self.activation_forward = affine_forward
		self.activation_back = affine_backward


