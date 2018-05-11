from abc import ABCMeta, abstractmethod, abstractproperty
import numpy as np

class Layer(metaclass=ABCMeta):

	def __init__(self, input_size, output_size, weight_scale, dtype, name=None):
		'''
		Initialize each layer in the network with a weight matrix
		W chosen from a Gaussian distribution and an intercept
		of 0
		'''
		self.name = name
		self.input_size = input_size
		self.output_size = output_size
		self.weight_scale = weight_scale
		self.dtype = dtype
		self._W = (np.random.randn(input_size, output_size) * weight_scale).astype(dtype)
		self._b = np.zeros(output_size).astype(dtype)

	# @abstractmethod
	# def activation_forward(self):
	# 	return 

	# @abstractmethod
	# def activation_back(self):
	# 	return 

	# @abstractproperty
	# def activation(self):
	# 	''''''

	@abstractmethod
	def feed_forward(self, x, mode):
		'''
		Compute forward pass for a single layer
		and returns the output

		x: input to the layer
		params: dictionary containing parameters

		out: output of layer
		side-effect: caches result of forward pass
		'''

	@abstractmethod
	def feed_backward(self, dscores):
		'''
		Compute backpropagation for a single layer

		dscores: upstream derivative
		Returns dx, dw, db
		'''
	

	@property
	def W(self):
		return self._W

	@W.setter
	def W(self, weights):
		self._W = weights

	@property
	def b(self):
		return self._b

	@b.setter
	def b(self, b):
		self._b = b

	@property
	def params(self):
		params = {}
		params['W%d'%self.name] = self._W
		params['b%d'%self.name] = self._b
		return params

	def update(self, params):
		'''
		Updates weights and bes
		params: Dictionary containin W, b, gamma, beta
		'''
		self._W = params['W']
		self._b = params['b']