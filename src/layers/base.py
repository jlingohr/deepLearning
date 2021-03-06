from abc import ABCMeta, abstractmethod, abstractproperty
import numpy as np

class Layer(metaclass=ABCMeta):

	def __init__(self, weight_scale, dtype, name=None):
		'''
		Initialize each layer in the network with a weight matrix
		W chosen from a Gaussian distribution and an intercept
		of 0
		'''
		self.name = name
		self.weight_scale = weight_scale
		self.dtype = dtype
		self._W = None
		self._b = None
		self.layers = None

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
		p = {}
		p['W%d'%self.name] = self._W
		p['b%d'%self.name] = self._b
		return p

	@params.setter
	def params(self, new_params):
		self._W = new_params['W']
		self._b = new_params['b']
