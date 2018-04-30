import numpy as np 
import random
from abc import ABCMeta, abstractmethod, abstractproperty

from src.activation import *


class AbstractLayer(metaclass=ABCMeta):

	def __init__(self, level, input_size, output_size, weight_scale, dtype):
		'''
		Initialize each layer in the network with a weight matrix
		W chosen from a Gaussian distribution and an intercept
		of 0
		'''
		self.level = level
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
	def feed_forward(self, x, mode):
		'''
		Compute forward pass for a single layer
		and returns the output

		x: input to the layer
		params: dictionary containing parameters

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
		Returns dx, dw, db
		'''
		grads = {}
		dx, dw, db = self.activation_back(dscores, self.cache)
		grads['W%d'%self.level] = dw
		grads['b%d'%self.level] = db
		return dx, grads

	def update(self, params):
		'''
		Updates weights and biases
		params: Dictionary containin W, b, gamma, beta
		'''
		self.W = params['W']
		self.b = params['b']

	def params(self):
		'''
		Returns parameters of layer in a dictionary
		'''
		params = {}
		params['W%d'%self.level] = self.W
		params['b%d'%self.level] = self.b
		return params


class ConnectedLayer(AbstractLayer):
	activation_forward = None
	activation_back = None
	
	def __init__(self, level, input_size, output_size, weight_scale, dtype):
		AbstractLayer.__init__(self, level, input_size, output_size, weight_scale, dtype)
		self.activation_forward = affine_relu_forward
		self.activation_back = affine_relu_backward


class OutputLayer(AbstractLayer):
	activation_forward = None
	activation_back = None
	
	def __init__(self, level, input_size, output_size, weight_scale, dtype):
		AbstractLayer.__init__(self, level, input_size, output_size, weight_scale, dtype)
		self.activation_forward = affine_forward
		self.activation_back = affine_backward

class BatchNormLayer(ConnectedLayer):
	activation_forward = None
	activation_back = None

	def __init__(self, level, input_size, output_size, weight_scale, dtype):
		'''
		Special layer to use when using batch normalization
		'''
		ConnectedLayer.__init__(self, level, input_size, output_size, weight_scale, dtype)
		self.gamma = np.ones((1,1))
		self.beta = np.zeros((1,1))
		self.activation_forward = affine_batchnorm_forward
		self.activation_back = affine_batchnorm_backward
		self.bn_params = {}

	def feed_forward(self, x, mode):
		self.bn_params['mode'] = mode
		out, cache = self.activation_forward(x, self.W, self.b, self.gamma, self.beta, self.bn_params)
		self.cache = cache
		return out

	def feed_backward(self, dscores):
		grads = {}
		dx, dw, db, dgamma, dbeta = self.activation_back(dscores, self.cache)
		grads['W%d'%self.level] = dw
		grads['b%d'%self.level] = db
		grads['gamma%d'%self.level] = dgamma
		grads['beta%d'%self.level] = dbeta
		return dx, grads

	def params(self):
		params = ConnectedLayer.params(self)
		params['gamma%d'%self.level] = self.gamma
		params['beta%d'%self.level] = self.beta
		return params

	def update(self, params):
		ConnectedLayer.update(self, params)
		self.gamma = params['gamma']
		self.beta = params['beta']

class LayerFactory(object):
	LAYER = {
		None: ConnectedLayer,
		'batchnorm': BatchNormLayer
	}

	def __init__(self, weight_scale, dtype):
		self.weight_scale = weight_scale
		self.dtype = dtype

	def make_ConnectedLayer(self, level, input_size, output_size, weight_scale, dtype):
		return ConnectedLayer(level, input_size, output_size, self.weight_scale, self.dtype)

	def make_BatchNormLayer(self, level, input_size, output_size, weight_scale, dtype):
		return BatchNormLayer(level, input_size, output_size, self.weight_scale, self.dtype)

	def make(self, level, input_size, output_size, normalization=None):
		return self.LAYER[normalization](level, input_size, output_size, self.weight_scale, self.dtype)


