from .base import Layer
from ..activations.activations import AffineBatchNorm
import numpy as np

class BatchNormalization(Layer):
	def __init__(self, input_size, output_size, weight_scale, dtype, name=None):
		'''
		Special layer to use when using batch normalization
		'''
		Layer.__init__(self, weight_scale, dtype, name)
		self._W = (np.random.randn(input_size, output_size) * weight_scale).astype(dtype)
		self._b = np.zeros(output_size).astype(dtype)
		self.activation = AffineBatchNorm()
		self._gamma = np.ones((1,1))
		self._beta = np.zeros((1,1))
		self.bn_params = {}

	def feed_forward(self, x, mode):
		self.bn_params['mode'] = mode
		out, cache = self.activation.forward(x=x, w=self._W, b=self._b, gamma=self._gamma, beta=self._beta, bn_params=self.bn_params)
		self.cache = cache
		return out

	def feed_backward(self, dscores):
		grads = {}
		dx, dw, db, dgamma, dbeta = self.activation.backward(dscores, self.cache)
		grads['W%d'%self.name] = dw
		grads['b%d'%self.name] = db
		grads['gamma%d'%self.name] = dgamma
		grads['beta%d'%self.name] = dbeta
		return dx, grads

	@property
	def params(self):
		p = {}
		p['W%d'%self.name] = self._W
		p['b%d'%self.name] = self._b
		p['gamma%d'%self.name] = self._gamma
		p['beta%d'%self.name] = self._beta
		return p

	@params.setter
	def params(self, new_params):
		self._W = new_params['W']
		self._b = new_params['b']
		self._gamma = new_params['gamma']
		self._beta = new_params['beta']
