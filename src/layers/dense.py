from .base import Layer
from ..activations import activations

import numpy as np

class Dense(Layer):	
	def __init__(self, input_size, output_size, weight_scale, dtype, name=None, activation='affine'):
		Layer.__init__(self, weight_scale, dtype, name)
		self._W = (np.random.randn(input_size, output_size) * weight_scale).astype(dtype)
		self._b = np.zeros(output_size).astype(dtype)
		self.activation = activations.ACTIVATIONS[activation]

	def feed_forward(self, x, mode):
		out, cache = self.activation.forward(x = x, w = self._W, b = self._b)
		self.cache = cache
		return out

	def feed_backward(self, dscores):
		grads = {}
		dx, dw, db = self.activation.backward(dscores, self.cache)
		grads['W%d'%self.name] = dw
		grads['b%d'%self.name] = db
		return dx, grads