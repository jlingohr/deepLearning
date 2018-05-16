from .base import Layer
from ..activations import fast_activations as activations

import numpy as np 

class Convolutional(Layer):
	'''
	Convolutional layer

	Parameters:
	- weight_scale:
	- dtype
	- filter_size
	- num_filters
	- channels
	- stage_params: Dictionary containing parameters for a single convolutiona stage
		- conv_param: parameters for the convolutional stage
			- stride
			- pad
		- pool_param: parameters for the pooling stage
			- pool_height
			- pool_width
			- stride
		- activation: What kind of activation (only Relu support for now)
	'''
	def __init__(self, weight_scale, dtype, filter_size, num_filters, channels, stage_params, name=None):
		Layer.__init__(self, weight_scale, dtype, name)
		self.filter_size = filter_size
		self.num_filters = num_filters
		self.channels = channels
		self.stage_params = stage_params
		self._W = np.random.normal(0, scale=weight_scale, size=(num_filters, channels, filter_size, filter_size)).astype(dtype)
		self._b = np.zeros((num_filters,))

		self.activation = self._initialize_activation() 

	def feed_forward(self, x, mode):
		conv_param = self.stage_params.get('conv_param',{'stride': 1, 'pad': (self.filter_size - 1) // 2})
		pool_param = self.stage_params.get('pool_param')
		out, cache = self.activation.forward(x = x, w = self._W, b = self._b, 
				conv_param=conv_param, pool_param=pool_param)
		self.cache = cache
		return out

	def feed_backward(self, dscores):
		grads = {}
		dx, dw, db = self.activation.backward(dscores, self.cache)
		grads['W%d'%self.name] = dw
		grads['b%d'%self.name] = db
		return dx, grads

	def _initialize_activation(self):
		return activations.ConvMaxPoolRelu() if self.stage_params.get('pool_param') is not None else activations.ConvolutionRelu()

	