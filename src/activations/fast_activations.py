import numpy as np 
from .fast_functions import *
from .activations import Activation 

class Convolution(Activation):
	def forward(self, **kwargs):
		x = kwargs['x']
		w = kwargs['w']
		b = kwargs['b']
		conv_param = kwargs['conv_param']
		return conv_forward_fast(x, w, b, conv_param)

	def backward(dout, cache):
		return conv_backward_fast(dout, cache)
		
class MaxPool(Activation):
	def forward(self, **kwargs):
		x = kwargs['x']
		pool_param = kwargs['pool_param']
		return max_pool_forward_fast(x, pool_param)

	def backward(self, dout, cache):
		return max_pool_backward_fast(dout, cache)

class ConvolutionRelu(Activation):
	def forward(self, **kwargs):
		x = kwargs['x'] 
		w = kwargs['w'] 
		b = kwargs['b']
		conv_param = kwargs['conv_param']
		return conv_relu_forward_fast(x, w, b, conv_param)

	def backward(self, dout, cache):
		return conv_relu_backward_fast(dout, cache)

class ConvMaxPoolRelu(Activation):
	def forward(self, **kwargs):
		x = kwargs['x']
		w = kwargs['w']
		b = kwargs['b']
		conv_param = kwargs['conv_param']
		pool_param = kwargs['pool_param']
		return conv_relu_pool_forward_fast(x, w, b, conv_param, pool_param)
	   
	def backward(self, dout, cache):
		return conv_relu_pool_backward_fast(dout, cache)