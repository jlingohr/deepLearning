import numpy as np 
from abc import ABCMeta, abstractmethod, abstractproperty

from .functions import *

class Activation(metaclass=ABCMeta):
	@abstractmethod
	def forward(self, **kwargs):
		return

	@abstractmethod
	def backward(self, dout, cache):
		return

class Affine(Activation):
	def forward(self, **kwargs):
		x = kwargs['x']
		w = kwargs['w']
		b = kwargs['b']
		return affine_forward(x, w, b)

	def backward(self, dout, cache):
		return affine_backward(dout, cache)

class Relu(Activation):
	def forward(self, **kwargs):
		return relu_forward(kwargs['x'])

	def backward(self, dout, cache):
		return relu_backward(dout, cache)

class AffineRelu(Activation):
	def forward(self, **kwargs):
		x = kwargs['x']
		w = kwargs['w']
		b = kwargs['b']
		return affine_relu_forward(x, w, b)

	def backward(self, dout, cache):
		return affine_relu_backward(dout, cache)

def sigmoid(z):
	'''
	Sigmoid function
	'''
	return 1.0/(1.0 + np.exp(-z))

class Batchnorm(Activation):
	def forward(self, **kwargs):
		x = kwargs['x'] 
		gamma = kwargs['gamma']
		beta = kwargs['beta'] 
		bn_param = kwargs['bn_param']
		return batchnorm_forward(x, gamma, beta, bn_param)


	def backward(self, dout, cache):
		return batchnorm_backward(dout, cache)

class AffineBatchNorm(Activation):
	def forward(self, **kwargs):
		x = kwargs['x']
		w = kwargs['w']
		b = kwargs['b']
		gamma = kwargs['gamma']
		beta = kwargs['beta']
		bn_params = kwargs['bn_params']

		return affine_batchnorm_forward(x, w, b, gamma, beta, bn_params)

	def backward(self, dout, cache):
		return affine_batchnorm_backward(dout, cache)

class Convolution(Activation):
	def forward(self, **kwargs):
		x = kwargs['x']
		w = kwargs['w']
		b = kwargs['b']
		conv_param = kwargs['conv_param']
		return conv_forward_naive(x, w, b, conv_param)

	def backward(dout, cache):
		return conv_backward_naive(dout, cache)
		
class MaxPool(Activation):
	def forward(self, **kwargs):
		x = kwargs['x']
		pool_param = kwargs['pool_param']
		return max_pool_forward_naive(x, pool_param)

	def backward(self, dout, cache):
		return max_pool_backward_naive(dout, cache)

class ConvolutionRelu(Activation):
	def forward(self, **kwargs):
		x = kwargs['x'] 
		w = kwargs['w'] 
		b = kwargs['b']
		conv_param = kwargs['conv_param']
		return conv_relu_forward_naive(x, w, b, conv_param)

	def backward(self, dout, cache):
		return conv_relu_backward_naive(dout, cache)

class ConvMaxPoolRelu(Activation):
	def forward(self, **kwargs):
		x = kwargs['x']
		w = kwargs['w']
		b = kwargs['b']
		conv_param = kwargs['conv_param']
		pool_param = kwargs['pool_param']
		return conv_relu_pool_forward_naive(x, w, b, conv_param, pool_param)
	   
	def backward(self, dout, cache):
		return conv_relu_pool_backward_naive(dout, cache)

ACTIVATIONS = {
	'affine': Affine(),
	'relu': Relu(),
	'affine-relu': AffineRelu(),
	'norm':Batchnorm()
}

