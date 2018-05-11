import numpy as np 
import random

from src.activation import *




class OutputLayer(Base):
	activation_forward = None
	activation_back = None
	
	def __init__(self, level, input_size, output_size, weight_scale, dtype):
		Base.__init__(self, level, input_size, output_size, weight_scale, dtype)
		self.activation_forward = affine_forward
		self.activation_back = affine_backward



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


