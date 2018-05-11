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

class AffineBatchnorm(Activation):
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

def conv_forward_naive(x, w, b, conv_param):
	"""
	A naive implementation of the forward pass for a convolutional layer.

	The input consists of N data points, each with C channels, height H and
	width W. We convolve each input with F different filters, where each filter
	spans all C channels and has height HH and width WW.

	Input:
	- x: Input data of shape (N, C, H, W)
	- w: Filter weights of shape (F, C, HH, WW)
	- b: Biases, of shape (F,)
	- conv_param: A dictionary with the following keys:
	  - 'stride': The number of pixels between adjacent receptive fields in the
		horizontal and vertical directions.
	  - 'pad': The number of pixels that will be used to zero-pad the input. 
		

	During padding, 'pad' zeros should be placed symmetrically (i.e equally on both sides)
	along the height and width axes of the input. Be careful not to modfiy the original
	input x directly.

	Returns a tuple of:
	- out: Output data, of shape (N, F, H', W') where H' and W' are given by
	  H' = 1 + (H + 2 * pad - HH) / stride
	  W' = 1 + (W + 2 * pad - WW) / stride
	- cache: (x, w, b, conv_param)
	"""
	out = None
	N, C, H, W = x.shape
	F, _, HH, WW = w.shape
	pad = conv_param['pad']
	stride = conv_param['stride']
	Hp = int(1 + (H + 2 * pad - HH) / stride)
	Wp = int(1 + (W + 2 * pad - WW) / stride)

	pad_width = ((0,0), (0,0), (pad,pad), (pad,pad))
	padded = np.pad(x, pad_width=pad_width, mode='constant', constant_values=0)

	out = np.zeros((N, F, Hp, Wp))
	
	for data_ind in range(N):
		for filter_ind in range(F):
			for fw in range(Wp):
				ws = fw*stride
				for fh in range(Hp):
					hs = fh*stride
					out[data_ind, filter_ind, fh, fw] += np.sum(padded[data_ind][:, hs:hs+HH,
						ws:ws+WW] * w[filter_ind]) + b[filter_ind]
	cache = (x, w, b, conv_param)
	return out, cache

def conv_backward_naive(dout, cache):
	"""
	A naive implementation of the backward pass for a convolutional layer.

	Inputs:
	- dout: Upstream derivatives.
	- cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

	Returns a tuple of:
	- dx: Gradient with respect to x
	- dw: Gradient with respect to w
	- db: Gradient with respect to b
	"""
	dx, dw, db = None, None, None
	x, w, b, conv_params = cache
	N, C, H, W = x.shape
	F, _, HH, WW = w.shape

	stride = conv_params['stride']
	pad = conv_params['pad']
	pad_width = ((0,0), (0,0), (pad,pad), (pad,pad))
	x_padded = np.pad(x, pad_width=pad_width, mode='constant', constant_values=0)
	dx_padded = np.zeros(x_padded.shape)
	
	Hp = int(1 + (H + 2 * pad - HH) / stride)
	Wp = int(1 + (W + 2 * pad - WW) / stride)

	dw = np.zeros(w.shape)
	dx = np.zeros(x.shape)
	db = np.zeros(b.shape)

	for n in range(N):
		for f in range(F):
			db[f] += dout[n, f].sum()
			for j in range(Hp):
				for i in range(Wp):
					dw[f] += x_padded[n, :, j*stride:j*stride+HH, i*stride:i*stride+WW] * dout[n, f, j, i]
					dx_padded[n,:,j*stride:j*stride+HH,i*stride:i*stride+WW] += w[f]*dout[n,f,j,i]
	dx = dx_padded[:,:,pad:pad+H,pad:pad+W]
	return dx, dw, db

def max_pool_forward_naive(x, pool_param):
	"""
	A naive implementation of the forward pass for a max-pooling layer.

	Inputs:
	- x: Input data, of shape (N, C, H, W)
	- pool_param: dictionary with the following keys:
	  - 'pool_height': The height of each pooling region
	  - 'pool_width': The width of each pooling region
	  - 'stride': The distance between adjacent pooling regions

	No padding is necessary here. Output size is given by 

	Returns a tuple of:
	- out: Output data, of shape (N, C, H', W') where H' and W' are given by
	  H' = 1 + (H - pool_height) / stride
	  W' = 1 + (W - pool_width) / stride
	- cache: (x, pool_param)
	"""
	out = None
	
	N, C, H, W = x.shape
	HH = pool_param['pool_height']
	WW = pool_param['pool_width']
	stride = pool_param['stride']
	Hp = int(1 + (H-HH)/stride)
	Wp = int(1 + (W-WW)/stride)

	out = np.zeros((N,C,Hp,Wp))

	for n in range(N):
		for j in range(Hp):
			for i in range(Wp):
				out[n,:,j,i] = np.amax(x[n,:,j*stride:j*stride+HH,i*stride:i*stride+WW], axis=(-1,-2))

	cache = (x, pool_param)
	return out, cache


def max_pool_backward_naive(dout, cache):
	"""
	A naive implementation of the backward pass for a max-pooling layer.

	Inputs:
	- dout: Upstream derivatives
	- cache: A tuple of (x, pool_param) as in the forward pass.

	Returns:
	- dx: Gradient with respect to x
	"""
	dx = None

	x, pool_param = cache
	N,C,H,W = x.shape
	HH = pool_param['pool_height']
	WW = pool_param['pool_width']
	stride = pool_param['stride']
	Hp = int(1 + (H-HH)/stride)
	Wp = int(1 + (W-WW)/stride)

	dx = np.zeros_like(x)

	for n in range(N):
		for c in range(C):
			for j in range(Hp):
				for i in range(Wp):
					ind = np.argmax(x[n,c,j*stride:j*stride+HH,i*stride:i*stride+WW])
					ind1, ind2 = np.unravel_index(ind, (HH,WW))
					dx[n,c,j*stride:j*stride+HH,i*stride:i*stride+WW][ind1, ind2] = dout[n,c,j,i]

	return dx

def conv_relu_forward_naive(x, w, b, conv_param):
    """
    A convenience layer that performs a convolution followed by a ReLU.

    Inputs:
    - x: Input to the convolutional layer
    - w, b, conv_param: Weights and parameters for the convolutional layer

    Returns a tuple of:
    - out: Output from the ReLU
    - cache: Object to give to the backward pass
    """
    a, conv_cache = conv_forward_naive(x, w, b, conv_param)
    out, relu_cache = relu_forward(a)
    cache = (conv_cache, relu_cache)
    return out, cache

def conv_relu_backward_naive(dout, cache):
    """
    Backward pass for the conv-relu convenience layer.
    """
    conv_cache, relu_cache = cache
    da = relu_backward(dout, relu_cache)
    dx, dw, db = conv_backward_naive(da, conv_cache)
    return dx, dw, db

def conv_relu_pool_forward_naive(x, w, b, conv_param, pool_param):
    """
    Convenience layer that performs a convolution, a ReLU, and a pool.

    Inputs:
    - x: Input to the convolutional layer
    - w, b, conv_param: Weights and parameters for the convolutional layer
    - pool_param: Parameters for the pooling layer

    Returns a tuple of:
    - out: Output from the pooling layer
    - cache: Object to give to the backward pass
    """
    a, conv_cache = conv_forward_naive(x, w, b, conv_param)
    s, relu_cache = relu_forward(a)
    out, pool_cache = max_pool_forward_naive(s, pool_param)
    cache = (conv_cache, relu_cache, pool_cache)
    return out, cache

def conv_relu_pool_backward_naive(dout, cache):
    """
    Backward pass for the conv-relu-pool convenience layer
    """
    conv_cache, relu_cache, pool_cache = cache
    ds = max_pool_backward_naive(dout, pool_cache)
    da = relu_backward(ds, relu_cache)
    dx, dw, db = conv_backward_naive(da, conv_cache)
    return dx, dw, db

ACTIVATIONS = {
	'affine': Affine(),
	'relu': Relu(),
	'affine-relu': AffineRelu(),
	'norm':Batchnorm()
}
