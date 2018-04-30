import numpy as np 


def affine_forward(x, w, b):
	'''
	Compute affine forward pass for a single layer
	x: numpy array containing input data of shape (N, d_1, ..., d_k)
	w: numpy array of weights
	b: numpy array of biases

	Returns a tuple of (out, cache) where out is the output
	and cache is (x, w, b)
	'''
	out = (x.reshape(x.shape[0], w.shape[0]))@w + b
	cache = (x, w, b)

	return out, cache

def affine_backward(dout, cache):
	"""
	Computes the backward pass for an affine layer.

	Inputs:
	- dout: Upstream derivative, of shape (N, M)
	- cache: Tuple of:
	  - x: Input data, of shape (N, d_1, ... d_k)
	  - w: Weights, of shape (D, M)
	  - b: Biases, of shape (M,)

	Returns a tuple of:
	- dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
	- dw: Gradient with respect to w, of shape (D, M)
	- db: Gradient with respect to b, of shape (M,)
	"""
	x, w, b = cache
	dx, dw, db = None, None, None

	dx = (dout@w.T).reshape(x.shape)
	dw = x.reshape(x.shape[0], w.shape[0]).T@dout
	db = np.sum(dout, axis=0)
	return dx, dw, db


def relu_forward(x):
	'''
	Compute forward passing using ReLu activation
	'''
	out = np.maximum(0, x)
	cache = x
	return out, cache

def relu_backward(dout, cache):
	'''
	Computes the backward pass for a layer of rectified linear units (ReLUs).

	Input:
	- dout: Upstream derivatives, of any shape
	- cache: Input x, of same shape as dout

	Returns:
	- dx: Gradient with respect to x
	'''
	dx, x = None, cache
	dx = dout
	dx[cache <= 0] = 0
	return dx


def affine_relu_forward(x, w, b):
	'''
	Perform affine transformation followed by ReLu
	- x: Input to the affine layer
	- w, b: Weights for the affine layer

	Returns a tuple of:
	- out: Output from the ReLU
	- cache: Object to give to the backward pass
	'''
	a, fc_cache = affine_forward(x, w, b)
	out, relu_cache = relu_forward(a)
	cache = (fc_cache, relu_cache)
	return out, cache

def affine_relu_backward(dout, cache):
	'''
	Backward pass for the affine-relu convenience layer
	'''
	fc_cache, relu_cache = cache
	da = relu_backward(dout, relu_cache)
	dx, dw, db = affine_backward(da, fc_cache)
	return dx, dw, db

def sigmoid(z):
	'''
	Sigmoid function
	'''
	return 1.0/(1.0 + np.exp(-z))

def batchnorm_forward(x, gamma, beta, bn_param):
	"""
	Forward pass for batch normalization.

	running_mean = momentum * running_mean + (1 - momentum) * sample_mean
	running_var = momentum * running_var + (1 - momentum) * sample_var

	Input:
	- x: Data of shape (N, D)
	- gamma: Scale parameter of shape (D,)
	- beta: Shift paremeter of shape (D,)
	- bn_param: Dictionary with the following keys:
	  - mode: 'train' or 'test'; required
	  - eps: Constant for numeric stability
	  - momentum: Constant for running mean / variance.
	  - running_mean: Array of shape (D,) giving running mean of features
	  - running_var Array of shape (D,) giving running variance of features

	Returns a tuple of:
	- out: of shape (N, D)
	- cache: A tuple of values needed in the backward pass
	"""
	mode = bn_param['mode']
	eps = bn_param.get('eps', 1e-5)
	momentum = bn_param.get('momentum', 0.9)

	N, D = x.shape
	running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
	running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))

	out, cache = None, None
	if mode == 'train':
		# normalize data
		mu = np.mean(x, axis=0)
		var = np.var(x, axis=0)
		normalized = (x-mu)/np.sqrt(var+eps)
		out = gamma*normalized + beta
		# Update running mean and variance
		running_mean = momentum*running_mean + (1 - momentum)*mu
		running_var = momentum*running_var + (1 - momentum)*var
		# Cache for backwards pass
		cache = (x, normalized, gamma, beta, mu, var, eps)
	elif mode == 'test':
		# normalize data using running mean and variance from training
		normalized = (x - running_mean)/np.sqrt(running_var+eps)
		out = gamma*normalized + beta
	else:
		raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

	# Store the updated running means back into bn_param
	bn_param['running_mean'] = running_mean
	bn_param['running_var'] = running_var

	return out, cache


def batchnorm_backward(dout, cache):
	"""
	Backward pass for batch normalization.

	Inputs:
	- dout: Upstream derivatives, of shape (N, D)
	- cache: Variable of intermediates from batchnorm_forward.

	Returns a tuple of:
	- dx: Gradient with respect to inputs x, of shape (N, D)
	- dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
	- dbeta: Gradient with respect to shift parameter beta, of shape (D,)
	"""
	dx, dgamma, dbeta = None, None, None
	x, normalized, gamma, beta, mu, var, eps = cache
	N,D = dout.shape

	dx_norm = dout * gamma
	
	dx = (1. / N) * (1/np.sqrt(var + eps)) * (N*dx_norm - np.sum(dx_norm, axis=0) - normalized*np.sum(dx_norm*normalized, axis=0))
	
	dgamma = (dout * normalized).sum(axis = 0)
	dbeta = dout.sum(axis = 0)
	return dx, dgamma, dbeta

def affine_batchnorm_forward(x, w, b, gamma, beta, bn_params):
	a, fc_cache = affine_forward(x, w, b)
	c, batch_cache = batchnorm_forward(a, gamma, beta, bn_params)
	out, relu_cache = relu_forward(c)
	cache = (fc_cache, batch_cache, relu_cache)
	return out, cache

def affine_batchnorm_backward(dout, cache):
	fc_cache, batch_cache, relu_cache = cache
	da = relu_backward(dout, relu_cache)
	dc, dgamma, dbeta = batchnorm_backward(da, batch_cache)
	dx, dw, db = affine_backward(dc, fc_cache)
	return dx, dw, db, np.sum(dgamma), np.sum(dbeta)

