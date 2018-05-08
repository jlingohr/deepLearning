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

