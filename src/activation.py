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
	D = np.prod(x.shape[1:])
	x2 = x.reshape(x.shape[0], D)
	out = x2@w + b
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

    x2 = x.reshape(x.shape[0], np.prod(x.shape[1:]))
    dx = (dout@w.T).reshape(x.shape)
    dw = x2.T@dout
    db = np.sum(dout.T, axis=1)
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