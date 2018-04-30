import numpy as np
import random

from src.layers import ConnectedLayer, OutputLayer, LayerFactory
from src.loss import softmax_loss

class MLP(object):
	def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10, normalization=None,
		weight_scale=1e-2, reg=0.0, dtype=np.float32):
		'''
		- hidden_dims: A list of integers giving the size of each hidden layer.
	    - input_dim: An integer giving the size of the input.
	    - num_classes: An integer giving the number of classes to classify.
	    - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=0 then
	      the network should not use dropout at all.
	    - normalization: Whether or not the network should use batch normalization.
	    - reg: Scalar giving L2 regularization strength.
	    - weight_scale: Scalar giving the standard deviation for random
	      initialization of the weights.
	    - dtype: A numpy datatype object; all computations will be performed using
	      this datatype. float32 is faster but less accurate, so you should use
	      float64 for numeric gradient checking.
	    - seed: If not None, then pass this random seed to the dropout layers. This
	      will make the dropout layers deteriminstic so we can gradient check the
	      model.
      	'''
		self.num_layers = len(hidden_dims)+1
		self.dtype = dtype
		self.reg = reg
		self.normalization = normalization
		self.layers = self.__initialize_layers(hidden_dims, input_dim, num_classes, weight_scale)

	def loss(self, X, y=None):
		'''
		Compute loss and gradient for network. If y is None then run a forward pass
		and return an array of classification scores where scores[i,c] is the
		classification score for X[i] and class c. Otherwise run forward and
		backward pass an return a tuple (loss, grads) where loss is the scalar
		value of the loss function and grads contains the gradients of the loss
		for each layer.

		X: array of input data
		y: array of labels where y[i] gives the label for X[i]

		'''
		X = X.astype(self.dtype)
		mode = 'test' if y is None else 'train'
		params = {'mode':mode}

		scores = self.forward_prop(X, mode)
		# If test mode return early
		if y is None:
			return scores
		# Use backpropagation
		loss, grads = self.backward_prop(scores, y)
		return loss, grads


	def forward_prop(self, X, mode):
		'''
		Do forward pass through the network to compute the scores
		X: Array of input data
		'''
		scores = X
		for layer in self.layers:
			scores = layer.feed_forward(scores, mode)
		return scores

	def backward_prop(self, scores, y):
		'''
		Compute loss and gradients using backpropagation
		scores: Loss computed using forward propagation
		y: array of labels

		Returns regularized loss and gradients
		'''
		loss, dscores = softmax_loss(scores, y) 
		for layer in self.layers:
			loss += 0.5*self.reg*np.sum(layer.W**2)
		grads = {}
		
		delta, grad = self.layers[-1].feed_backward(dscores) 
		grads.update(grad)
		grads['W%d'%self.num_layers] += self.reg*self.layers[-1].W

		for l in range(2, self.num_layers+1):
			delta, grad = self.layers[-l].feed_backward(delta)
			grads.update(grad)
			grads['W%d'%(self.num_layers-l+1)] += self.reg*self.layers[-l].W
		return loss, grads

	def params(self):
		'''
		Return weights and coefficients of each layer in a parameter
		dictionary.
		Returns:
		params: dictionary of weights W and coefficients b
		'''
		params = {}
		for l in range(len(self.layers)):
			layer = self.layers[l]
			params.update(layer.params())
		return params

	def update_params(self, params):
		'''
		Update layers with new weights and biases
		params: Dictionary containing weights W and coefficients
		b.
		'''
		for l in range(len(self.layers)):
			p = l+1
			d = {k.replace(str(p),''):v for k, v in params.items() if str(p) in k}
			self.layers[l].update(d)


	def __initialize_layers(self, hidden_dims, input_dim, num_classes, weight_scale):
		factory = LayerFactory(weight_scale, self.dtype)
		dims = [input_dim] + hidden_dims + [num_classes]
		layers = []

		for l in range(self.num_layers-1):
			p = l+1
			layer = factory.make(p, dims[l], dims[l+1], self.normalization)
			layers.append(layer)

		layer = OutputLayer(self.num_layers, dims[-2], dims[-1], weight_scale, self.dtype)
		layers.append(layer)

		return layers