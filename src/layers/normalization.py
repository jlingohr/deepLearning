class BatchNormLayer(ConnectedLayer):
	activation_forward = None
	activation_back = None

	def __init__(self, level, input_size, output_size, weight_scale, dtype):
		'''
		Special layer to use when using batch normalization
		'''
		ConnectedLayer.__init__(self, level, input_size, output_size, weight_scale, dtype)
		self.gamma = np.ones((1,1))
		self.beta = np.zeros((1,1))
		self.activation_forward = affine_batchnorm_forward
		self.activation_back = affine_batchnorm_backward
		self.bn_params = {}

	def feed_forward(self, x, mode):
		self.bn_params['mode'] = mode
		out, cache = self.activation_forward(x, self.W, self.b, self.gamma, self.beta, self.bn_params)
		self.cache = cache
		return out

	def feed_backward(self, dscores):
		grads = {}
		dx, dw, db, dgamma, dbeta = self.activation_back(dscores, self.cache)
		grads['W%d'%self.level] = dw
		grads['b%d'%self.level] = db
		grads['gamma%d'%self.level] = dgamma
		grads['beta%d'%self.level] = dbeta
		return dx, grads

	def params(self):
		params = ConnectedLayer.params(self)
		params['gamma%d'%self.level] = self.gamma
		params['beta%d'%self.level] = self.beta
		return params

	def update(self, params):
		ConnectedLayer.update(self, params)
		self.gamma = params['gamma']
		self.beta = params['beta']