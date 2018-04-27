
from src.layers import *
from src.network.multilayernetwork import MLP

def test_layers():
	# Test layer properly instantiated
	layer = ConnectedLayer(2,3,1)
	print("Expect W size (2,3), got: (%d,%d)\n" %(layer.W.shape[0], layer.W.shape[1]))
	
	# Test output layer
	layer = OutputLayer(2,3,1)
	print("Expect W size (2,3), got: (%d,%d)\n" %(layer.W.shape[0], layer.W.shape[1]))

def test_network():
	# Test network construction
	network = MLP([2,3], 2, 2)
	print("Expect 3 layers total, got: %d" %len(network.layers))



if __name__ == "__main__":
	test_layers()
	test_network()