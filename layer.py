import numpy as np

# Base class
class Layer:
    def __init__(self):
        self.input = None
        self.output = None

    # computes the output Y of a layer for a given input X
    def forward_propagation(self, input):
        raise NotImplementedError

    # computes dE/dX for a given dE/dY (and update parameters if any)
    def backward_propagation(self, upstream_gradient, learning_rate):
        raise NotImplementedError


class SoftmaxLayer:
    def __init__(self, input_size):
        self.input_size = input_size
    
    def forward(self, input):
        # input size [#batch, -1]
        self.input = input
        tmp = np.exp(self.input)
        self.output = tmp / np.sum(tmp, axis=1, keepdims=Ture) # [#batch, units]
        return self.output
    
    def backward(self, upstream_gradient, learning_rate):
        input_error = np.zeros(upstream_gradient.shape)
        out = np.tile(self.output.T, self.input_size)
        return self.output * np.dot(upstream_gradient, np.identity(self.input_size) - out)
