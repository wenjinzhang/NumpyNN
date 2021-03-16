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
    def backward_propagation(self, upstream_gradient, learning_rate, gamma):
        raise NotImplementedError


# inherit from base class Layer
class FCLayer(Layer):
    # input_size = number of input neurons
    # output_size = number of output neurons
    def __init__(self, input_size, output_size):
        np.random.seed(138)
        # self.weights = np.random.rand(input_size, output_size) - 0.5
        # self.bias = np.random.rand(1, output_size) - 0.5
        self.weights = np.random.randn(input_size, output_size) * np.sqrt(1. / input_size)
        self.bias = np.zeros((1, output_size)) * np.sqrt(1. / input_size)

        self.V_dW = np.zeros(self.weights.shape)
        self.V_dB = np.zeros(self.bias.shape)

    # returns output for a given input
    def forward_propagation(self, input_data):
        self.input = input_data
        self.output = np.dot(self.input, self.weights) + self.bias
        return self.output

    # computes dE/dW, dE/dB for a given upstream_gradient=dE/dY. Returns input_error=dE/dX.
    def backward_propagation(self, upstream_gradient, learning_rate=0.1, gamma=0.9):
        # print("upstream gradient",upstream_gradient.shape)
        # print("weight",self.weights.shape)
        # print("bias",self.bias.shape)
        # print("input", self.input.shape)
        
        dHidden = np.dot(upstream_gradient, self.weights.T)
        dW = np.dot(self.input.T, upstream_gradient)
        dB = np.sum(upstream_gradient, axis=0, keepdims=True)

        # print("local gradient",dHidden.shape)
        # print("weight",dW.shape)
        # print("bias",dB.shape)

        # update parameter with gradient
        # self.weights -= learning_rate * dW
        # self.bias -= learning_rate * dB

        # implement momentum algorithm
        # https://gluon.mxnet.io/chapter06_optimization/momentum-scratch.html
        self.V_dW = gamma * self.V_dW + dW
        self.V_dB = gamma * self.V_dB + dB
        
        # update parameters with velocity
        self.weights -= learning_rate * self.V_dW
        self.bias -= learning_rate * self.V_dB

        return dHidden


# inherit from base class Layer
class ActivationLayer(Layer):
    def __init__(self, activation, activation_prime):
        self.activation = activation
        self.activation_prime = activation_prime

    # returns the activated input
    def forward_propagation(self, input_data):
        self.input = input_data
        self.output = self.activation(self.input)
        return self.output

    # Returns gradient=dE/dX for a given upstream_gradient=dE/dY.
    # learning_rate is not used because there is no "learnable" parameters.
    def backward_propagation(self, upstream_gradient, learning_rate, gamma):
        return self.activation_prime(self.input) * upstream_gradient

