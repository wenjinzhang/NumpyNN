from layer import Layer
import numpy as np

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

    # returns output for a given input
    def forward_propagation(self, input_data):
        self.input = input_data
        self.output = np.dot(self.input, self.weights) + self.bias
        return self.output

    # computes dE/dW, dE/dB for a given upstream_gradient=dE/dY. Returns input_error=dE/dX.
    def backward_propagation(self, upstream_gradient, learning_rate):
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

        # input_error = np.dot(upstream_gradient, self.weights.T)
        # weights_error = np.dot(self.input.T, upstream_gradient)
        # dBias = np.sum(upstream_gradient, axis=0, keepdims=True)
        # update parameters
        self.weights -= learning_rate * dW
        self.bias -= learning_rate * dB
        return dHidden