from layer import Layer

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

    # Returns input_error=dE/dX for a given upstream_gradient=dE/dY.
    # learning_rate is not used because there is no "learnable" parameters.
    def backward_propagation(self, upstream_gradient, learning_rate):
        return self.activation_prime(self.input) * upstream_gradient
