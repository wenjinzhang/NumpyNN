from layer import Layer

# inherit from base class Layer
class FlattenLayer(Layer):
    # returns the flattened input
    def forward_propagation(self, input_data):
        self.input = input_data
        self.output = input_data.flatten().reshape((1,-1))
        return self.output

    # Returns input_error=dE/dX for a given upstream_gradient=dE/dY.
    # learning_rate is not used because there is no "learnable" parameters.
    def backward_propagation(self, upstream_gradient, learning_rate):
        return upstream_gradient.reshape(self.input.shape)