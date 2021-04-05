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

    # computes dE/dW, dE/dB for a given upstream_gradient=dE/dY. Returns dE/dX.
    def backward_propagation(self, upstream_gradient, learning_rate=0.1, gamma=0.9):
        dHidden = np.dot(upstream_gradient, self.weights.T)
        dW = np.dot(self.input.T, upstream_gradient)
        dB = np.sum(upstream_gradient, axis=0, keepdims=True)
        # implement momentum algorithm
        # https://gluon.mxnet.io/chapter06_optimization/momentum-scratch.html
        self.V_dW = gamma * self.V_dW + dW
        self.V_dB = gamma * self.V_dB + dB
        # gradient descent with momentum
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


class Conv2Layer(Layer):
    # input_size = number of input neurons
    # output_size = number of output neurons
    def __init__(self, f_in, f_out, kernel_size=3, strides=1, padding="valid", use_bias=True):
        np.random.seed(138)

        self.f_in = f_in
        self.f_out = f_out
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.use_bias = True

        self.filter_shape = (kernel_size, kernel_size, f_in, f_out)
        self.bias_shape = (1, 1, 1, f_out)

        self.filters = np.random.uniform(-0.05, 0.05, size=self.filter_shape)
        self.bias = np.random.uniform(-0.05, 0.05, size=self.bias_shape)

        # self.weights = np.random.randn(input_size, output_size) * np.sqrt(1. / input_size)
        # self.bias = np.zeros((1, output_size)) * np.sqrt(1. / input_size)

        self.V_dW = np.zeros(self.filters.shape)
        self.V_dB = np.zeros(self.bias.shape)


    def init_grad(self):
        self.dW = np.zeros(self.filters.shape)
        self.dB = np.zeros(self.bias.shape)


    # returns output for a given input
    def forward_propagation(self, X):
        # 4+D tensor with shape: batch_shape + (rows, cols, channels)
        (batch_size, prev_height, prev_width, prev_channels) = X.shape

        self.input = X

        # following code can be improve
        if self.padding == 'same':
            pad_h = int(((prev_height - 1)*self.strides + self.kernel_size - prev_height) / 2)
            pad_w = int(((prev_width - 1)*self.strides + self.kernel_size - prev_width) / 2)
            n_H = prev_height
            n_W = prev_width
        else:
            pad_h = 0
            pad_w = 0
            n_H = int((prev_height - self.kernel_size) /self.strides) + 1
            n_W = int((prev_width - self.kernel_size) /self.strides) + 1

        self.pad_h, self.pad_w = pad_h, pad_w

        # output metrix
        Z = np.zeros(shape=(batch_size, n_H, n_W, self.f_out))
        X_pad = pad_inputs(X, (pad_h, pad_w))

        for i in range(batch_size):
            x = X_pad[i]
            for h in range(n_H):
                for w in range(n_W):
                    vert_start = self.strides * h
                    vert_end = vert_start + self.kernel_size
                    horiz_start = self.strides * w
                    horiz_end = horiz_start + self.kernel_size

                    for c in range(self.f_out):
                        x_slice = x[vert_start: vert_end, horiz_start: horiz_end, :]

                        Z[i, h, w, c] = conv_single_step(x_slice, self.filters[:, :, :, c], self.bias[:, :, :, c])

        return Z

    # computes dE/dW, dE/dB for a given upstream_gradient=dE/dY. Returns dE/dX.
    def backward_propagation(self, dZ, learning_rate=0.1, gamma=0.9):
        # dHidden = np.dot(upstream_gradient, self.weights.T)
        # dW = np.dot(self.input.T, upstream_gradient)
        # dB = np.sum(upstream_gradient, axis=0, keepdims=True)
        
        (batch_size, prev_height, prev_width, prev_channels) = self.input.shape

        dA = np.zeros((batch_size, prev_height, prev_width, prev_channels))
        self.init_grad()

        A_pad = pad_inputs(self.input, (self.pad_h, self.pad_w))
        dA_pad = pad_inputs(dA, (self.pad_h, self.pad_w))

        for i in range(batch_size):
            a_pad = A_pad[i]
            da_pad = dA_pad[i]

            for h in range(prev_height):
                for w in range(prev_width):
                    vert_start =  self.strides * h
                    vert_end = vert_start + self.kernel_size
                    horiz_start =  self.strides * w
                    horiz_end = horiz_start + self.kernel_size

                    for c in range(self.f_out):
                        a_slice = a_pad[vert_start: vert_end, horiz_start: horiz_end, :]
                        da_pad[vert_start:vert_end, horiz_start:horiz_end, :] += self.filters[:, :, :, c] * dZ[i, h, w, c]
                        self.dW[:, :, :, c] += a_slice * dZ[i, h, w, c]
                        self.dB[:, :, :, c] += dZ[i, h, w, c]
            dA[i, :, :, :] = da_pad[self.pad_h: -self.pad_h, self.pad_w: -self.pad_w, :]

        # apply gradiet
        # implement momentum algorithm
        # https://gluon.mxnet.io/chapter06_optimization/momentum-scratch.html
        self.V_dW = gamma * self.V_dW + self.dW
        self.V_dB = gamma * self.V_dB + self.dB
        # gradient descent with momentum
        self.filters -= learning_rate * self.V_dW
        self.bias -= learning_rate * self.V_dB
        return dA


def conv_single_step(input, weights, bias):
    '''
        Function to apply one filter to input slice.
        :param input:[numpy array]: slice of input data of shape (f, f, n_C_prev)
        :param W:[numpy array]: One filter of shape (f, f, n_C_prev)
        :param b:[numpy array]: Bias value for the filter. Shape (1, 1, 1)
        :return:
    '''
    # print(np.shape(input))
    # print(np.shape(weights))
    # print(np.shape(bias)
    return np.sum(np.multiply(input, weights)) + float(bias)


def pad_inputs(X, pad):
    '''
    Function to apply zero padding to the image
    :param X:[numpy array]: Dataset of shape (m, height, width, depth)
    :param pad:[int]: number of columns to pad
    :return:[numpy array]: padded dataset
    '''
    return np.pad(X, ((0, 0), (pad[0], pad[0]), (pad[1], pad[1]), (0, 0)), 'constant')


class PoolLayer(Layer):
    # input_size = number of input neurons
    # output_size = number of output neurons
    def __init__(self, kernel_shape=2, stride=1, mode="max", name=None):
        np.random.seed(0)
        self.kernel_shape = kernel_shape
        self.stride = stride
        self.mode = mode
        self.name = name

    # returns output for a given input
    def forward_propagation(self, X):
        # self.input = input_data
        # self.output = np.dot(self.input, self.weights) + self.bias
        self.input = X
        (batch_size, prev_height, prev_width, prev_channels) = X.shape
        filter_shape_h, filter_shape_w = self.kernel_shape, self.kernel_shape

        n_H = int(1 + (prev_height - filter_shape_h) / self.stride)
        n_W = int(1 + (prev_width - filter_shape_w) / self.stride)
        n_C = prev_channels

        A = np.zeros((batch_size, n_H, n_W, n_C))

        for i in range(batch_size):
            for h in range(n_H):
                for w in range(n_W):

                    vert_start = h * self.stride
                    vert_end = vert_start + filter_shape_h
                    horiz_start = w * self.stride
                    horiz_end = horiz_start + filter_shape_w

                    for c in range(n_C):

                        if self.mode == 'average':
                            A[i, h, w, c] = np.mean(X[i, vert_start: vert_end, horiz_start: horiz_end, c])
                        else:
                            A[i, h, w, c] = np.max(X[i, vert_start: vert_end, horiz_start: horiz_end, c])

        return A
    

    # computes dE/dW, dE/dB for a given upstream_gradient=dE/dY. Returns dE/dX.
    def backward_propagation(self, dZ, learning_rate=0.1, gamma=0.9):
        filter_shape_h, filter_shape_w = self.kernel_shape, self.kernel_shape

        (batch_size, prev_height, prev_width, prev_channels) = self.input.shape
        m, n_H, n_W, n_C = dZ.shape

        dA = np.zeros(shape=(batch_size, prev_height, prev_width, prev_channels))

        for i in range(batch_size):
            a = self.input[i]

            for h in range(n_H):
                for w in range(n_W):

                    vert_start = h * self.stride
                    vert_end = vert_start + filter_shape_h
                    horiz_start = w * self.stride
                    horiz_end = horiz_start + filter_shape_w

                    for c in range(n_C):

                        if self.mode == 'average':
                            da = dZ[i, h, w, c]
                            dA[i, vert_start: vert_end, horiz_start: horiz_end, c] += \
                                self.distribute_value(da, (self.kernel_shape, self.kernel_shape))

                        else:
                            a_slice = a[vert_start: vert_end, horiz_start: horiz_end, c]
                            mask = self.create_mask(a_slice)
                            dA[i, vert_start: vert_end, horiz_start: horiz_end, c] += \
                                dZ[i, h, w, c] * mask

        return dA


    def distribute_value(self, dz, shape):
        (n_H, n_W) = shape
        average = 1 / (n_H * n_W)
        return np.ones(shape) * dz * average

    def create_mask(self, x):
        return x == np.max(x)


class FlattenLayer(Layer):

    # computes the output Y of a layer for a given input X
    def forward_propagation(self, input):
        self.shape = input.shape
        data = np.ravel(input).reshape(self.shape[0], -1)
        return data

    # computes dE/dX for a given dE/dY (and update parameters if any)
    def backward_propagation(self, upstream_gradient, learning_rate=0.1, gamma=0.9):
        return upstream_gradient.reshape(self.shape)