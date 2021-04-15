from layer import Layer
import numpy as np

class FastConv(Layer):

    def __init__(self, f_in, f_out, kernel_size=3, strides=1, padding="valid", use_bias=True):
        np.random.seed(138)

        self.f_in = f_in
        self.f_out = f_out
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.use_bias = True

        # 4+D tensor with shape: (K, K, f_in, f_out)
        self.filter_shape = (f_out, f_in, kernel_size, kernel_size)
        self.bias_shape = (f_out)

        self.filters = np.random.uniform(-0.05, 0.05, size=self.filter_shape)
        self.bias = np.random.uniform(-0.05, 0.05, size=self.bias_shape)

        self.V_dW = np.zeros(self.filters.shape)
        self.V_dB = np.zeros(self.bias.shape)


    # computes the output Y of a layer for a given input X
    def forward_propagation(self, X):
        # 4+D tensor with shape: batch_shape + (f_in, rows, cols)
        (batch_size, prev_channels, prev_height, prev_width) = X.shape
        self.input = X
       
        if self.padding == 'same':
            # p = (K – 1) / 2
            pad = int((self.kernel_size - 1) / 2)
            n_H = prev_height
            n_W = prev_width
        else:
            pad = 0
            n_H = int((prev_height - self.kernel_size) /self.strides) + 1
            n_W = int((prev_width - self.kernel_size) /self.strides) + 1

        self.pad= pad
        self.n_H , self.n_W = n_H, n_W
        
        # output metrix
        Z = np.zeros(shape=(batch_size, n_H, n_W, self.f_out))

        n_C = self.f_out
        X_col = im2col(X, self.kernel_size, self.kernel_size, self.strides, self.pad)

        w_col = self.filters.reshape((self.f_out, -1))
        b_col = self.bias.reshape(-1, 1)
        # Perform matrix multiplication.
        out = w_col @ X_col + b_col
        # Reshape back matrix to image.
        out = np.array(np.hsplit(out, batch_size)).reshape((batch_size, n_C, n_H, n_W))
        self.cache = X, X_col, w_col

        return out
        

    # computes dE/dX for a given dE/dY (and update parameters if any)
    def backward_propagation(self, dout, learning_rate=0.01, gamma=0.99):
        """
            Distributes error from previous layer to convolutional layer and
            compute error for the current convolutional layer.
            Parameters:
            - dout: error from previous layer.
            
            Returns:
            - dX: error of the current convolutional layer.
            - self.W['grad']: weights gradient.
            - self.b['grad']: bias gradient.
        """
        X, X_col, w_col = self.cache
        batch_size, _, _, _ = X.shape

        # Compute bias gradient.
        self.dB = np.sum(dout, axis=(0,2,3))

        # Reshape dout properly.
        dout = dout.reshape(dout.shape[0] * dout.shape[1], dout.shape[2] * dout.shape[3])
        dout = np.array(np.vsplit(dout, batch_size))
        dout = np.concatenate(dout, axis=-1)
        # Perform matrix multiplication between reshaped dout and w_col to get dX_col.
        dX_col = w_col.T @ dout
        # Perform matrix multiplication between reshaped dout and X_col to get dW_col.
        dw_col = dout @ X_col.T
        # Reshape back to image (col2im).
        dX = col2im(dX_col, X.shape, self.kernel_size, self.kernel_size, self.strides, self.pad)
        # Reshape dw_col into dw.
        self.dW = dw_col.reshape(self.filter_shape)
        # print("dw shape", np.shape(self.dW))
        # print("w shape", self.filters.shape)
        # print("db shape", np.shape(self.dB))
        # print("b shape", self.bias.shape)
        # print("x", np.shape(dX))

        self.V_dW = gamma * self.V_dW + self.dW
        self.V_dB = gamma * self.V_dB + self.dB
        # gradient descent with momentum
        self.filters -= learning_rate * self.V_dW
        self.bias -= learning_rate * self.V_dB
        return dX


class FastPool(Layer):
    # input_size = number of input neurons
    # output_size = number of output neurons
    def __init__(self, kernel_size=2, stride=2, mode="max", padding="valide", name=None):
        np.random.seed(0)
        self.kernel_size = kernel_size
        self.stride = stride
        self.mode = mode
        self.name = name
        self.padding = padding

    # returns output for a given input
    def forward_propagation(self, X):
        # self.input = input_data
        # self.output = np.dot(self.input, self.weights) + self.bias
        (batch_size, prev_channels, prev_height, prev_width) = X.shape
        """
        Apply average pooling.

        Parameters:
        - X: Output of activation function.

        Returns:
        - A_pool: X after average pooling layer. 
        """

        if self.padding == 'same':
            # p = (K – 1) / 2
            pad = int((self.kernel_size - 1) / 2)
            n_H = prev_height
            n_W = prev_width
        else:
            pad = 0
            n_H = int((prev_height - self.kernel_size) /self.stride) + 1
            n_W = int((prev_width - self.kernel_size) /self.stride) + 1

        self.pad= pad
        self.n_H, self.n_W = n_H, n_W
        self.cache = X
        m, n_C_prev, n_H_prev, n_W_prev = X.shape
        n_C = n_C_prev

        X_col = im2col(X, self.kernel_size, self.kernel_size, self.stride, self.pad)
        X_col = X_col.reshape(n_C, X_col.shape[0]//n_C, -1)
        if self.mode == 'average':
            A_pool = np.mean(X_col, axis=1)
        else:
            A_pool = np.max(X_col, axis=1)
        # Reshape A_pool properly.
        A_pool = np.array(np.hsplit(A_pool, m))
        A_pool = A_pool.reshape(m, n_C, n_H, n_W)

        return A_pool

    # computes dE/dW, dE/dB for a given upstream_gradient=dE/dY. Returns dE/dX.
    def backward_propagation(self, dout, learning_rate=0.1, gamma=0.9):
        X = self.cache
        m, n_C_prev, n_H_prev, n_W_prev = X.shape
        n_C = n_C_prev
        
        dout_flatten = dout.reshape(n_C, -1) / (self.kernel_size * self.kernel_size)
        dX_col = np.repeat(dout_flatten, self.kernel_size*self.kernel_size, axis=0)
        dX = col2im(dX_col, X.shape, self.kernel_size, self.kernel_size, self.stride, self.pad)

        # Reshape dX properly.
        dX = dX.reshape(m, -1)
        dX = np.array(np.hsplit(dX, n_C_prev))
        dX = dX.reshape(m, n_C_prev, n_H_prev, n_W_prev)
        return dX


def im2col(X, HF, WF, stride, pad):
    """
        Transforms our input image into a matrix.

        Parameters:
        - X: input image.
        - HF: filter height.
        - WF: filter width.
        - stride: stride value.
        - pad: padding value.

        Returns:
        -cols: output matrix.
    """
    # Padding
    X_padded = np.pad(X, ((0,0), (0,0), (pad, pad), (pad, pad)), mode='constant')
    i, j, d = get_indices(X.shape, HF, WF, stride, pad)
    # Multi-dimensional arrays indexing.
    cols = X_padded[:, d, i, j]
    cols = np.concatenate(cols, axis=-1)
    return cols

def col2im(dX_col, X_shape, HF, WF, stride, pad):
    """
        Transform our matrix back to the input image.

        Parameters:
        - dX_col: matrix with error.
        - X_shape: input image shape.
        - HF: filter height.
        - WF: filter width.
        - stride: stride value.
        - pad: padding value.

        Returns:
        -x_padded: input image with error.
    """
    # Get input size
    N, D, H, W = X_shape
    # Add padding if needed.
    H_padded, W_padded = H + 2 * pad, W + 2 * pad
    X_padded = np.zeros((N, D, H_padded, W_padded))
    
    # Index matrices, necessary to transform our input image into a matrix. 
    i, j, d = get_indices(X_shape, HF, WF, stride, pad)
    # Retrieve batch dimension by spliting dX_col N times: (X, Y) => (N, X, Y)
    dX_col_reshaped = np.array(np.hsplit(dX_col, N))
    # Reshape our matrix back to image.
    # slice(None) is used to produce the [::] effect which means "for every elements".
    np.add.at(X_padded, (slice(None), d, i, j), dX_col_reshaped)

    # Remove padding from new image if needed.
    if pad == 0:
        return X_padded
    elif type(pad) is int:
        return X_padded[:, :, pad:-pad, pad:-pad ]

def get_indices(X_shape, HF, WF, stride, pad):
    """
        Returns index matrices in order to transform our input image into a matrix.

        Parameters:
        -X_shape: Input image shape.
        -HF: filter height.
        -WF: filter width.
        -stride: stride value.
        -pad: padding value.

        Returns:
        -i: matrix of index i.
        -j: matrix of index j.
        -d: matrix of index d. 
            (Use to mark delimitation for each channel
            during multi-dimensional arrays indexing).
    """
    # get input size
    m, n_C, n_H, n_W = X_shape

    # get output size
    out_h = int((n_H + 2 * pad - HF) / stride) + 1
    out_w = int((n_W + 2 * pad - WF) / stride) + 1

    # ----Compute matrix of index i----

    # Level 1 vector.
    level1 = np.repeat(np.arange(HF), WF)
    # Duplicate for the other channels.
    level1 = np.tile(level1, n_C)
    # Create a vector with an increase by 1 at each level.
    everyLevels = stride * np.repeat(np.arange(out_h), out_w)
    # Create matrix of index i at every levels for each channel.
    i = level1.reshape(-1, 1) + everyLevels.reshape(1, -1)

    # ----Compute matrix of index j----
    
    # Slide 1 vector.
    slide1 = np.tile(np.arange(WF), HF)
    # Duplicate for the other channels.
    slide1 = np.tile(slide1, n_C)
    # Create a vector with an increase by 1 at each slide.
    everySlides = stride * np.tile(np.arange(out_w), out_h)
    # Create matrix of index j at every slides for each channel.
    j = slide1.reshape(-1, 1) + everySlides.reshape(1, -1)

    # ----Compute matrix of index d----

    # This is to mark delimitation for each channel
    # during multi-dimensional arrays indexing.
    d = np.repeat(np.arange(n_C), HF * WF).reshape(-1, 1)
    return i, j, d

if __name__ == "__main__":
    img = np.ones((128, 1, 28, 28))
    conv = FastCNN(1, 3, padding="same")
    next = conv.forward_propagation(img)
    print(np.shape(next))
    conv.backward_propagation(next)