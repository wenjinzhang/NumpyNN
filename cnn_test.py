from layer import Layer
import numpy as np

class FastCNN(Layer):

    def __init__(self, f_in, f_out, kernel_size=3, strides=1, padding="valid", use_bias=True):
        np.random.seed(138)

        self.f_in = f_in
        self.f_out = f_out
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.use_bias = True

        # 4+D tensor with shape: (K, K, f_in, f_out)
        self.filter_shape = (kernel_size, kernel_size, f_in, f_out)
        self.bias_shape = (f_out)

        self.filters = np.random.uniform(-0.05, 0.05, size=self.filter_shape)
        self.bias = np.random.uniform(-0.05, 0.05, size=self.bias_shape)

        self.V_dW = np.zeros(self.filters.shape)
        self.V_dB = np.zeros(self.bias.shape)

    # computes the output Y of a layer for a given input X
    def forward_propagation(self, X):
        # 4+D tensor with shape: batch_shape + (rows, cols, f_in)
        (batch_size, prev_height, prev_width, prev_channels) = X.shape
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
        print(pad, n_H, n_W)
        
        # output metrix
        Z = np.zeros(shape=(batch_size, n_H, n_W, self.f_out))

        X_pad = self.pad_inputs(X, (pad, pad))

        print(X_pad.shape)
        for i in range(batch_size):
            im = X_pad[i,:,:,:]
            # print("im", im.shape)
            im_col = self.im2col(im, self.kernel_size, self.kernel_size, self.strides)
            filter_col = np.reshape(self.filters,(-1, self.f_out))
            # print("im_col", im_col.shape)
            # print("filter_col", filter_col.shape)
            # print("b", self.bias.shape)
            mul = im_col.dot(filter_col) + self.bias
            Z[i,:,:,:] = self.col2im(mul, n_H, n_W)
        
        return Z

    # computes dE/dX for a given dE/dY (and update parameters if any)
    def backward_propagation(self, dZ, learning_rate, gamma):

        (batch_size, prev_height, prev_width, prev_channels) = X.shape
        x = self.input


        col_dZ = np.reshape(dZ, [self.batchsize, -1, self.output_channels])

        for i in range(self.batchsize):
            self.w_gradient += np.dot(self.col_image[i].T, col_dZ[i]).reshape(self.filters.shape)

        self.b_gradient += np.sum(col_eta, axis=(0, 1))

        dA = np.zeros(self.input.shape)



    def pad_inputs(self, X, pad):
        '''
        Function to apply zero padding to the image
        :param X:[numpy array]: Dataset of shape (m, height, width, depth)
        :param pad:[int]: number of columns to pad
        :return:[numpy array]: padded dataset
        '''
        return np.pad(X, ((0, 0), (pad[0], pad[0]), (pad[1], pad[1]), (0, 0)), 'constant')
        
    def im2col(self, x, hh, ww, stride):

        """
        Args:
        x: image matrix to be translated into columns, (H,W, f_in)
        hh: filter height
        ww: filter width
        stride: stride
        Returns:
        col: (new_h*new_w, hh*ww*f_in) matrix, each column is a cube that will convolve with a filter
                new_h = (H-hh) // stride + 1, new_w = (W-ww) // stride + 1
        """
        h, w, f_in = x.shape
        new_h = (h-hh) // stride + 1
        new_w = (w-ww) // stride + 1
        col = np.zeros([new_h*new_w, f_in*hh*ww])

        for i in range(new_h):
            for j in range(new_w):
                patch = x[ i*stride:i*stride+hh, j*stride:j*stride+ww, ...] 
                col[i*new_w+j,:] = np.reshape(patch,-1)
        return col

    def col2im(self, mul, h_new, w_new):
        """
        Args:
        mul: (h_new*w_new, f_out) matrix, each col should be reshaped to C*h_new*w_new when f_out>0, or h_new*w_new when f_out = 0
        h_new: reshaped filter height
        w_new: reshaped filter width
        f_out: reshaped filter channel, if 0, reshape the filter to 2D, Otherwise reshape it to 3D
        Returns:
        (h_new, w_new, f_out) matrix
        """
        return np.reshape(mul,(h_new, w_new, -1))

if __name__ == "__main__":
    img = np.ones((128, 28, 28, 1))
    conv = FastCNN(1, 3, padding="same")
    next = conv.forward_propagation(img)
    