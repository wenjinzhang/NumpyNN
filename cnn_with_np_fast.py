import numpy as np

from network import Network
from layer import FCLayer, ActivationLayer, FlattenLayer, Conv2Layer,PoolLayer
from activations import relu, relu_prime
from losses import softmax_cross_entropy_with_logits, softmax_cross_entropy_with_logits_prime
from fast_cnn import FastConv, FastPool
from keras.datasets import mnist
from keras.utils import np_utils

# load MNIST from keras datasets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# training data : 60000 samples
# reshape and normalize input data
x_train = x_train.reshape(x_train.shape[0], 1, 28, 28)
x_train = x_train.astype('float32')
x_train /= 255

y_train = y_train.reshape(y_train.shape[0], 1)

x_test = x_test.reshape(x_test.shape[0], 1, 28, 28)
x_test = x_test.astype('float32')
x_test /= 255
# y_test = np_utils.to_categorical(y_test)
y_test = y_test.reshape(y_test.shape[0], 1)

# Network
net = Network(batch_size=128)
net.add(FastConv(1, 16, padding='same')) # input_shape=(bacth, 28, 28, 1)
net.add(ActivationLayer(relu, relu_prime))
net.add(FastPool(kernel_size=2, stride=2, mode="average"))
net.add(FastConv(16, 32, padding='same'))
net.add(ActivationLayer(relu, relu_prime))
net.add(FastPool(kernel_size=2, stride=2, mode="average"))
net.add(FlattenLayer())
net.add(FCLayer(7*7*32, 200))
net.add(ActivationLayer(relu, relu_prime))
net.add(FCLayer(200, 50))                   # input_shape=(batch, 200)      ;   output_shape=(batch, 50)
net.add(ActivationLayer(relu, relu_prime))
net.add(FCLayer(50, 10))                    # input_shape=(batch, 50)       ;   output_shape=(batch, 10)

net.use(softmax_cross_entropy_with_logits, softmax_cross_entropy_with_logits_prime)

net.fit(x_train, y_train, epochs=10, learning_rate=0.01, evaluation=(x_test, y_test), gamma=0.9)

