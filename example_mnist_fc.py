import numpy as np

from network import Network
from fc_layer import FCLayer
from activation_layer import ActivationLayer
from activations import relu, relu_prime
from losses import softmax_cross_entropy_with_logits, softmax_cross_entropy_with_logits_prime

from keras.datasets import mnist
from keras.utils import np_utils

# load MNIST from server
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# training data : 60000 samples
# reshape and normalize input data
x_train = x_train.reshape(x_train.shape[0], 28*28)
x_train = x_train.astype('float32')
x_train /= 255
# encode output which is a number in range [0,9] into a vector of size 10
# e.g. number 3 will become [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
# y_train = np_utils.to_categorical(y_train)
y_train = y_train.reshape(y_train.shape[0], 1)


x_test = x_test.reshape(x_test.shape[0], 28*28)
x_test = x_test.astype('float32')
x_test /= 255
# y_test = np_utils.to_categorical(y_test)
y_test = y_test.reshape(y_test.shape[0], 1)
# Network
net = Network()
net.add(FCLayer(784, 200))                # input_shape=(bacth, 784)    ;   output_shape=(batch, 200)
net.add(ActivationLayer(relu, relu_prime))
net.add(FCLayer(200, 50))                   # input_shape=(batch, 200)      ;   output_shape=(batch, 50)
net.add(ActivationLayer(relu, relu_prime))
net.add(FCLayer(50, 10))                    # input_shape=(batch, 50)       ;   output_shape=(batch, 10)

net.use(softmax_cross_entropy_with_logits, softmax_cross_entropy_with_logits_prime)

net.fit(x_train, y_train, epochs=40, learning_rate=0.01, evaluation=(x_test, y_test))

# out = net.predict(x_test[0:10])
# print("\n")
# print("predicted values : ")
# print(out, end="\n")
# print("true values : ")
# print(y_test[0:3])
