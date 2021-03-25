# NumpyNN
## Building Neural Networks with only numpy

Now it supports fc layer, relu, tanh, softmax, mse, cross_entropy, Mini-batch gradient descent without momentum and Mini-batch gradient descent with momentum
## How to build neural network with NumpyNN?
Create a fc neural network with Network class
```
net = Network()

# input_shape=(bacth, 784)
net.add(FCLayer(784, 200)) 

net.add(ActivationLayer(relu, relu_prime))

net.add(FCLayer(200, 50))

net.add(ActivationLayer(relu, relu_prime))

net.add(FCLayer(50, 10)) 

net.use(softmax_cross_entropy_with_logits, softmax_cross_entropy_with_logits_prime)
```

Training nueral network
```
net.fit(x_train, y_train, epochs=10, learning_rate=0.1, evaluation=(x_test, y_test), gamma=0.9)
```

## Example: testing MNIST dataset
*You may need to install keras for downloading mnist dataset*
```
#test on MNIST dataset
python nn_with_np.py

# compare with keras 
python nn_with_keras.py

# compare with pytorch
python nn_with_pytorch.py
```
 Result:
 ![Image](https://github.com/wenjinzhang/NumpyNN/blob/master/test.png)



