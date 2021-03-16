import numpy as np

# activation function and its derivative
def tanh(x):
    return np.tanh(x)

def tanh_prime(x):
    return 1-np.tanh(x)**2

def relu(x):
    return np.maximum(x, 0)
    
def relu_prime(x):
    # 0 when x < 0; 1 when x > 0
    return np.array(x >= 0).astype('int')