import numpy as np

# loss function and its derivative
def mse(y_true, y_pred):
    return np.mean(np.power(y_true-y_pred, 2))

def mse_prime(y_true, y_pred):
    return 2*(y_pred-y_true)/y_true.size

def softmax_cross_entropy_with_logits(y_true, y_pred):
    # y_true shape: [batch, units] default[128,10]
    # y_pred shape: [batch, 1] default[128, 1] no one-hot
    batch_size = np.shape(y_pred)[0]
    probs = softmax(y_pred)
    y = np.reshape(y_true, (batch_size,))
    correct_logprobs = -np.log(probs[range(batch_size), y])
    data_loss =np.sum(correct_logprobs) / batch_size
    return data_loss

def softmax_cross_entropy_with_logits_prime(y_true, y_pred):
    batch_size = np.shape(y_pred)[0]
    y = np.reshape(y_true, (batch_size,))
    gradient = softmax(y_pred)
    gradient[range(batch_size), y] -= 1
    gradient /= batch_size
    return gradient

def softmax(x):
    # input size [#batch, -1]
    tmp = np.exp(x)
    output = tmp / np.sum(tmp, axis=1, keepdims=True) # [#batch, units]
    return output

def stable_softmax(X):
    exps = np.exp(X - np.max(X))
    return exps / np.sum(exps, axis=1, keepdims=True)