import numpy as np
import time
class Network:
    def __init__(self, batch_size=128):
        self.layers = []
        self.batch_size = batch_size
        self.loss = None
        self.loss_prime = None

    # add layer to network
    def add(self, layer):
        self.layers.append(layer)

    # set loss to use
    def use(self, loss, loss_prime):
        self.loss = loss
        self.loss_prime = loss_prime

    # predict output for given input
    def predict(self, input_data):
        # num of sample 
        samples = len(input_data)
        result = []

        for i in range(0, samples, self.batch_size):
            # forward propagation
            end_idx = min(i + self.batch_size, samples)
            batch_x = input_data[i:end_idx, :]
            output = batch_x
            for layer in self.layers:
                output = layer.forward_propagation(output)
            current = np.argmax(output, axis=1).reshape((-1, 1))
            result.append(current)

        result = np.concatenate(result, axis=0)

        return result

    # train the network
    def fit(self, x_train, y_train, epochs, learning_rate, print_interval=5, evaluation = None, gamma=0.9):
        # num of samples 
        samples = len(x_train)
        
        # training loop
        for i in range(epochs):
            start_time = time.time()
            err = 0
            batch_idx = 0
            train_result = []
            for j in range(0, samples, self.batch_size):
                # take a batch each time
                end_idx = min(j + self.batch_size, samples)
                batch_x = x_train[j:end_idx, :]
                batch_y = y_train[j:end_idx, :]
                
                # forward propagation
                output = batch_x
                # print("input shape", np.shape(output))
                for layer in self.layers:
                    output = layer.forward_propagation(output)

                current = np.argmax(output, axis=1).reshape((-1, 1))
                train_result.append(current)

                # print("output shape", output.shape)
                loss = self.loss(batch_y, output)
                
                # backward propagation
                gradient = self.loss_prime(batch_y, output)
                for layer in reversed(self.layers):
                    gradient = layer.backward_propagation(gradient, learning_rate, gamma)
                
                if batch_idx % print_interval == 0:
                     print('epoch %d/%d batch: %d error:%f' % (i+1, epochs, batch_idx,loss))
                batch_idx += 1
            train_result = np.concatenate(train_result, axis=0)
            
            train_accuracy = sum([y_ == y for y_, y in zip(train_result, y_train)])/y_train.shape[0] * 100

            # evaluate when getting evaluation dataset
            if evaluation != None:
                (test_x, test_y) = evaluation
                preds_y = self.predict(test_x)
                accuracy = sum([y_ == y for y_, y in zip(preds_y, test_y)])/test_x.shape[0] * 100
                # print('Epoch %d/%d train_accuracy: %f test_accuracy: %f time: %.2fs'% (i+1, epochs, learning_rate, train_accuracy, accuracy, time.time() - start_time))
                print("Epoch:[{}/{}] Train_accuracy: {:.2f}%; Test_accuracy:{:.2f}%; Time: {:.2f}s".format(
                i+1, epochs, train_accuracy[0],  accuracy[0], time.time() - start_time))
                

