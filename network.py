import numpy as np

class Network:
    def __init__(self, batch_size=128):
        self.layers = []
        self.batch_size = batch_size
        self.loss = None
        self.loss_prime = None
        self.beta = 0.9

    # add layer to network
    def add(self, layer):
        self.layers.append(layer)

    # set loss to use
    def use(self, loss, loss_prime):
        self.loss = loss
        self.loss_prime = loss_prime

    # predict output for given input
    def predict(self, input_data):
        # sample dimension first
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
    def fit(self, x_train, y_train, epochs, learning_rate, print_interval=100, evaluation = None):
        # sample dimension first
        samples = len(x_train)
        # print("trainx_shape", np.shape(x_train))
        # print("trainy_shape", np.shape(y_train))
        # training loop
        last_accuacy = 0
        for i in range(epochs):
            err = 0
            batch_idx = 0
            for j in range(0, samples, self.batch_size):
                # take a batch each time
                end_idx = min(j + self.batch_size, samples)
                batch_x = x_train[j:end_idx, :]
                batch_y = y_train[j:end_idx, :]
                # print("j:", j, ",end_idx", end_idx)
                # forward propagation
                output = batch_x
                # print("input shape", np.shape(output))
                
                for layer in self.layers:
                    output = layer.forward_propagation(output)

                # print("output shape", output.shape)
                loss = self.loss(batch_y, output)
                
                # backward propagation
                gradient = self.loss_prime(batch_y, output)
                for layer in reversed(self.layers):
                    gradient = layer.backward_propagation(gradient, learning_rate)
                
                # if batch_idx % print_interval == 0:
                #      print('epoch %d/%d batch: %d error:%f' % (i+1, epochs, batch_idx,loss))
                # batch_idx += 1

            # evaluate when getting evaluation dataset
            if evaluation != None:
                (test_x, test_y) = evaluation
                preds_y = self.predict(test_x)
                accuracy = sum([y_ == y for y_, y in zip(preds_y, test_y)])/test_x.shape[0] * 100
                if accuracy - last_accuacy < 1:
                    learning_rate = learning_rate/10.0
                last_accuacy = accuracy
                print('epoch %d/%d lr: %f; accuracy error:%f' % (i+1, epochs, learning_rate, accuracy))
                

