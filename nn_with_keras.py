from keras.datasets import mnist
from keras.utils import to_categorical
from keras import models
from keras import layers

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype('float32') / 255
test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype('float32') / 255

train_labels_one_hot = to_categorical(train_labels)
test_labels_one_hot = to_categorical(test_labels)

network = models.Sequential()
network.add(layers.Dense(200, activation='relu', input_shape=(28 * 28,)))
network.add(layers.Dense(50, activation='relu'))
network.add(layers.Dense(10, activation='softmax'))

network.compile(optimizer='rmsprop',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

network.fit(train_images, train_labels_one_hot, validation_data = (test_images, test_labels_one_hot), epochs=10, batch_size=128)