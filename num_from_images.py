import math

import numpy as np
import tensorflow as tf


class NativeDense:
    def __init__(self, input_size, output_size, activation):
        self.activation = activation  # функция активации
        w_shape = (input_size, output_size)  # размерность матрицы весов
        w_initial_value = tf.random.uniform(shape=w_shape, minval=0,
                                            maxval=1e-1)  # случайные значения по нормальному распределению
        self.W = tf.Variable(w_initial_value)  # создание матрицы с переменными значениями весов

        b_shape = (output_size,)  # размерность вектора
        b_initial_value = tf.zeros(b_shape)  # нули вектора
        self.b = tf.Variable(b_initial_value)

    def __call__(self, inputs):
        return self.activation(tf.matmul(inputs, self.W) + self.b)  # Произведение весов с биасом

    @property
    def weights(self):
        return [self.W, self.b]


class NativeSequential:
    def __init__(self, layers):
        self.layers = layers

    def __call__(self, inputs):
        x = inputs
        for layer in self.layers:  # итерация по слоям с вызовом обучающего метода
            x = layer(x)
        return x

    @property
    def weights(self):
        weights = []
        for layer in self.layers:
            weights += layer.weights
        return weights


class BatchGenerator:
    def __init__(self, images, labels, batch_size=128):
        assert len(images) == len(labels)
        self.index = 0
        self.images = images
        self.labels = labels
        self.batch_size = batch_size
        self.num_batches = math.ceil(len(images) / batch_size)

    def next(self):
        images = self.images[self.index: self.index + self.batch_size]
        labels = self.labels[self.index: self.index + self.batch_size]
        self.index += self.batch_size
        return images, labels


def update_weights(gradients, weights):
    optimizer = tf.keras.optimizers.SGD(learning_rate=1e-3)  # оптимизатор
    optimizer.apply_gradients(zip(gradients, weights))  # применение к весам


def one_training_step(model, images_batch, labels_batch):
    with tf.GradientTape() as tape:
        predictions = model(images_batch)  # вызов модели для обучения одной эпохи
        per_sample_losses = tf.keras.losses.sparse_categorical_crossentropy(y_true=labels_batch,
                                                                            y_pred=predictions)  # потери в виде вектора
        average_loss = tf.reduce_mean(per_sample_losses)  # усреднение потерь
    gradients = tape.gradient(target=average_loss, sources=model.weights)  # вычисление градиента
    update_weights(gradients, model.weights)  # обновление весов
    return average_loss


def fit(model, images, labels, epochs, batch_size=128):
    for epoch_counter in range(epochs):
        print(f"Epoch {epoch_counter}")
    batch_generator = BatchGenerator(images, labels, batch_size) #генерирование батча
    for batch_counter in range(batch_generator.num_batches):
        images_batch, labels_batch = batch_generator.next()
        loss = one_training_step(model, images_batch, labels_batch) #вычисление потерь
        if batch_counter % 100 == 0:
            print(f"loss at batch {batch_counter}: {loss:.2f}")


model = NativeSequential([NativeDense(28 * 28, 512, tf.nn.relu), NativeDense(512, 10, tf.nn.softmax)])

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype("float32") / 255
test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype("float32") / 255

fit(model, train_images, train_labels, 10, 128)

predictions = model(test_images)
predictions = predictions.numpy()
predicted_labels = np.argmax(predictions, axis=1)
matches = predicted_labels == test_labels
print(f"accuracy: {matches.mean():.2f}")
