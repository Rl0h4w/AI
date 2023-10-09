import numpy as np
import tensorflow as tf

(train_data, train_labels), (test_data, test_labels) = tf.keras.datasets.imdb.load_data(num_words=10_000)

word_index = tf.keras.datasets.imdb.get_word_index()
reversed_word_index = {value: key for key, value in word_index.items()}
decoded_review = " ".join([reversed_word_index.get(i - 3, "?") for i in train_data[0]])


def vectorize_sequences(sequences, dimension=10_000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        for j in sequence:
            results[i, j] = 1
    return results


x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

y_train = np.asarray(train_labels).astype("float32")
y_test = np.asarray(test_labels).astype("float32")

model = tf.keras.Sequential([tf.keras.layers.Dense(512, tf.nn.relu),
                             tf.keras.layers.Dense(1, tf.nn.sigmoid)])
model.compile(optimizer="rmsprop",
              loss="binary_crossentropy",
              metrics=["accuracy"])

x_val = x_train[:10_000]
partial_x_train = x_train[10_000:]
y_val = y_train[:10_000]
partial_y_train = y_train[10_000:]

history = model.fit(partial_x_train, partial_y_train, epochs=4, batch_size=512, validation_data=(x_val, y_val))

results = model.evaluate(x_test, y_test)
print(results)