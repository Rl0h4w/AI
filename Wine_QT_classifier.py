import numpy as np
import tensorflow as tf


def get_data():
    with open("datasets/WineQT/WineQT.csv") as file:
        labels = file.readline()
        features = np.zeros(shape=(1143, 11))
        labels = np.zeros(shape=(1143,))
        for i, line in enumerate(file.readlines()):
            data = [float(i) for i in line.split(",")]
            features[i] = data[:-2]
            labels[i] = data[-2]
        mean = features.mean(axis=0)
        features -= mean
        std = features.std(axis=0)
        features /= std
        return features, labels


features, labels = get_data()
print(features[0], labels[0])
train_x = features[:750]
val_x = features[750:]
train_y = labels[:750]
val_y = labels[750:]

inputs = tf.keras.Input(shape=(11,))
x = tf.keras.layers.Dense(32, activation="relu")(inputs)
x = tf.keras.layers.Dense(32, activation="relu")(x)
outputs = tf.keras.layers.Dense(10, activation="softmax")(x)

model = tf.keras.Model(inputs=inputs, outputs=outputs)
model.compile(optimizer="rmsprop", loss=tf.keras.losses.sparse_categorical_crossentropy,
              metrics=[tf.keras.metrics.sparse_categorical_accuracy])
model.fit(x=train_x, y=train_y, validation_data=(val_x, val_y), batch_size=128, epochs=30,
          callbacks=[tf.keras.callbacks.ModelCheckpoint("Wine_QT_classifier.tf", save_best_only=True)])
