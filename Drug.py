import random

import numpy as np
import tensorflow as tf


def get_data():
    with open("datasets/Drugs/drug200.csv") as file:
        names = file.readline().split()
        train = np.zeros((200, 5))
        labels = np.zeros((200, 1))
        arr = file.readlines()
        random.shuffle(arr)
        for i, line in enumerate(arr):
            age, sex, bp, ch, na, drug = line.strip().split(',')
            age = int(age)
            sex = 0 if sex == "F" else 1
            bp = -1 if bp == "LOW" else 0 if bp == "NORMAL" else 1
            ch = 0 if ch == "NORMAL" else 1
            na = float(na)
            drug = {"druga": 0, "drugb": 1, "drugc": 2, "drugx": 3, "drugy": 4}[drug.lower()]
            train[i] = age, sex, bp, ch, na
            labels[i] = drug
        mean = train.mean(axis=0)
        train -= mean
        std = train.std(axis=0)
        train /= std
        return train, labels


def get_model():
    inputs = tf.keras.Input(shape=(5,))
    x = tf.keras.layers.Dense(32, activation="relu")(inputs)
    x = tf.keras.layers.Dense(32, activation="relu")(x)
    outputs = tf.keras.layers.Dense(5, activation="softmax")(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs)


model = get_model()
model.compile(optimizer="rmsprop", loss=tf.keras.losses.sparse_categorical_crossentropy,
              metrics=tf.keras.metrics.sparse_categorical_accuracy)
x, y = get_data()
x_train = x[:150]
y_train = y[:150]
x_val = x[150:]
y_val = y[150:]
history = model.fit(x=x_train, y=y_train, validation_data=(x_val, y_val),
                    callbacks=[tf.keras.callbacks.ModelCheckpoint(filepath="Drug.tf", save_best_only=True)], epochs=130)
