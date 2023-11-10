import numpy as np
import tensorflow as tf

with open("datasets/Weather/jena_climate_2009_2016.csv") as file:
    data = file.read()

lines = data.split('\n')
header = lines[0].split(',')
lines = lines[1:]

temperature = np.zeros((len(lines, )))
raw_data = np.zeros((len(lines), len(header) - 1))
for i, line in enumerate(lines):
    values = [float(x) for x in line.split(',')[1:]]
    temperature[i] = values[1]
    raw_data[i, :] = values[:]

num_train_samples = int(len(raw_data) * 0.8)
num_val_samples = int(len(raw_data) * 0.1)
num_test_samples = len(raw_data) - num_val_samples - num_train_samples

mean = raw_data[:].mean(axis=0)
raw_data -= mean
std = raw_data[:].std(axis=0)
raw_data /= std

sampling_rate = 6
sequence_length = 120
delay = sampling_rate * (sequence_length + 24 - 1)
batch_size = 256

train_dataset = tf.keras.utils.timeseries_dataset_from_array(
    raw_data[:-delay],
    targets=temperature[delay:],
    sampling_rate=sampling_rate,
    sequence_length=sequence_length,
    shuffle=True,
    batch_size=batch_size,
    start_index=0,
    end_index=num_train_samples
)
val_dataset = tf.keras.utils.timeseries_dataset_from_array(
    raw_data[:-delay],
    targets=temperature[delay:],
    sampling_rate=sampling_rate,
    sequence_length=sequence_length,
    shuffle=True,
    batch_size=batch_size,
    start_index=0,
    end_index=num_train_samples
)

test_dataset = tf.keras.utils.timeseries_dataset_from_array(
    raw_data[:-delay],
    targets=temperature[delay:],
    sampling_rate=sampling_rate,
    sequence_length=sequence_length,
    shuffle=True,
    batch_size=batch_size,
    start_index=0,
    end_index=num_train_samples
)

inputs = tf.keras.layers.Input(shape=(sequence_length, raw_data.shape[-1]))
x = tf.keras.layers.LSTM(32, recurrent_dropout=0.5)(inputs)
x = tf.keras.layers.Dropout(0.5)(x)
outputs = tf.keras.layers.Dense(1)(x)

model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
callbacks = [tf.keras.callbacks.ModelCheckpoint("jena_lstm.tf", save_best_only=True)]
model.compile(optimizer="rmsprop", loss="mse", metrics=["mae"])
model.fit(train_dataset, epochs=100, validation_data=val_dataset, callbacks=callbacks)
