import tensorflow as tf
import numpy as np

(train_data, train_targets), (test_data, test_targets) = tf.keras.datasets.boston_housing.load_data()
mean = train_data.mean(axis=0)  # среднее значение каждого признака (столбца) в тренировочных данных.
train_data -= mean  # Эта строка вычитает среднее значение из тренировочных данных, центрируя их вокруг нуля.
std = train_data.std(axis=0)  # Эта строка вычисляет стандартное отклонение каждого признака в тренировочных данных.
train_data /= std  # Эта строка делит тренировочные данные на стандартное отклонение, масштабируя их для получения единичной дисперсии.
test_data -= mean
test_data /= std


def build_model():
    model = tf.keras.models.Sequential([tf.keras.layers.Dense(units=64, activation=tf.nn.relu),
                                        tf.keras.layers.Dense(units=64, activation=tf.nn.relu),
                                        tf.keras.layers.Dense(units=1)])
    model.compile(optimizer="rmsprop",
                  loss="mse",
                  metrics=["mae"])
    return model


k = 4
num_val_samples = len(train_data) // k
num_epochs = 130
all_mae_histories = []
for i in range(k):
    print(f"Processing fold #{i}")
    val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]
    val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]
    partial_train_data = np.concatenate([train_data[:i*num_val_samples], train_data[(i+1) * num_val_samples:]], axis=0)
    partial_train_targets = np.concatenate([train_targets[:i*num_val_samples], train_targets[(i+1) * num_val_samples:]], axis=0)
    model = build_model()
    history = model.fit(partial_train_data, partial_train_targets, validation_data=(val_data, val_targets), epochs=num_epochs, batch_size=16, verbose=0)
    mae_history = history.history["val_mae"]
    all_mae_histories.append(mae_history)

print(all_mae_histories)