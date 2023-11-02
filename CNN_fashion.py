import tensorflow as tf

(train_data, train_labels), (test_data, test_labels) = tf.keras.datasets.fashion_mnist.load_data()

train_data = train_data.astype("float32") / 255.
test_data = test_data.astype("float32") / 255.
inputs = tf.keras.layers.Input(shape=(28, 28, 1))
x = tf.keras.layers.Conv2D(32, (3, 3), input_shape=(28, 28, 1), activation=tf.nn.relu, padding='valid')(inputs)
x = tf.keras.layers.MaxPooling2D(padding="same")(x)
x = tf.keras.layers.Conv2D(64, (3, 3), activation=tf.nn.relu, padding='valid')(x)
x = tf.keras.layers.MaxPooling2D(padding="same")(x)
x = tf.keras.layers.Conv2D(128, (3, 3), activation=tf.nn.relu, padding='valid')(x)
x = tf.keras.layers.Flatten()(x)
output = tf.keras.layers.Dense(10, activation=tf.nn.softmax)(x)
model = tf.keras.Model(inputs=inputs, outputs=output)
model.summary()
model.compile(optimizer=tf.keras.optimizers.RMSprop(), loss=tf.keras.losses.sparse_categorical_crossentropy,
              metrics=[tf.keras.metrics.sparse_categorical_accuracy])
model.fit(train_data, train_labels, epochs=6, validation_data=(test_data, test_labels))
