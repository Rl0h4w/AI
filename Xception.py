import tensorflow as tf

inputs = tf.keras.layers.Input(shape=(180, 180, 3))
x = tf.keras.layers.Rescaling(1. / 255)(inputs)
x = tf.keras.layers.Conv2D(32, 5, use_bias=False)(x)

for size in [2 ** i for i in range(5, 10)]:
    residual = x
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation(tf.nn.relu)(x)
    x = tf.keras.layers.SeparableConv2D(size, 3, padding="same", use_bias=False)(x)

    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation(tf.nn.relu)(x)
    x = tf.keras.layers.SeparableConv2D(size, 3, padding="same", use_bias=False)(x)

    x = tf.keras.layers.MaxPooling2D(3, strides=2, padding="same")(x)
    residual = tf.keras.layers.Conv2D(size, 1, padding="same", use_bias=False)(residual)
    x = tf.keras.layers.add([x, residual])

x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.5)(x)
outputs = tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)(x)
model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
