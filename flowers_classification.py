import glob
import os
import shutil

import tensorflow as tf

_URL = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"

zip_file = tf.keras.utils.get_file(origin=_URL,
                                   fname="flower_photos.tgz",
                                   extract=True)

base_dir = os.path.join(os.path.dirname(zip_file), 'flower_photos')
classes = ['roses', 'daisy', 'dandelion', 'sunflowers', 'tulips']
try:
    for cl in classes:
        img_path = os.path.join(base_dir, cl)
        images = glob.glob(img_path + '/*.jpg')
        print("{}: {} Images".format(cl, len(images)))
        num_train = int(round(len(images) * 0.8))
        train, val = images[:num_train], images[num_train:]

        for t in train:
            if not os.path.exists(os.path.join(base_dir, 'train', cl)):
                os.makedirs(os.path.join(base_dir, 'train', cl))
            shutil.move(t, os.path.join(base_dir, 'train', cl))

        for v in val:
            if not os.path.exists(os.path.join(base_dir, 'val', cl)):
                os.makedirs(os.path.join(base_dir, 'val', cl))
            shutil.move(v, os.path.join(base_dir, 'val', cl))
except Exception as e:
    pass
train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'val')

batch_size = 64
IMG_SHAPE = 180

image_gen_train = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=45,
    width_shift_range=.15,
    height_shift_range=.15,
    horizontal_flip=True,
    zoom_range=0.5
)

train_data_gen = image_gen_train.flow_from_directory(
    batch_size=batch_size,
    directory=train_dir,
    shuffle=True,
    target_size=(IMG_SHAPE, IMG_SHAPE),
    class_mode='sparse'
)
image_gen_val = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)

val_data_gen = image_gen_val.flow_from_directory(batch_size=batch_size,
                                                 directory=val_dir,
                                                 target_size=(IMG_SHAPE, IMG_SHAPE),
                                                 class_mode='sparse')
inputs = tf.keras.layers.Input(shape=(180, 180, 3))
x = tf.keras.layers.Conv2D(32, 5, use_bias=False, input_shape=(180, 180, 3))(inputs)
for size in [32, 64, 128, 256, 512]:
    residual = x
    for i in range(2):
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation(tf.nn.relu)(x)
        x = tf.keras.layers.SeparableConv2D(size, 3, use_bias=False, padding="same")(x)
    x = tf.keras.layers.MaxPooling2D(3, strides=2, padding="same")(x)

    residual = tf.keras.layers.Conv2D(size, 1, padding="same", use_bias=False)(x)
    x = tf.keras.layers.add([x, residual])

x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.5)(x)

outputs = tf.keras.layers.Dense(5, activation=tf.nn.softmax)(x)
model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
callback = [tf.keras.callbacks.ModelCheckpoint("flowers_classification.tf", save_best_only=True)]
model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
model.fit(train_data_gen, epochs=150, validation_data=val_data_gen, callbacks=callback)
