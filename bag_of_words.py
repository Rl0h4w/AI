import tensorflow as tf


def text_to_bag_of_words(max_tokens, train_ds, val_ds, test_ds):
    text_vectorization = tf.keras.layers.TextVectorization(max_tokens=max_tokens, output_mode="multi_hot")
    text_only_train_ds = train_ds.map(lambda x, y: x)
    text_vectorization.adapt(text_only_train_ds)

    binary_1gram_train_ds = train_ds.map(lambda x, y: (text_vectorization(x), y), num_paralleel_calls=4)
    binary_1gram_val_ds = val_ds.map(lambda x, y: (text_vectorization(x), y), num_paralleel_calls=4)
    binary_1gram_test_ds = test_ds.map(lambda x, y: (text_vectorization(x), y), num_paralleel_calls=4)
