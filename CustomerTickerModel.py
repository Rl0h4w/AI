import tensorflow as tf


class CustomerTickerModel(tf.keras.Model):

    def __init__(self, num_departments):
        super().__init__()
        self.concat_layer = tf.keras.layers.Concatenate()
        self.mixing_layer = tf.keras.layers.Dense(64, activation=tf.nn.relu)
        self.priority_scorer = tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)
        self.department_classifier = tf.keras.layers.Dense(num_departments, activation=tf.nn.softmax)

    def call(self, inputs):
        title = inputs["title"]
        text_body = inputs["text_body"]
        tags = inputs["tags"]

        features = self.concat_layer([title, text_body, tags])
        features = self.mixing_layer(features)

        priority = self.priority_scorer(features)
        department = self.department_classifier(features)

        return priority, department

