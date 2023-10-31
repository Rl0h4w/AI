import tensorflow as tf
import numpy as np

vocabulary_size = 10_000
num_tags = 100
num_departments = 4

title = tf.keras.Input(shape=(vocabulary_size,), name="title")
text_body = tf.keras.Input(shape=(vocabulary_size,), name="text_body")
tags = tf.keras.Input(shape=(num_tags,), name="tags")

features = tf.keras.layers.Concatenate()([title, text_body, tags])
features = tf.keras.layers.Dense(64, activation=tf.nn.relu)(features)
priority = tf.keras.layers.Dense(1, tf.nn.sigmoid, name="priority")(features)
department = tf.keras.layers.Dense(num_departments, tf.nn.softmax, name="department")(features)
model = tf.keras.Model(inputs=[title, text_body, tags], outputs=[priority, department])

num_samples = 100

title_data = np.random.randint(0, 2, size=(num_samples, vocabulary_size))
text_body_data = np.random.randint(0, 2, size=(num_samples, vocabulary_size))
tags_data = np.random.randint(0, 2, size=(num_samples, num_tags))

priority_data = np.random.random(size=(num_samples, 1))
department_data = np.random.randint(0, 2, size=(num_samples, num_departments))

model.compile(optimizer="rmsprop",
              loss={"priority": tf.keras.losses.mae,
                    "department": tf.keras.losses.categorical_crossentropy},
              metrics={"priority": [tf.keras.metrics.mae],
                       "department": [tf.keras.metrics.categorical_accuracy]})
model.fit({"title": title_data, "text_body": text_body_data, "tags": tags_data},
          {"priority": priority_data, "department": department_data}, epochs=1)
model.evaluate({"title": title_data, "text_body": text_body_data, "tags": tags_data},
          {"priority": priority_data, "department": department_data})
priority_preds, department_preds = model.predict({"title": title_data, "text_body": text_body_data, "tags": tags_data})
