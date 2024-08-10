import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype("float32") / 225.0
x_test = x_test.astype("float32") / 225.0

model = keras.Sequential()
model.add(keras.Input(shape=(None, 28)))
model.add(
    layers.Bidirectional(
        layers.LSTM(256, return_sequences=True, activation='tanh')
    )
)
model.add(
    layers.Bidirectional(
        layers.LSTM(256, activation='tanh')
    )
) 
model.add(layers.Dense(10))

print(model.summary())

model.compile(
    optimizer=keras.optimizers.Adam(),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

model.fit(x_train, y_train, epochs=10, batch_size=64, verbose=1)
model.evaluate(x_test, y_test, batch_size=64, verbose=2)

