import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(-1, 28*28).astype("float32") / 255.0
x_test = x_test.reshape(-1, 28*28).astype("float32") / 255.0

# Sequential Model
model_sequential = keras.Sequential(
    [
        keras.Input(shape=(28*28)),
        layers.Dense(512, activation='relu'),
        layers.Dense(256, activation='relu'),
        layers.Dense(10)
    ]
)

# Functional Model
def new_model():
    inputs = keras.Input(shape=(28 * 28))
    x = layers.Dense(512, activation='relu')(inputs)
    x = layers.Dense(256, activation='relu')(x)
    outputs = layers.Dense(10)(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model

model_functional = new_model()

# Compile models with the same settings
for model in [model_sequential, model_functional]:
    model.compile(
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        optimizer=keras.optimizers.RMSprop(learning_rate=0.001),
        metrics=["accuracy"],
    )

# Train both models
history_sequential = model_sequential.fit(x_train, y_train, batch_size=32, epochs=15, verbose=1)
history_functional = model_functional.fit(x_train, y_train, batch_size=32, epochs=15, verbose=1)

# Evaluate both models
eval_sequential = model_sequential.evaluate(x_test, y_test, batch_size=32, verbose=2)
eval_functional = model_functional.evaluate(x_test, y_test, batch_size=32, verbose=2)

print("Sequential Model Accuracy:", eval_sequential[1])
print("Functional API Model Accuracy:", eval_functional[1])
