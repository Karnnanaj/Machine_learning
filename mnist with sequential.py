# grayscale image recognition model with mnist dataset
# runs with sequential API, which supports only one input and one output
# the input image is flattened to an array

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

physical_devices = tf.config.list_physical_devices('GPU')
print("Num GPUs Available: ", len(physical_devices))

"""
# #Perform a simple operation to see if it uses the GPU
# matrix1 = tf.random.uniform((10000, 10000))
# matrix2 = tf.random.uniform((10000, 10000))
# result = tf.matmul(matrix1, matrix2)
# print(result)
"""

(x_train, y_train), (x_test, y_test) = mnist.load_data()

"""
# for mnist dataset
#just for showing the dimension of the dataset
print(x_train.shape)
print(y_train.shape)
"""

# conversion of image (2d array) into a 1d array
x_train = x_train.reshape(-1, 28*28).astype("float32") / 255.0
x_test = x_test.reshape(-1, 28*28).astype("float32") / 255.0

# defining model method 1
model = keras.Sequential(
    [
        keras.Input(shape=(28*28)),             #flatten the image/input
        layers.Dense(512, activation='relu'),
        layers.Dense(256, activation='relu'),
        layers.Dense(10, activation='softmax', name='output_layer')
    ]
)
"""
# defining model method 2
model = keras.Sequential()
model.add(keras.Input(shape=(784)))
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(256, activation='relu', name='my_layer'))
model.add(layers.Dense(10))


import sys
sys.exit() #to terminate a program

model = keras.Model(inputs= model.inputs,
                  outputs=[layer.output for layer in model.layers])

features = model.predict(x_train)
for feature in features:
    print(feature.shape)
"""

print(model.summary())

model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    optimizer=keras.optimizers.RMSprop(learning_rate=0.001),
    metrics=["accuracy"],
)

# this starts the model training, epochs are the number of training cycles
model.fit(x_train, y_train, batch_size=32, epochs=15, verbose=1)
# to test the model with unseen data
model.evaluate(x_test, y_test, batch_size=32, verbose=2)