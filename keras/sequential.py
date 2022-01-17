import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

# load data set
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(x_train.shape)

# Flatten the dataset for NN consumption
# Original shape: (60000, 28, 28)
# -1: keep the value of dimension 0
# 28*28 = (dimension 1 * dimension 2)
# astype("float32") = minimize the complication of the data
# /255.0 = normalize values by lowering the range, for faster training ( 0-255 => 0-1 )
x_train = x_train.reshape(-1, 28 * 28).astype("float32") / 255.0
x_test = x_test.reshape(-1, 28 * 28).astype("float32") / 255.0
print(x_train.shape)

# Soft activation, can be automated later
# x_train = tf.convert_to_tensor(x_train)

# Create model: Sequential API
# 1 input - 1 output
model = keras.Sequential(
    [
        keras.Input(shape=(28*28)),  # Allows use of model.summary(), will be created @ model training
        # list of layers
        layers.Dense(512, activation='relu'),  # nodes, activation function
        layers.Dense(256, activation='relu'),
        layers.Dense(10)
    ]
)

# Show nn information
print(model.summary())

# Tell keras how to train the model
model.compile(
    # Loss function to evaluate performance
    # from_logits = True => added because there was no soft-max activation in the model creation
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=keras.optimizers.Adam(lr=0.001),
    metrics=['accuracy']
)

# Training of the nn
model.fit(x_train, y_train, batch_size=32, epochs=6, verbose=2)  # verbose=2: prints after each epoch

# Evaluate nn
model.evaluate(x_test, y_test, batch_size=32, verbose=2)
