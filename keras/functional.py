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


# Create model: Functional API
# multiple inputs - multiple outputs
inputs = keras.Input(shape=(784))
x = layers.Dense(512, activation='relu', name='first_layer')(inputs)  # initialize layer, then call it with the inputs
x = layers.Dense(512, activation='relu', name='second_layer')(x)
# x = layers.Dense(512, activation='relu', name='third_layer')(x)
# x = layers.Dense(512, activation='relu', name='fourth_layer')(x)
outputs = layers.Dense(10, activation='softmax', name='output_layer')(x)
model = keras.Model(inputs=inputs, outputs=outputs)

# Show nn information
print(model.summary())

# SGD optimizer options
# lr_schedule = keras.optimizers.schedules.ExponentialDecay(
#     initial_learning_rate=1e-2,
#     decay_steps=10000,
#     decay_rate=0.9)

# Tell keras how to train the model
model.compile(
    # Loss function to evaluate performance
    loss=keras.losses.SparseCategoricalCrossentropy(),
    optimizer=keras.optimizers.Adam(),
    # optimizer=keras.optimizers.RMSprop(learning_rate=0.001),
    # optimizer=keras.optimizers.SGD(learning_rate=lr_schedule),
    metrics=['accuracy']
)

# Training of the nn
model.fit(x_train, y_train, batch_size=32, epochs=6, verbose=2)  # verbose=2: prints after each epoch

# Evaluate nn
model.evaluate(x_test, y_test, batch_size=32, verbose=2)
