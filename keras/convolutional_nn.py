import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import cifar10

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

# load data set
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print(x_train.shape)

# astype("float32") = minimize the complication of the data
# /255.0 = normalize values by lowering the range, for faster training ( 0-255 => 0-1 )
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0
print(x_train.shape)


# Soft activation, can be automated later
# x_train = tf.convert_to_tensor(x_train)

# Create model: Sequential API
# model = keras.Sequential(
#     [
#         keras.Input(shape=(32, 32, 3)),  # Image size: 32*32 | 3=color image/RGB
#         # list of layers
#         # 32 output channels
#         # kernel size: 3 | (3, 3)
#         # convolutional padding: 'same' will keep 32*32 | 'valid' will output 30*30
#         layers.Conv2D(32, 3, padding='valid', activation='relu'),
#         layers.MaxPooling2D(pool_size=(2, 2)),  # half pool size: 15*15
#         layers.Conv2D(64, 3, activation='relu'),
#         layers.MaxPooling2D(),
#         layers.Conv2D(128, 3, activation='relu'),
#         layers.Flatten(),
#         layers.Dense(64, activation='relu'),
#         layers.Dense(10)
#     ]
# )

# Create model: Functional API
def conv_model():
    inputs = keras.Input(shape=(32, 32, 3))
    x = layers.Conv2D(32, 3)(inputs)
    x = layers.BatchNormalization()(x)
    x = keras.activations.relu(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, 5, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = keras.activations.relu(x)
    x = layers.Conv2D(128, 3)(x)
    x = layers.BatchNormalization()(x)
    x = keras.activations.relu(x)
    x = layers.Flatten()(x)
    x = layers.Dense(64, activation='relu')(x)
    outputs = layers.Dense(10)(x)
    x_model = keras.Model(inputs=inputs, outputs=outputs)
    return x_model


model = conv_model()

# Show nn information
print(model.summary())

# Tell keras how to train the model
model.compile(
    # Loss function to evaluate performance
    # from_logits = True => added because there was no soft-max activation in the model creation
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=keras.optimizers.Adam(lr=3e-4),
    metrics=['accuracy']
)

# Training of the nn
model.fit(x_train, y_train, batch_size=64, epochs=10, verbose=2)  # verbose=2: prints after each epoch

# Evaluate nn
model.evaluate(x_test, y_test, batch_size=64, verbose=2)
