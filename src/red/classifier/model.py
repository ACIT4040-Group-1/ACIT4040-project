from tensorflow.keras.layers import Dense, Dropout
from tensorflow import keras
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import tensorflow as tf


def get_model():
    dense_net = DenseNet121(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3)
    )
    dense_net.trainable = False
    model = tf.keras.Sequential()

    model.add(dense_net)
    model.add(keras.layers.GlobalAveragePooling2D())

    for _ in range(2):
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(rate=0.5))

    model.add(Dense(units=1, activation='sigmoid'))

    model.compile(optimizer=Adam(),
                  loss='binary_crossentropy',
                  metrics='accuracy')
    return model
