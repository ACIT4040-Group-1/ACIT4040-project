import tensorflow as tf
from keras import Sequential
from keras.layers import Dense, Dropout

from blue.source.main import config


def get_akselnet():
    akselnet = config['models']['akselnet']
    pretrained_model = tf.keras.applications.DenseNet121(
        input_shape=akselnet['input_shape'],
        include_top=False,
        weights='imagenet',
        pooling='avg')
    model = Sequential()
    model.add(pretrained_model)
    model.add(Dense(units=128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(units=64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(units=1, activation='sigmoid'))
    optimizer = tf.keras.optimizers.Adam()
    model.compile(optimizer=optimizer,
                  loss='binary_crossentropy',
                  metrics='accuracy')
    return model
