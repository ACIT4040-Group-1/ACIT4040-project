import tensorflow as tf
from keras import Sequential
from keras.layers import Dense, Dropout
from src.common.utils import get_config
from config import config
from keras.layers.pooling import GlobalAveragePooling2D


def get_exampleNet():
    net_config = config['models']['exampleNet']
    pretrained_model = tf.keras.applications.DenseNet121(
        input_shape=net_config['input_shape'],
        include_top=False,
        weights='imagenet',
        pooling='avg')
    pretrained_model.trainable = False
    model = Sequential()
    model.add(pretrained_model)
    model.add(Dense(units=32, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(units=32, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(units=1, activation='sigmoid'))
    optimizer = tf.keras.optimizers.Adam()

    model.compile(optimizer=optimizer,
                  loss=net_config['loss'],
                  metrics=net_config['metrics'])
    return model


def get_akselnet():
    net_config = config['models']['akselnet']
    pretrained_model = tf.keras.applications.DenseNet121(
        input_shape=net_config['input_shape'],
        include_top=False,
        weights='imagenet'
        #pooling='avg'
    )
    pretrained_model.trainable = False
    model = Sequential()
    model.add(pretrained_model)
    model.add(GlobalAveragePooling2D())
    model.add(Dense(units=256, activation='relu'))
    model.add(Dropout(rate=0.5))
    model.add(Dense(units=512, activation='relu'))
    model.add(Dropout(rate=0.5))
    model.add(Dense(units=256, activation='relu'))
    model.add(Dropout(rate=0.5))
    model.add(Dense(units=1, activation='sigmoid'))
    optimizer = tf.keras.optimizers.Adam()
    print(net_config['loss'])
    print(net_config['metrics'])
    model.compile(optimizer=optimizer,
                  loss=net_config['loss'],
                  metrics=net_config['metrics'])
    return model
