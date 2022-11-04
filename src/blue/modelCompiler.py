import keras.optimizers
import tensorflow as tf
from keras import Sequential, Model
from keras.layers import Dense, Dropout, concatenate, Input
from keras.models import load_model
from config import config
from keras.layers.pooling import GlobalAveragePooling2D
import ssl

ssl._create_default_https_context = ssl._create_unverified_context


def add_prefix(in_model, prefix: str):
    model = in_model
    model._name = prefix + model.name
    for layer in model.layers:
        layer._name = prefix + layer.name

    return model


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
        # pooling='avg'
    )
    pretrained_model.trainable = False
    model = Sequential()
    model.add(pretrained_model)
    model.add(GlobalAveragePooling2D())
    model.add(Dense(units=256, activation='relu'))
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


def get_res_net151_detector():
    net_config = config['models']['resNet50_detector']
    pretrained_model = tf.keras.applications.ResNet152V2(
        input_shape=net_config['input_shape'],
        include_top=False,
        weights='imagenet'
        # pooling='avg'
    )
    pretrained_model.trainable = False
    model = Sequential()
    model.add(pretrained_model)
    model.add(GlobalAveragePooling2D())
    model.add(Dense(units=256, activation='relu'))
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


def get_maryamnet():
    net_config = config['models']['maryamnet']
    pretrained_model = tf.keras.applications.ResNet50(
        input_shape=net_config['input_shape'],
        include_top=False,
        weights='imagenet'
        # pooling='avg'
    )
    pretrained_model.trainable = False
    model = Sequential()
    model.add(pretrained_model)
    model.add(GlobalAveragePooling2D())
    # classifier neural network
    model.add(Dense(units=256, activation='relu'))  # hidden layer
    model.add(Dropout(rate=0.5))
    model.add(Dense(units=512, activation='relu'))
    model.add(Dropout(rate=0.5))
    model.add(Dense(units=256, activation='relu'))  # hidden layer
    model.add(Dropout(rate=0.5))
    model.add(Dense(units=1, activation='sigmoid'))  # output layer
    optimizer = tf.keras.optimizers.Adam()
    print(net_config['loss'])
    print(net_config['metrics'])
    model.compile(optimizer=optimizer,
                  loss=net_config['loss'],
                  metrics=net_config['metrics'])
    return model


def get_combined_model():
    inputs = Input(shape=(None, None, 3), name='input')

    net_config = config['models']['combined']
    modelA = load_model('src/blue/models/resNet151_detector.h5')
    modelA.trainable = False
    modelA = add_prefix(modelA, 'modelA_')
    modelA = modelA(inputs)
    #print(f"modelA summary: ")
    #modelA.summary()


    modelB = load_model('src/blue/models/denseNet121_detector.h5')
    modelB.trainable = False
    modelB=add_prefix(modelB,'modelB_')
    modelB = modelB(inputs)
    #print(modelB.inputs)
    #print(f"modelB summary: ")
    #modelB.summary()


    # concatinated branch:
    combined = concatenate([modelA, modelB])

    # combined outputs
    y = Dense(2, activation="relu")(combined)
    y = Dense(1, activation="sigmoid")(y)

    model = Model(inputs=inputs, outputs=y)

    optimizer = keras.optimizers.Adam()
    print(net_config['loss'])
    print(net_config['metrics'])
    model.compile(optimizer=optimizer,
                  loss=net_config['loss'],
                  metrics=net_config['metrics'])
    return model

if __name__ == "__main__":
    model = get_combined_model()
    print('combined model summary:')
    print(model.summary())
